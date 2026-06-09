/**
 * specificity.cpp
 * ──────────────────
 * Calcul sparse C++ des indices de spécificité de Lafon (1980).
 * Version finale v4 — tous bugs corrigés, scores positifs ET négatifs,
 * parallélisation OpenMP sans data race.
 *
 * Compilation manuelle :
 *   c++ -std=c++17 -O3 -march=native -fopenmp -shared -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       specificity_v4.cpp \
 *       -o _specificity$(python3-config --extension-suffix)
 *
 * Via setup.py / pip install -e . (voir setup.py joint).
 *
 * Interface Python :
 *   from _specificity import compute_specificity_sparse
 *   rows_out, cols_out, specs_out = compute_specificity_sparse(
 *       rows,            # np.int32   — indices lignes COO
 *       cols,            # np.int32   — indices colonnes COO
 *       vals,            # np.float32 — cooccurrences brutes k
 *       row_sums,        # np.int64   — fréquences marginales lignes (= n)
 *       col_sums,        # np.int64   — fréquences marginales colonnes (= K)
 *       N,               # int        — total occurrences corpus
 *       clip_max=100.0,  # float      — écrêtage symétrique ± clip_max
 *       k_min=3,         # int        — ignore cooccurrences brutes < k_min
 *       spec_threshold=0.0  # float   — ignore |spec| ≤ seuil
 *   )
 *   # specs_out > 0 → sur-représentation  (attraction lexicale)
 *   # specs_out < 0 → sous-représentation (répulsion lexicale)
 */

#include "specificity.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────

py::tuple compute_specificity_sparse(
    py::array_t<int,       py::array::c_style | py::array::forcecast> rows_in,
    py::array_t<int,       py::array::c_style | py::array::forcecast> cols_in,
    py::array_t<float,     py::array::c_style | py::array::forcecast> vals_in,
    py::array_t<long long, py::array::c_style | py::array::forcecast> row_sums,
    py::array_t<long long, py::array::c_style | py::array::forcecast> col_sums,
    long long N,
    double    clip_max,
    long long k_min,
    double    spec_threshold
) {
    // ── Validation ────────────────────────────────────────────────────
    auto buf_r  = rows_in.request();
    auto buf_c  = cols_in.request();
    auto buf_v  = vals_in.request();
    auto buf_rs = row_sums.request();
    auto buf_cs = col_sums.request();

    const long long nnz = static_cast<long long>(buf_r.shape[0]);
    if (static_cast<long long>(buf_c.shape[0]) != nnz ||
        static_cast<long long>(buf_v.shape[0]) != nnz)
        throw std::invalid_argument("rows, cols, vals must have the same length");
    if (N <= 0)
        throw std::invalid_argument("N must be > 0");
    if (clip_max <= 0.0)
        throw std::invalid_argument("clip_max must be > 0");

    const int*       R  = static_cast<const int*      >(buf_r.ptr);
    const int*       C  = static_cast<const int*      >(buf_c.ptr);
    const float*     V  = static_cast<const float*    >(buf_v.ptr);
    const long long* RS = static_cast<const long long*>(buf_rs.ptr);
    const long long* CS = static_cast<const long long*>(buf_cs.ptr);

    // ── Nombre de threads ─────────────────────────────────────────────
    // BUG-3 corrigé : #pragma omp single pour éviter la data race.
    int n_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel default(none) shared(n_threads)
    {
        #pragma omp single
        { n_threads = omp_get_num_threads(); }
    }
#endif

    // ── Buffers par thread (zéro contention) ─────────────────────────
    std::vector<std::vector<int>>    out_rows(n_threads);
    std::vector<std::vector<int>>    out_cols(n_threads);
    std::vector<std::vector<double>> out_vals(n_threads);

    // Heuristique : ~20 % des nnz significatifs (deux signes)
    const size_t hint = static_cast<size_t>(nnz) / 5;
    for (int t = 0; t < n_threads; ++t) {
        out_rows[t].reserve(hint / static_cast<size_t>(n_threads) + 1);
        out_cols[t].reserve(hint / static_cast<size_t>(n_threads) + 1);
        out_vals[t].reserve(hint / static_cast<size_t>(n_threads) + 1);
    }

    // ── Boucle principale parallèle ───────────────────────────────────
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 512) \
        default(none) \
        shared(R, C, V, RS, CS, N, clip_max, k_min, spec_threshold, \
               nnz, out_rows, out_cols, out_vals)
#endif
    for (long long idx = 0; idx < nnz; ++idx) {
        const int       i = R[idx];
        const int       j = C[idx];
        const long long k = static_cast<long long>(V[idx]);

        // ── Filtres O(1) ──────────────────────────────────────────────
        if (i == j)    continue;   // diagonale : mot avec lui-même
        if (k < k_min) continue;   // cooccurrence trop rare

        const long long Fi = RS[i];   // n dans Lafon : nb occurrences contexte i
        const long long fj = CS[j];   // K dans Lafon : nb occurrences mot j
        if (Fi <= 0 || fj <= 0) continue;

        // ── Calcul hypergéométrique signé ─────────────────────────────
        //
        // log_hypergeom_sf_signed gère en interne :
        //   - choix queue droite (k ≥ μ) ou gauche (k < μ)
        //   - cas limites k ≤ 0 et k > kmax  (BUG-5 : kmax non dupliqué ici)
        //   - protection log(0) dans les récurrences  (BUG-1)
        //   - cast explicite dans lgamma_init           (BUG-2)
        const auto [log_p, sign] = log_hypergeom_sf_signed(k, N, fj, Fi);

        // ── Conversion en score Lafon signé ───────────────────────────
        //
        // ORDRE DES TESTS (BUG-4 corrigé) :
        //   1. !isfinite EN PREMIER : POS_INF ≥ 0 est vrai, donc tester
        //      log_p ≥ 0 avant isfinite faisait sauter le cas k > kmax.
        //   2. log_p ≥ 0 : p ≥ 1 → non significatif → on skippe.
        //   3. Cas normal : score = -sign · log_p · log10(e)
        double spec;
        if (!std::isfinite(log_p)) {
            // p = 0 exactement (k > kmax ou pmf_init non fini) → clip
            spec = static_cast<double>(sign) * clip_max;
        } else if (log_p >= 0.0) {
            // p ≥ 1 : non significatif
            continue;
        } else {
            // Formule Lafon : spec = -log10(p) signé
            //   sign=+1, log_p<0 → spec = -1 · log_p · log10e > 0  (sur-repr)
            //   sign=-1, log_p<0 → spec = +1 · log_p · log10e < 0  (sous-repr)
            spec = -static_cast<double>(sign) * log_p * LOG10_E_VAL;

            if (std::abs(spec) <= spec_threshold) continue;

            // Écrêtage symétrique
            if      (spec >  clip_max) spec =  clip_max;
            else if (spec < -clip_max) spec = -clip_max;
        }

        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        out_rows[tid].push_back(i);
        out_cols[tid].push_back(j);
        out_vals[tid].push_back(spec);
    }

    // ── Fusion ────────────────────────────────────────────────────────
    size_t total = 0;
    for (int t = 0; t < n_threads; ++t)
        total += out_rows[t].size();

    auto arr_rows = py::array_t<int>(   {static_cast<py::ssize_t>(total)});
    auto arr_cols = py::array_t<int>(   {static_cast<py::ssize_t>(total)});
    auto arr_spec = py::array_t<double>({static_cast<py::ssize_t>(total)});

    int*    pr = static_cast<int*>   (arr_rows.request().ptr);
    int*    pc = static_cast<int*>   (arr_cols.request().ptr);
    double* ps = static_cast<double*>(arr_spec.request().ptr);

    size_t offset = 0;
    for (int t = 0; t < n_threads; ++t) {
        const size_t sz = out_rows[t].size();
        std::copy(out_rows[t].begin(), out_rows[t].end(), pr + offset);
        std::copy(out_cols[t].begin(), out_cols[t].end(), pc + offset);
        std::copy(out_vals[t].begin(), out_vals[t].end(), ps + offset);
        offset += sz;
    }

    return py::make_tuple(arr_rows, arr_cols, arr_spec);
}

// ── Liaison PyBind11 ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_specificity, m) {
    m.doc() = R"pbdoc(
        Calcul C++ sparse des indices de spécificité de Lafon (1980). v4.

        Bugs corrigés vs versions précédentes
        ──────────────────────────────────────
        v2 : overflow long long dans la récurrence (produit → somme de logs)
        v3 : scores négatifs par tail inversion + récurrence descendante
             log_max corrigé dans le log-sum-exp
        v4 : log(0/négatif) dans les récurrences (test d ≤ 0 avant log)
             cast explicite long long→double dans lgamma (N > 2^53)
             data race OpenMP sur n_threads (#pragma omp single)
             ordre des tests log_p : !isfinite avant >= 0 (POS_INF >= 0 = true)
             kmax non dupliqué entre .cpp et .hpp

        Modèle
        ──────
        Hypergéométrique Lafon 1980.
          spec > 0 → sur-représentation  : P(X ≥ k) anormalement petite
          spec < 0 → sous-représentation : P(X ≤ k) anormalement petite
          Unité : -log10(p-valeur) signé.

        Algorithme
        ──────────
        Tail inversion + backward recurrence.
        Boucle interne ≤ kmax/2 itérations en moyenne.
        Parallélisation OpenMP, buffers par thread, zéro contention.
        Interface COO → COO, jamais de matrice V×V dense, O(nnz) mémoire.
    )pbdoc";

    m.def(
        "compute_specificity_sparse",
        &compute_specificity_sparse,
        py::arg("rows"),
        py::arg("cols"),
        py::arg("vals"),
        py::arg("row_sums"),
        py::arg("col_sums"),
        py::arg("N"),
        py::arg("clip_max")       = 100.0,
        py::arg("k_min")          = 3,
        py::arg("spec_threshold") = 0.0,
        R"pbdoc(
            Indices de spécificité hypergéométriques signés (Lafon 1980). v4.

            Paramètres
            ----------
            rows, cols : np.int32   — indices COO de la matrice de cooccurrence
            vals       : np.float32 — cooccurrences brutes k
            row_sums   : np.int64   — fréquences marginales lignes  (= n)
            col_sums   : np.int64   — fréquences marginales colonnes (= K)
            N          : int        — total des occurrences du corpus
            clip_max   : float      — valeur absolue maximale du score (défaut 100)
            k_min      : int        — cooccurrence brute minimale traitée (défaut 3)
            spec_threshold : float  — |spec| ≤ seuil → entrée non émise (défaut 0)

            Retourne
            --------
            (rows_out, cols_out, specs_out) — triplets COO numpy
              specs_out > 0 : sur-représentation  (attraction lexicale)
              specs_out < 0 : sous-représentation (répulsion lexicale)

            Complexité
            ----------
            Temps   : O(nnz · kmax/2) — tail inversion
            Mémoire : O(nnz) — jamais de matrice V×V
            Threads : OpenMP parallèle sur nnz, schedule(dynamic, 512)
        )pbdoc"
    );
}
