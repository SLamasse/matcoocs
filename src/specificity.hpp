#pragma once
/**
 *  specificity.hpp
 * ──────────────────
 * Calcul C++ des indices de spécificité de Lafon (1980) sur données SPARSES.
 *
 * Historique des corrections
 * ──────────────────────────
 * v1 → v2 : récurrence log-prob (zéro lgamma dans la boucle interne)
 * v2 → v3 : overflow long long corrigé ; scores négatifs (tail inversion +
 *            récurrence descendante) ; log_max corrigé dans log-sum-exp
 * v3 → v4 : tous les bugs identifiés en revue de code corrigés :
 *
 *   BUG-1 (CRITIQUE) — log(0) ou log(négatif) dans les récurrences
 *     N-K-n+x+1 (ascendante) ou N-K-n+x (descendante) peuvent être ≤ 0
 *     quand K+n ≥ N (corpus très dense). std::log(0)=-inf corrompait
 *     silencieusement toute la récurrence.
 *     Correction : test d ≤ 0 avant chaque log, retour NEG_INF pour
 *     signaler la fin de queue à l'appelant.
 *
 *   BUG-2 (SÉRIEUX) — cast implicite long long → double dans lgamma
 *     Pour N > 2^53 (~9e15) la conversion perdait des bits.
 *     Correction : static_cast<double> explicite sur tous les arguments.
 *
 *   BUG-3 (SÉRIEUX) — data race OpenMP sur n_threads
 *     Tous les threads écrivaient n_threads simultanément → UB C++11.
 *     Correction : #pragma omp single.
 *
 *   BUG-4 (CRITIQUE) — ordre des tests sur log_p dans le dispatcher
 *     POS_INF >= 0.0 est vrai → le test "log_p >= 0 → non significatif"
 *     court-circuitait le cas k > kmax et retournait 0 au lieu de clip_max.
 *     Correction : tester !isfinite EN PREMIER, avant log_p >= 0.
 *
 *   BUG-5 (MODÉRÉ) — duplication fragile de kmax entre .cpp et .hpp
 *     Correction : seule log_hypergeom_sf_signed est responsable de kmax.
 *
 * Modèle statistique (Lafon 1980)
 * ────────────────────────────────
 * Urne de N occurrences dont K appartiennent au mot j (colonne).
 * On tire n = Fi occurrences (toutes les occurrences du contexte i, ligne).
 * k = nombre observé de cooccurrences (i, j).
 * H0 : tirage sans remise équiprobable.
 *
 *   Sur-représentation  : P(X ≥ k) anormalement petite → spec > 0
 *   Sous-représentation : P(X ≤ k) anormalement petite → spec < 0
 *
 * Score = -log10(p-valeur) signé, unité et convention Lafon 1980.
 *
 * Algorithme — tail inversion + backward recurrence
 * ───────────────────────────────────────────────────
 *   μ = K·n/N  (espérance sous H0)
 *   k ≥ μ  →  queue droite  : sommer P(x), x = k … kmax  (ascendant)
 *   k <  μ  →  queue gauche : sommer P(x), x = k … 0     (descendant)
 * Longueur de boucle ≤ kmax/2 en moyenne → facteur ~2 vs v2.
 * Le filtre μ+1 de la v2 est supprimé (inutile et asymétrique).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace py = pybind11;

// ── Constantes ────────────────────────────────────────────────────────────────
static constexpr double EARLY_EXIT  = 35.0;            // nats ; exp(-35) ≈ 6e-16
static constexpr double LOG10_E_VAL = 0.4342944819032518;
static constexpr double POS_INF     =  std::numeric_limits<double>::infinity();
static constexpr double NEG_INF     = -std::numeric_limits<double>::infinity();

// ── log P(X = k) hypergéométrique — initialisation O(1) ──────────────────────
//
// log[ C(K,k) · C(N-K, n-k) / C(N,n) ]
//
// BUG-2 corrigé : static_cast<double> explicite sur TOUS les arguments de
// lgamma pour éviter la perte de précision quand N > 2^53.
inline double log_hypergeom_pmf_init(
    long long k, long long N, long long K, long long n
) {
    const double dk = static_cast<double>(k);
    const double dN = static_cast<double>(N);
    const double dK = static_cast<double>(K);
    const double dn = static_cast<double>(n);

    return (  std::lgamma(dK + 1.0) - std::lgamma(dk + 1.0) - std::lgamma(dK - dk + 1.0)
            + std::lgamma(dN - dK + 1.0) - std::lgamma(dn - dk + 1.0)
                                         - std::lgamma(dN - dK - dn + dk + 1.0)
            - std::lgamma(dN + 1.0) + std::lgamma(dn + 1.0) + std::lgamma(dN - dn + 1.0)
           );
}

// ── Récurrence ASCENDANTE : log P(X=x+1) depuis log P(X=x) ───────────────────
//
// P(X=x+1)/P(X=x) = [(K-x)(n-x)] / [(x+1)(N-K-n+x+1)]
//
// Logs séparés (pas de produit long long) pour éviter l'overflow.
// BUG-1 corrigé : si N-K-n+x+1 ≤ 0, retourner NEG_INF (stoppe la boucle).
inline double log_hypergeom_pmf_next(
    double    log_px,
    long long x, long long K, long long n, long long N
) {
    const long long d_den = N - K - n + x + 1;
    if (d_den <= 0) return NEG_INF;   // BUG-1 : évite log(0) ou log(négatif)

    return log_px
         + std::log(static_cast<double>(K - x))
         + std::log(static_cast<double>(n - x))
         - std::log(static_cast<double>(x + 1))
         - std::log(static_cast<double>(d_den));
}

// ── Récurrence DESCENDANTE : log P(X=x-1) depuis log P(X=x) ──────────────────
//
// P(X=x-1)/P(X=x) = [x · (N-K-n+x)] / [(K-x+1)(n-x+1)]
//
// BUG-1 corrigé : si N-K-n+x ≤ 0, retourner NEG_INF.
inline double log_hypergeom_pmf_prev(
    double    log_px,
    long long x, long long K, long long n, long long N
) {
    const long long d_num = N - K - n + x;
    if (d_num <= 0) return NEG_INF;   // BUG-1 : évite log(0) ou log(négatif)

    return log_px
         + std::log(static_cast<double>(x))
         + std::log(static_cast<double>(d_num))
         - std::log(static_cast<double>(K - x + 1))
         - std::log(static_cast<double>(n - x + 1));
}

// ── Accumulation log-sum-exp numériquement stable ────────────────────────────
//
// Met à jour {log_sum, log_max} avec log_px.
// log_max = maximum de TOUS les log_px vus (référence pour early-exit).
// Retourne false si la contribution est négligeable (early-exit).
//
// Note : log_max est mis à jour AVANT le test early-exit, ce qui garantit
// que l'early-exit est évalué par rapport au vrai maximum global, y compris
// dans la queue gauche où la distribution peut monter avant de descendre.
inline bool logsumexp_update(double log_px, double& log_sum, double& log_max)
{
    if (log_px > log_max) log_max = log_px;
    if (log_px < log_max - EARLY_EXIT) return false;

    if (log_px > log_sum)
        log_sum = log_px + std::log1p(std::exp(log_sum - log_px));
    else
        log_sum = log_sum + std::log1p(std::exp(log_px - log_sum));

    return true;
}

// ── Calcul central : log-p-valeur signée ─────────────────────────────────────
//
// Retourne std::pair<double, int>{log_p, signe} où :
//   signe = +1  →  sur-représentation  : log_p = log P(X ≥ k)
//   signe = -1  →  sous-représentation : log_p = log P(X ≤ k)
//   log_p est en base naturelle (≤ 0 quand p ≤ 1)
//
// Cas limites (gérés ici, PAS dans le dispatcher .cpp — BUG-5 corrigé) :
//   k ≤ 0      →  {0.0,    -1}  p = 1 (queue gauche triviale, spec = 0)
//   k > kmax   →  {POS_INF, +1} p = 0 (impossible sous H0, spec = clip_max)
//
// Convention appelant (.cpp) :
//   K = col_sums[j] = fréquence du mot j dans le corpus
//   n = row_sums[i] = fréquence du contexte i
inline std::pair<double, int> log_hypergeom_sf_signed(
    long long k, long long N, long long K, long long n
) {
    const long long kmax = std::min(K, n);

    if (k <= 0)   return {0.0,    -1};
    if (k > kmax) return {POS_INF, +1};

    const double mu = static_cast<double>(K) * static_cast<double>(n)
                    / static_cast<double>(N);

    double log_px = log_hypergeom_pmf_init(k, N, K, n);
    if (!std::isfinite(log_px)) return {POS_INF, +1};

    double log_sum = log_px;
    double log_max = log_px;

    if (static_cast<double>(k) >= mu) {
        // ── Queue DROITE : P(X ≥ k) → signe +1 ──────────────────────
        // Distribution décroissante depuis k (car k ≥ μ) ; early-exit rapide.
        for (long long x = k; x < kmax; ++x) {
            log_px = log_hypergeom_pmf_next(log_px, x, K, n, N);
            if (!std::isfinite(log_px)) break;
            if (!logsumexp_update(log_px, log_sum, log_max)) break;
        }
        return {log_sum, +1};

    } else {
        // ── Queue GAUCHE : P(X ≤ k) → signe -1 ──────────────────────
        // Distribution peut monter avant de descendre (mode < k possible) ;
        // log_max est mis à jour dans logsumexp_update avant early-exit.
        for (long long x = k; x > 0; --x) {
            log_px = log_hypergeom_pmf_prev(log_px, x, K, n, N);
            if (!std::isfinite(log_px)) break;
            if (!logsumexp_update(log_px, log_sum, log_max)) break;
        }
        return {log_sum, -1};
    }
}

// ── Déclaration (implémentation dans specificity.cpp) ──────────────────────
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
);
