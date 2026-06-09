"""
src/matrix/specificity.py
──────────────────────────
Calcul des indices de spécificité de Lafon (1980)
basés sur la loi hypergéométrique : score = -log10(p-valeur) signé.

Stratégie d'exécution
──────────────────────
1. Extension C++ (_specificity.so) si disponible et use_cpp=True
   → Interface COO sparse : jamais de matrice dense, O(nnz) mémoire.
   → Parallélisé OpenMP.

2. Fallback Python vectorisé lot par lot
   → Même interface COO, même résultat, ~10–30× plus lent que C++.

Scores
──────
  spec > 0  →  sur-représentation  (attraction lexicale)
  spec < 0  →  sous-représentation (répulsion lexicale)

Sortie
──────
sp.csr_matrix  (sparse, V × V) — JAMAIS de DataFrame dense.
compute_specificity(...) → sp.csr_matrix

Corrections v4
──────────────
- Backend Python : filtre « k > μ+1 » supprimé (tuait toute sous-repr.)
- Backend Python : calcul de P(X ≤ k) ajouté pour les négatifs
- Backend Python : filtre « specs > 0 » remplacé par « specs != 0 »
- _try_cpp_sparse : passage du paramètre spec_threshold à la v4 C++
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from .utils import SparseMatrix, WordToId, SPEC_CLIP_MAX, get_logger

_LOG = get_logger("matrix.specificity")


# ── API publique ───────────────────────────────────────────────────────────────

def compute_specificity(
    matrix_sparse: SparseMatrix,
    word_to_id: WordToId,
    clip_max: float = SPEC_CLIP_MAX,
    use_cpp: bool = True,
    k_min: int = 3,
    spec_threshold: float = 0.0,
) -> sp.csr_matrix:
    """
    Calcule la matrice de spécificité hypergéométrique signée.

    Retourne une **sp.csr_matrix** — jamais de dense pour éviter le crash
    mémoire sur les grands vocabulaires (>10 000 mots).

    Parameters
    ----------
    matrix_sparse  : sp.csr_matrix  (V × V) — cooccurrences brutes
    word_to_id     : dict[str, int]
    clip_max       : valeur ±clip_max quand p-valeur == 0  (défaut 100)
    use_cpp        : si True, tente l'extension C++, repli Python sinon
    k_min          : seuil minimal de cooccurrence brute (défaut 3)
    spec_threshold : |spec| ≤ seuil → entrée non émise   (défaut 0.0)

    Returns
    -------
    sp.csr_matrix  (V × V)
      valeurs > 0 : sur-représentation  (attraction)
      valeurs < 0 : sous-représentation (répulsion)
    """
    V = len(word_to_id)

    # Statistiques marginales — O(V), jamais de dense
    N        = int(matrix_sparse.sum())
    row_sums = np.asarray(matrix_sparse.sum(axis=1)).flatten().astype(np.int64)
    col_sums = np.asarray(matrix_sparse.sum(axis=0)).flatten().astype(np.int64)

    # Triplets COO des cases non-nulles
    mat_coo = matrix_sparse.tocoo()
    rows_in = mat_coo.row.astype(np.int32)
    cols_in = mat_coo.col.astype(np.int32)
    vals_in = mat_coo.data.astype(np.float32)

    nnz = len(rows_in)
    _LOG.info(f"Calcul des spécificités ({V}×{V}, {nnz:,} non-nuls)…")

    if use_cpp:
        result = _try_cpp_sparse(
            rows_in, cols_in, vals_in,
            row_sums, col_sums, N, clip_max, V, k_min, spec_threshold,
        )
        if result is not None:
            _LOG.info("Calcul C++ terminé.")
            return result

    return _python_sparse(
        rows_in, cols_in, vals_in,
        row_sums, col_sums, N, clip_max, V, k_min, spec_threshold,
    )


def specificity_to_csv(
    mat: sp.csr_matrix,
    words: list[str],
    path,
    source_matrix: sp.csr_matrix | None = None,
) -> None:
    """
    Sauvegarde la matrice de spécificité en CSV.

    Pour les grands vocabulaires, utilise le format long (3 colonnes)
    pour éviter d'écrire une matrice carrée de plusieurs Go.

    Parameters
    ----------
    mat           : matrice de spécificité (sp.csr_matrix, valeurs signées)
    words         : liste des mots (index et colonnes)
    path          : chemin de sortie (ex. results/spec.csv)
    source_matrix : matrice de fréquences brutes originale (optionnel).
                    Si fournie, un fichier *_freq.csv est exporté en
                    parallèle, filtré sur le même masque non-nul que la
                    spécificité.  Destiné à l'AFC.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = len(words)

    # ── Export spécificité ──────────────────────────────────────────
    if V > 5_000:
        coo = mat.tocoo()
        pd.DataFrame({
            "mot_pole":     [words[r] for r in coo.row],
            "mot_contexte": [words[c] for c in coo.col],
            "spec":         coo.data,
        }).to_csv(path, sep=";", index=False)
        _LOG.info(f"Spécificité (format long) → {path}")
    else:
        df = pd.DataFrame(mat.toarray(), index=words, columns=words)
        df.to_csv(path)
        _LOG.info(f"Spécificité (format carré) → {path}")

    # ── Export fréquences brutes (même masque) ──────────────────────
    if source_matrix is not None:
        _export_freq_csv(mat, source_matrix, words, path, V)


def specificity_to_npz(
    mat: sp.csr_matrix,
    words: list[str],
    path,
    source_matrix: sp.csr_matrix | None = None,
) -> None:
    """
    Sauvegarde la matrice de spécificité au format NPZ (scipy sparse).

    Si source_matrix est fourni, sauvegarde également un fichier
    *_freq.npz contenant les fréquences brutes filtrées sur le même
    masque non-nul.  Les deux fichiers partagent le même vocabulaire.

    Parameters
    ----------
    mat           : matrice de spécificité (sp.csr_matrix, valeurs signées)
    words         : liste des mots
    path          : chemin de sortie (ex. results/spec.npz)
    source_matrix : matrice de fréquences brutes originale (optionnel)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _save_npz_with_words(mat, words, path)
    _LOG.info(f"Spécificité NPZ → {path}")

    if source_matrix is not None:
        freq_filtered = _filter_to_mask(mat, source_matrix)
        freq_p        = _freq_sibling(path, ".npz")
        _save_npz_with_words(freq_filtered, words, freq_p)
        _LOG.info(f"Fréquences NPZ (masque spec) → {freq_p}")


def load_specificity(path, words: list[str]) -> sp.csr_matrix:
    """
    Charge une matrice de spécificité depuis un CSV (format long ou carré).
    Détecte automatiquement le format selon le nombre de colonnes.
    Les valeurs négatives (sous-représentation) sont préservées.
    """
    path = Path(path)
    df   = pd.read_csv(path, sep=";", nrows=1)

    if "mot_pole" in df.columns:
        # Format long
        df_full    = pd.read_csv(path, sep=";")
        word_to_id = {w: i for i, w in enumerate(words)}
        rows = [word_to_id[w] for w in df_full["mot_pole"]     if w in word_to_id]
        cols = [word_to_id[w] for w in df_full["mot_contexte"] if w in word_to_id]
        vals = df_full["spec"].values[:len(rows)]   # préserve les négatifs
        V    = len(words)
        return sp.csr_matrix((vals, (rows, cols)), shape=(V, V))
    else:
        # Format carré
        df_sq = pd.read_csv(path, sep=";", index_col=0)
        return sp.csr_matrix(df_sq.values)


def load_freq_matrix(spec_path, words: list[str]) -> sp.csr_matrix | None:
    """
    Charge la matrice de fréquences brutes associée à une matrice de
    spécificité, si elle existe (fichier *_freq.csv ou *_freq.npz).

    Retourne None si le fichier est absent (aucune exception levée).

    Parameters
    ----------
    spec_path : chemin du fichier de spécificité (CSV ou NPZ)
    words     : liste des mots du vocabulaire

    Returns
    -------
    sp.csr_matrix | None
    """
    spec_path = Path(spec_path)

    # Cherche d'abord le NPZ (plus rapide), puis le CSV
    for suffix in (".npz", ".csv"):
        freq_p = _freq_sibling(spec_path.with_suffix(suffix), suffix)
        if freq_p.exists():
            _LOG.info(f"Chargement matrice fréquences : {freq_p}")
            if suffix == ".npz":
                return sp.load_npz(str(freq_p))
            else:
                return _load_freq_csv(freq_p, words)

    _LOG.info("Aucune matrice de fréquences trouvée à côté de la spécificité.")
    return None


# ── Backend C++ ────────────────────────────────────────────────────────────────

def _try_cpp_sparse(
    rows_in, cols_in, vals_in,
    row_sums, col_sums,
    N: int, clip_max: float, V: int, k_min: int,
    spec_threshold: float = 0.0,
) -> sp.csr_matrix | None:
    """
    Tente le calcul via l'extension C++ (specificity_v4).
    Retourne None si l'extension n'est pas disponible.

    Le C++ v4 retourne des scores signés (positifs ET négatifs).
    spec_threshold filtre |spec| ≤ seuil directement côté C++.
    """
    try:
        _spec = importlib.import_module("src.matrix._specificity")
    except ModuleNotFoundError:
        _LOG.info(
            "Extension C++ non trouvée → repli Python. "
            "Lancez `pip install -e .` pour compiler."
        )
        return None

    _LOG.info(
        f"Calcul via extension C++ v4 "
        f"(sparse COO, k_min={k_min}, spec_threshold={spec_threshold})…"
    )
    rows_out, cols_out, specs_out = _spec.compute_specificity_sparse(
        rows_in, cols_in, vals_in,
        row_sums, col_sums,
        N,
        clip_max,
        k_min,
        spec_threshold,   # ← paramètre v4 : filtre |spec| ≤ seuil côté C++
    )
    return sp.csr_matrix(
        (specs_out, (rows_out, cols_out)), shape=(V, V)
    )


# ── Backend Python vectorisé ───────────────────────────────────────────────────

def _python_sparse(
    rows_in: np.ndarray,
    cols_in: np.ndarray,
    vals_in: np.ndarray,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    N: int,
    clip_max: float,
    V: int,
    k_min: int = 3,
    spec_threshold: float = 0.0,
) -> sp.csr_matrix:
    """
    Calcul Python vectorisé des spécificités de Lafon — scores signés.

    Traite par lots de 10 000 pour limiter la mémoire lors des appels
    scipy vectorisés.

    Corrections v4 :
    - Filtre « k > μ+1 » supprimé : il éliminait toute sous-représentation.
    - Calcul de P(X ≤ k) via hypergeom.cdf pour les cas k < μ (score négatif).
    - Filtre final « specs != 0 » au lieu de « specs > 0 ».
    - spec_threshold : |spec| ≤ seuil → non émis (cohérence avec C++).
    """
    _LOG.info(
        f"Calcul Python vectorisé "
        f"(sparse, par lots, k_min={k_min}, spec_threshold={spec_threshold})…"
    )
    BATCH = 10_000

    out_rows: list[np.ndarray] = []
    out_cols: list[np.ndarray] = []
    out_vals: list[np.ndarray] = []

    nnz = len(rows_in)

    for start in range(0, nnz, BATCH):
        end = min(start + BATCH, nnz)
        r_b = rows_in[start:end]
        c_b = cols_in[start:end]
        k_b = vals_in[start:end].astype(int)

        # ── Filtres de base ───────────────────────────────────────────
        mask = (r_b != c_b) & (k_b >= k_min)
        if not mask.any():
            continue

        r_m, c_m, k_m = r_b[mask], c_b[mask], k_b[mask]
        Fi_m = row_sums[r_m]   # n dans Lafon
        fj_m = col_sums[c_m]   # K dans Lafon

        mask2 = (Fi_m > 0) & (fj_m > 0)
        if not mask2.any():
            continue

        r_m, c_m, k_m = r_m[mask2], c_m[mask2], k_m[mask2]
        Fi_m, fj_m    = Fi_m[mask2], fj_m[mask2]

        # Espérance μ = K·n/N — sert à choisir la queue, pas à filtrer
        mu = Fi_m.astype(float) * fj_m.astype(float) / float(N)

        # ── Sur-représentation : k ≥ μ → P(X ≥ k), score > 0 ────────
        mask_pos = k_m >= mu
        if mask_pos.any():
            r_p, c_p, k_p = r_m[mask_pos], c_m[mask_pos], k_m[mask_pos]
            Fi_p, fj_p    = Fi_m[mask_pos], fj_m[mask_pos]

            with np.errstate(divide="ignore", invalid="ignore"):
                # P(X ≥ k) = sf(k-1, N, K, n)
                p_pos = hypergeom.sf(k_p - 1, N, fj_p, Fi_p)

            with np.errstate(divide="ignore"):
                sp_pos = np.where(p_pos > 0, -np.log10(p_pos), clip_max)

            # Écrêtage et filtre de seuil
            sp_pos = np.clip(sp_pos, 0.0, clip_max)
            sig = sp_pos > spec_threshold
            if sig.any():
                out_rows.append(r_p[sig])
                out_cols.append(c_p[sig])
                out_vals.append(sp_pos[sig])

        # ── Sous-représentation : k < μ → P(X ≤ k), score < 0 ───────
        mask_neg = k_m < mu
        if mask_neg.any():
            r_n, c_n, k_n = r_m[mask_neg], c_m[mask_neg], k_m[mask_neg]
            Fi_n, fj_n    = Fi_m[mask_neg], fj_m[mask_neg]

            with np.errstate(divide="ignore", invalid="ignore"):
                # P(X ≤ k) = cdf(k, N, K, n)
                p_neg = hypergeom.cdf(k_n, N, fj_n, Fi_n)

            with np.errstate(divide="ignore"):
                # score négatif : log10(p) quand p > 0, -clip_max sinon
                sp_neg = np.where(p_neg > 0, np.log10(p_neg), -clip_max)

            # Écrêtage symétrique et filtre de seuil
            sp_neg = np.clip(sp_neg, -clip_max, 0.0)
            sig = sp_neg < -spec_threshold
            if sig.any():
                out_rows.append(r_n[sig])
                out_cols.append(c_n[sig])
                out_vals.append(sp_neg[sig])

        if (start // BATCH) % 10 == 0:
            pct = end / nnz * 100
            print(
                f"[specificity]   {pct:.0f}%  ({end:,}/{nnz:,} non-nuls)",
                end="\r",
            )

    print()

    if out_rows:
        r = np.concatenate(out_rows).astype(np.int32)
        c = np.concatenate(out_cols).astype(np.int32)
        v = np.concatenate(out_vals)
        mat = sp.csr_matrix((v, (r, c)), shape=(V, V))
        n_pos = (mat.data > 0).sum()
        n_neg = (mat.data < 0).sum()
        _LOG.info(
            f"Spécificités calculées : {n_pos:,} positives, {n_neg:,} négatives"
        )
        return mat

    return sp.csr_matrix((V, V))


# ── Helpers privés ─────────────────────────────────────────────────────────────

def _freq_sibling(base: Path, suffix: str) -> Path:
    """Dérive le chemin *_freq.<suffix> depuis un chemin de base."""
    return base.with_name(base.stem + "_freq" + suffix)


def _filter_to_mask(
    spec_mat: sp.csr_matrix,
    source_matrix: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Retourne source_matrix filtrée sur le masque binaire de spec_mat.

    Seules les cases non-nulles dans spec_mat sont conservées.
    Les valeurs retournées sont celles de source_matrix (fréquences brutes),
    pas celles de spec_mat.
    Fonctionne correctement même si spec_mat contient des valeurs négatives.
    """
    mask      = spec_mat.copy()
    mask.data = np.ones_like(mask.data, dtype=np.float32)
    return sp.csr_matrix(source_matrix.multiply(mask))


def _export_freq_csv(
    spec_mat: sp.csr_matrix,
    source_matrix: sp.csr_matrix,
    words: list[str],
    spec_path: Path,
    V: int,
) -> None:
    """
    Exporte la matrice de fréquences brutes filtrée (même masque que spec_mat)
    dans un fichier *_freq.csv à côté du fichier de spécificité.
    """
    freq_filtered = _filter_to_mask(spec_mat, source_matrix)
    freq_path     = _freq_sibling(spec_path, ".csv")
    freq_path.parent.mkdir(parents=True, exist_ok=True)

    if V > 5_000:
        coo = freq_filtered.tocoo()
        pd.DataFrame({
            "mot_pole":     [words[r] for r in coo.row],
            "mot_contexte": [words[c] for c in coo.col],
            "freq":         coo.data,
        }).to_csv(freq_path, sep=";", index=False)
        _LOG.info(f"Fréquences brutes (format long, masque spec) → {freq_path}")
    else:
        df = pd.DataFrame(
            freq_filtered.toarray().astype(np.float32),
            index=words,
            columns=words,
        )
        df.to_csv(freq_path)
        _LOG.info(f"Fréquences brutes (format carré, masque spec) → {freq_path}")


def _save_npz_with_words(mat: sp.csr_matrix, words: list[str], path: Path) -> None:
    """Sauvegarde une csr_matrix + son vocabulaire dans un fichier NPZ."""
    m = mat.tocsr()
    np.savez(
        str(path),
        data    = m.data,
        indices = m.indices,
        indptr  = m.indptr,
        shape   = np.array(m.shape, dtype=np.int64),
        words   = np.array(words, dtype=object),
    )


def _load_freq_csv(path: Path, words: list[str]) -> sp.csr_matrix:
    """Charge un CSV de fréquences (format long ou carré) en csr_matrix."""
    df_head    = pd.read_csv(path, sep=";", nrows=1)
    word_to_id = {w: i for i, w in enumerate(words)}
    V          = len(words)

    if "mot_pole" in df_head.columns:
        df   = pd.read_csv(path, sep=";")
        rows = [word_to_id[w] for w in df["mot_pole"]     if w in word_to_id]
        cols = [word_to_id[w] for w in df["mot_contexte"] if w in word_to_id]
        vals = df["freq"].values[:len(rows)].astype(np.float32)
        return sp.csr_matrix((vals, (rows, cols)), shape=(V, V))
    else:
        df_sq = pd.read_csv(path, sep=";", index_col=0)
        return sp.csr_matrix(df_sq.values.astype(np.float32))
