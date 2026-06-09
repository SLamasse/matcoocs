"""
src/matrix/export.py
─────────────────────
Import / export des matrices : sparse NPZ, CSV dense (petite taille), CSV long.

Pour les grands vocabulaires (>5000 mots), on n'exporte/charge JAMAIS
en DataFrame dense — on utilise le format NPZ (scipy sparse) ou CSV long.

Paires spécificité / fréquences
────────────────────────────────
to_npz_pair(spec, freq, path, words) sauvegarde les deux matrices dans
deux fichiers NPZ synchronisés (*_freq.npz à côté du fichier principal).
load_freq_npz(path) charge la matrice de fréquences associée.
Ces fonctions sont le pendant NPZ de specificity_to_csv(source_matrix=…).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import get_logger

_LOG = get_logger("matrix.export")

# Seuil au-delà duquel on refuse de densifier
_DENSE_LIMIT = 5_000


# ── Export ─────────────────────────────────────────────────────────────────────

def to_dataframe(matrix: np.ndarray | sp.spmatrix, words: list[str]) -> pd.DataFrame:
    """Convertit en DataFrame — uniquement pour petites matrices."""
    arr = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    return pd.DataFrame(arr, index=words, columns=words)


def to_csv(
    matrix: np.ndarray | sp.spmatrix,
    words: list[str],
    path: Path,
    sparse: bool = False,
) -> None:
    """
    Exporte la matrice en CSV.
    Si V > _DENSE_LIMIT, force automatiquement le format long (3 colonnes).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = len(words)

    if sparse or V > _DENSE_LIMIT:
        if sp.issparse(matrix):
            coo = matrix.tocoo()
            rows, cols, values = coo.row, coo.col, coo.data
        else:
            nz = np.nonzero(matrix)
            rows, cols, values = nz[0], nz[1], matrix[nz]

        pd.DataFrame({
            "mot_pole":     [words[r] for r in rows],
            "mot_contexte": [words[c] for c in cols],
            "valeur":       values,
        }).to_csv(path, index=False)
        _LOG.info(f"Matrice exportée (format long, {len(rows):,} lignes) → {path}")
    else:
        to_dataframe(matrix, words).to_csv(path)
        _LOG.info(f"Matrice exportée (format carré) → {path}")


def to_npz(matrix: sp.spmatrix, path: Path, words: list[str]) -> None:
    """
    Sauvegarde une matrice sparse ET son vocabulaire dans un seul fichier NPZ.
    Les mots sont stockés dans le tableau 'words' — jamais de fichier .txt séparé.
    Format : data, indices, indptr, shape, words (tous dans le même .npz).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = matrix.tocsr()
    np.savez(
        str(path),
        data    = mat.data,
        indices = mat.indices,
        indptr  = mat.indptr,
        shape   = np.array(mat.shape, dtype=np.int64),
        words   = np.array(words, dtype=object),
    )
    _LOG.info(f"Matrice + vocabulaire ({len(words)} mots) → {path}")


def to_npz_pair(
    spec_mat: sp.spmatrix,
    freq_mat: sp.spmatrix,
    path: Path,
    words: list[str],
) -> None:
    """
    Sauvegarde la matrice de spécificité ET la matrice de fréquences brutes
    dans deux fichiers NPZ synchronisés partageant le même vocabulaire.

    Fichiers produits :
        <path>           → matrice de spécificité  (ex. results/spec.npz)
        <stem>_freq.npz  → matrice de fréquences   (ex. results/spec_freq.npz)

    La matrice de fréquences est filtrée sur le même masque non-nul que la
    matrice de spécificité afin que les deux soient strictement alignées.
    Utilisez load_freq_npz(path) pour charger la matrice de fréquences.

    Parameters
    ----------
    spec_mat : matrice de spécificité (sp.spmatrix)
    freq_mat : matrice de fréquences brutes source (sp.spmatrix)
    path     : chemin de sortie pour la spécificité
    words    : vocabulaire commun aux deux matrices
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ── Spécificité ────────────────────────────────────────────────
    to_npz(spec_mat, path, words)

    # ── Fréquences filtrées sur le masque de la spécificité ────────
    mask = spec_mat.tocsr().copy()
    mask.data = np.ones_like(mask.data, dtype=np.float32)
    freq_filtered = sp.csr_matrix(freq_mat.multiply(mask))

    freq_path = _freq_sibling(path, ".npz")
    to_npz(freq_filtered, freq_path, words)
    _LOG.info(f"Fréquences (masque spec, {freq_filtered.nnz:,} non-nuls) → {freq_path}")


# ── Chargement ─────────────────────────────────────────────────────────────────

def load_npz(path: Path) -> tuple[sp.csr_matrix, list[str]]:
    """
    Charge la matrice sparse ET son vocabulaire depuis un fichier NPZ unifié.
    Retourne (csr_matrix, words) — toujours synchronisés par construction.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Matrice NPZ introuvable : {path}")
    _LOG.info(f"Chargement NPZ : {path}")
    f     = np.load(str(path), allow_pickle=True)
    mat   = sp.csr_matrix(
        (f["data"], f["indices"], f["indptr"]),
        shape=tuple(f["shape"]),
    )
    words = f["words"].tolist()
    if mat.shape[0] != len(words):
        raise RuntimeError(
            f"Désalignement interne NPZ : {mat.shape[0]} lignes "
            f"mais {len(words)} mots — fichier corrompu."
        )
    return mat, words


def load_freq_npz(spec_path: Path) -> tuple[sp.csr_matrix, list[str]] | tuple[None, None]:
    """
    Charge la matrice de fréquences brutes associée à un fichier de
    spécificité NPZ (fichier *_freq.npz dans le même répertoire).

    Retourne (csr_matrix, words) si le fichier existe,
    (None, None) sinon — sans lever d'exception.

    Parameters
    ----------
    spec_path : chemin du fichier de spécificité (ex. results/spec.npz)

    Returns
    -------
    (sp.csr_matrix, list[str]) | (None, None)
    """
    freq_path = _freq_sibling(Path(spec_path), ".npz")
    if not freq_path.exists():
        _LOG.info(f"Matrice de fréquences NPZ absente : {freq_path}")
        return None, None
    _LOG.info(f"Chargement fréquences NPZ : {freq_path}")
    return load_npz(freq_path)


def load_csv(path: Path, sep: str = ",") -> sp.csr_matrix | pd.DataFrame:
    """
    Charge une matrice depuis un CSV.

    Détecte automatiquement le format :
    - Format long (3 colonnes mot_pole/mot_contexte/valeur) → sp.csr_matrix
    - Format carré (index en première colonne) → pd.DataFrame (petit vocab)

    Returns sp.csr_matrix pour les grands vocabulaires,
    pd.DataFrame pour les petits (≤ _DENSE_LIMIT colonnes).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Matrice introuvable : {path}\n"
            "Lancez d'abord l'étape 'matrix' via main.py."
        )
    _LOG.info(f"Chargement : {path}")

    # Lire juste l'en-tête pour détecter le format
    header = pd.read_csv(path, sep=sep, nrows=0)
    cols   = list(header.columns)

    if "mot_pole" in cols:
        # Format long → sparse
        return _load_long_csv(path, sep)
    else:
        # Format carré
        n_cols = len(cols)
        if n_cols > _DENSE_LIMIT:
            _LOG.info(f"Grand vocabulaire ({n_cols} cols) → chargement sparse")
            return _load_square_csv_sparse(path, sep)
        else:
            df = pd.read_csv(path, sep=sep, index_col=0)
            _LOG.info(f"Matrice dense chargée ({df.shape[0]}×{df.shape[1]})")
            return df


def load_matrix_sparse(path: Path) -> tuple[sp.csr_matrix, list[str]]:
    """
    Charge la matrice brute et retourne (csr_matrix, words).
    Utilise le NPZ si disponible (plus rapide), sinon lit le CSV.
    Jamais de dense pour les grandes matrices.
    """
    path    = Path(path)
    npz_path = path.with_suffix(".npz")

    if npz_path.exists():
        _LOG.info(f"Chargement NPZ rapide : {npz_path}")
        mat   = sp.load_npz(str(npz_path))
        words = _load_words_from_csv_header(path)
        return mat, words

    _LOG.info(f"Chargement CSV → sparse : {path}")
    return _load_any_csv_to_sparse(path)


# ── Helpers privés ─────────────────────────────────────────────────────────────

def _freq_sibling(base: Path, suffix: str) -> Path:
    """Dérive le chemin *_freq.<suffix> depuis un chemin de base."""
    return base.with_name(base.stem + "_freq" + suffix)


def _load_long_csv(path: Path, sep: str) -> sp.csr_matrix:
    """Charge un CSV long (mot_pole ; mot_contexte ; valeur) en csr_matrix."""
    df   = pd.read_csv(path, sep=sep)
    all_words = sorted(set(df["mot_pole"]) | set(df["mot_contexte"]))
    w2i  = {w: i for i, w in enumerate(all_words)}
    rows = df["mot_pole"].map(w2i).values
    cols = df["mot_contexte"].map(w2i).values
    vals = df["valeur"].values.astype(np.float32)
    V    = len(all_words)
    mat  = sp.csr_matrix((vals, (rows, cols)), shape=(V, V))
    _LOG.info(f"Matrice long CSV chargée : {V} mots, {len(vals):,} non-nuls")
    return mat


def _load_square_csv_sparse(path: Path, sep: str) -> sp.csr_matrix:
    """Charge un CSV carré en lisant ligne par ligne pour éviter la dense 35Go."""
    _LOG.info("Lecture par chunks (CSV carré grand vocabulaire)…")
    words = None
    w2i   = None
    rows_list, cols_list, vals_list = [], [], []

    for chunk in pd.read_csv(path, sep=sep, index_col=0, chunksize=500):
        if words is None:
            # Les colonnes du premier chunk donnent le vocabulaire complet
            # (on relit l'entête pour avoir tous les mots)
            words = list(pd.read_csv(path, sep=sep, nrows=0, index_col=0).columns)
            w2i   = {w: i for i, w in enumerate(words)}

        for term, row_series in chunk.iterrows():
            if term not in w2i:
                continue
            i    = w2i[term]
            mask = row_series.values != 0
            js   = [w2i[c] for c in chunk.columns[mask] if c in w2i]
            vs   = row_series.values[mask]
            rows_list.extend([i] * len(js))
            cols_list.extend(js)
            vals_list.extend(vs[:len(js)])

    V   = len(words)
    mat = sp.csr_matrix(
        (np.array(vals_list, dtype=np.float32),
         (np.array(rows_list, dtype=np.int32),
          np.array(cols_list, dtype=np.int32))),
        shape=(V, V),
    )
    _LOG.info(f"Matrice chargée sparse : {V} mots, {mat.nnz:,} non-nuls")
    return mat


def _load_words_from_csv_header(path: Path) -> list[str]:
    """Lit uniquement la ligne d'en-tête pour extraire la liste des mots."""
    header = pd.read_csv(path, nrows=0, index_col=0)
    return list(header.columns)


def _load_any_csv_to_sparse(path: Path) -> tuple[sp.csr_matrix, list[str]]:
    """Détecte le format et retourne (csr_matrix, words)."""
    header = pd.read_csv(path, nrows=0)
    cols   = list(header.columns)
    if "mot_pole" in cols:
        df       = pd.read_csv(path)
        words    = sorted(set(df["mot_pole"]) | set(df["mot_contexte"]))
        w2i      = {w: i for i, w in enumerate(words)}
        rows_arr = df["mot_pole"].map(w2i).values
        cols_arr = df["mot_contexte"].map(w2i).values
        vals_arr = df["valeur"].values.astype(np.float32)
        V        = len(words)
        return sp.csr_matrix((vals_arr, (rows_arr, cols_arr)), shape=(V, V)), words
    else:
        mat = _load_square_csv_sparse(path, ",")
        words = _load_words_from_csv_header(path)
        return mat, words
