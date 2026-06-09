"""
src/matrix/builder.py
─────────────────────
Construction du vocabulaire et de la matrice de cooccurrence brute.

Fonctions publiques
───────────────────
build_vocabulary(tokens, min_freq) → WordToId
build_cooccurrence_matrix(tokens, word_to_id, window) → sp.csr_matrix

Optimisations
─────────────
• Vocabulaire en une passe via pd.Series.value_counts
• Petits vocabulaires (< 8 000) : matrice dense + np.add.at → x20–x50 plus rapide
• Grands vocabulaires : accumulation COO par décalage → économe en RAM
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import WordToId, SparseMatrix, get_logger

_LOG = get_logger("matrix.builder")

# Seuil dense vs sparse
_DENSE_VOCAB_LIMIT = 8_000


# ── API publique ───────────────────────────────────────────────────────────────

def build_vocabulary(tokens: list[str], min_freq: int = 1) -> WordToId:
    """
    Construit un mapping mot → index en filtrant par fréquence minimale.

    Parameters
    ----------
    tokens   : liste plate de lemmes
    min_freq : seuil de fréquence d'inclusion

    Returns
    -------
    dict[str, int]  — ordonné par fréquence décroissante
    """
    freq = pd.Series(tokens).value_counts()
    vocab = freq[freq >= min_freq].index.tolist()
    _LOG.info(f"Vocabulaire : {len(vocab)} mots (min_freq={min_freq})")
    return {word: idx for idx, word in enumerate(vocab)}


def build_cooccurrence_matrix(
    tokens: list[str],
    word_to_id: WordToId,
    window: int = 30,
) -> SparseMatrix:
    """
    Construit la matrice de cooccurrence dirigée (lignes = pôles, colonnes = contextes).

    Choisit automatiquement entre accumulation dense et sparse selon
    la taille du vocabulaire.

    Parameters
    ----------
    tokens     : séquence plate de lemmes
    word_to_id : mapping mot → index (vocabulaire filtré)
    window     : taille de fenêtre de cooccurrence

    Returns
    -------
    sp.csr_matrix  — matrice V × V de cooccurrences brutes
    """
    size = len(word_to_id)

    # Tokens → IDs (filtre hors-vocab en une passe NumPy)
    id_seq = np.fromiter(
        (word_to_id[t] for t in tokens if t in word_to_id),
        dtype=np.int32,
    )
    n = len(id_seq)
    _LOG.info(
        f"Calcul des cooccurrences (fenêtre={window}, {n} tokens, {size} mots)…"
    )

    if size < _DENSE_VOCAB_LIMIT:
        return sp.csr_matrix(_accumulate_dense(id_seq, size, window))
    return _accumulate_sparse(id_seq, size, window)


# ── Implémentations privées ────────────────────────────────────────────────────

def _accumulate_dense(
    id_seq: np.ndarray,
    size: int,
    window: int,
) -> np.ndarray:
    """Accumulation dense via np.add.at — optimal pour petit vocabulaire."""
    mat = np.zeros((size, size), dtype=np.float32)
    for d in range(1, window + 1):
        if d >= len(id_seq):
            break
        np.add.at(mat, (id_seq[:-d], id_seq[d:]), 1)
    return mat


def _accumulate_sparse(
    id_seq: np.ndarray,
    size: int,
    window: int,
) -> sp.csr_matrix:
    """Accumulation COO par décalage — économe en RAM pour grand vocabulaire."""
    rows_list, cols_list = [], []
    for d in range(1, window + 1):
        if d >= len(id_seq):
            break
        rows_list.append(id_seq[:-d])
        cols_list.append(id_seq[d:])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.ones(len(rows), dtype=np.float32)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    mat.sum_duplicates()
    return mat
