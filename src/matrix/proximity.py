"""
src/matrix/proximity.py
────────────────────────
Métriques de proximité entre un mot-pôle et les autres termes.

Métriques
─────────
'raw'    : moyenne (ligne + colonne) du pôle — O(V) mémoire.
'pmi'    : PMI vectorisé sur marginales — O(V).
'cosine' : similarité de profil — dense (V≤4000) ou chunked (V>4000).

Supporte DataFrame dense ET sp.csr_matrix en entrée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from .utils import COSINE_DENSE_THRESHOLD, get_logger, minmax_normalize

_LOG = get_logger("matrix.proximity")


class ProximityEngine:
    """
    Calcule les scores de proximité entre un mot-pôle et les autres termes.

    Constructeurs
    -------------
    ProximityEngine.from_dataframe(df, target_word)
    ProximityEngine.from_sparse(mat, words, target_word)
    """

    def __init__(
        self,
        data: np.ndarray | sp.csr_matrix,
        words: list[str],
        pole_idx: int,
    ) -> None:
        self._data      = data
        self._words     = words
        self._pole_idx  = pole_idx
        self._is_sparse = sp.issparse(data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_word: str) -> "ProximityEngine":
        words    = df.index.tolist()
        pole_idx = words.index(target_word) if target_word in words else -1
        return cls(df.values.astype(np.float32), words, pole_idx)

    @classmethod
    def from_sparse(
        cls,
        mat: sp.csr_matrix,
        words: list[str],
        target_word: str,
    ) -> "ProximityEngine":
        pole_idx = words.index(target_word) if target_word in words else -1
        return cls(mat, words, pole_idx)

    def compute(
        self,
        metric: str = "cosine",
        batch_size: int = 512,
        pmi_min_count: int = 2,
    ) -> pd.Series:
        """
        Calcule et normalise les scores de proximité.

        Returns
        -------
        pd.Series normalisée [0, 1], indexée par les mots.
        """
        if self._pole_idx < 0:
            return pd.Series(0.0, index=self._words)

        if metric == "raw":
            scores = self._raw()
        elif metric == "pmi":
            scores = self._pmi(pmi_min_count)
        elif metric == "cosine":
            scores = self._cosine(batch_size)
        elif metric == "dice":
            scores = self._dice()
        else:
            raise ValueError(f"metric doit être 'raw', 'pmi', 'cosine' ou 'dice' (reçu: {metric!r})")

        return minmax_normalize(pd.Series(scores, index=self._words))

    # ── Métriques ──────────────────────────────────────────────────

    def _raw(self) -> np.ndarray:
        idx = self._pole_idx
        if self._is_sparse:
            row = np.asarray(self._data[idx].todense()).flatten().astype(float)
            col = np.asarray(self._data[:, idx].todense()).flatten().astype(float)
        else:
            row = self._data[idx].astype(float)
            col = self._data[:, idx].astype(float)
        return (row + col) / 2.0

    def _dice(self) -> np.ndarray:
        """
        Coefficient de Dice : 2 * cooc(pôle, j) / (freq(pôle) + freq(j))
        Symétrique, normalisé entre 0 et 1.
        Favorise les termes fréquemment co-occurrents avec le pôle,
        pénalise les termes très rares ou très fréquents.
        """
        idx = self._pole_idx
        if self._is_sparse:
            row   = np.asarray(self._data[idx].todense()).flatten().astype(float)
            col   = np.asarray(self._data[:, idx].todense()).flatten().astype(float)
            fi    = float(self._data[idx].sum())
            fj    = np.asarray(self._data.sum(axis=1)).flatten().astype(float)
        else:
            row   = self._data[idx].astype(float)
            col   = self._data[:, idx].astype(float)
            fi    = float(self._data[idx].sum())
            fj    = self._data.sum(axis=1).astype(float)

        cooc  = (row + col) / 2.0
        denom = fi + fj
        scores = np.where(denom > 0, 2.0 * cooc / denom, 0.0)
        return scores.astype(np.float32)

    def _pmi(self, pmi_min_count: int) -> np.ndarray:
        idx = self._pole_idx
        if self._is_sparse:
            total    = self._data.sum()
            p_target = self._data[idx].sum() / total
            p_others = np.asarray(self._data.sum(axis=0)).flatten() / total
            pole_row = np.asarray(self._data[idx].todense()).flatten().astype(float)
        else:
            total    = self._data.sum()
            p_target = self._data[idx].sum() / total
            p_others = self._data.sum(axis=0) / total
            pole_row = self._data[idx].astype(float)

        if total <= 0:
            return np.zeros(len(self._words))

        p_joint = pole_row / total
        mask    = pole_row >= pmi_min_count
        denom   = p_target * p_others
        scores  = np.zeros(len(self._words), dtype=np.float64)
        valid   = mask & (denom > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[valid] = np.log2(p_joint[valid] / denom[valid] + 1e-12)
        return np.where(np.isfinite(scores), scores, 0.0)

    def _cosine(self, batch_size: int) -> np.ndarray:
        idx = self._pole_idx
        V   = len(self._words)

        if self._is_sparse:
            return self._cosine_sparse_chunked(idx, batch_size)

        data = self._data  # float32 ndarray
        if V <= COSINE_DENSE_THRESHOLD:
            return cosine_similarity(data[idx:idx+1], data)[0].astype(np.float32)

        # Chunked dense
        target = data[idx].astype(np.float32)
        norm_t = np.linalg.norm(target)
        if norm_t == 0:
            return np.zeros(V, dtype=np.float32)
        target = target / norm_t

        sims = np.empty(V, dtype=np.float32)
        for start in range(0, V, batch_size):
            end   = min(start + batch_size, V)
            chunk = data[start:end].astype(np.float32)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sims[start:end] = (chunk / norms) @ target
        return sims

    def _cosine_sparse_chunked(self, pole_idx: int, batch_size: int) -> np.ndarray:
        V      = self._data.shape[0]
        target = np.asarray(self._data[pole_idx].todense()).flatten().astype(np.float32)
        norm_t = np.linalg.norm(target)
        if norm_t == 0:
            return np.zeros(V, dtype=np.float32)
        target = target / norm_t

        sims = np.empty(V, dtype=np.float32)
        for start in range(0, V, batch_size):
            end   = min(start + batch_size, V)
            chunk = self._data[start:end].toarray().astype(np.float32)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sims[start:end] = (chunk / norms) @ target
        return sims
