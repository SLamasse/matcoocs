"""
src/matrix/utils.py
────────────────────
Constantes, types et helpers partagés par les modules matrix.
"""

from __future__ import annotations
from typing import TypeAlias
import logging

import numpy as np
import scipy.sparse as sp


# ── Types ──────────────────────────────────────────────────────────────────────
WordToId: TypeAlias = dict[str, int]
SparseMatrix: TypeAlias = sp.csr_matrix


# ── Constantes ─────────────────────────────────────────────────────────────────
# Seuil de bascule dense → chunked pour le cosinus.
# En dessous : numpy dense (rapide). Au-dessus : chunked (économe en RAM).
COSINE_DENSE_THRESHOLD: int = 4_000

# Valeur de clamp pour -log10(0)
SPEC_CLIP_MAX: float = 100.0

# Conversion ln → log10
LOG10_E: float = 0.4342944819032518


# ── Logging ────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Helpers ────────────────────────────────────────────────────────────────────
def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Centre-réduit chaque ligne (in-place safe — retourne une copie)."""
    mu  = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mu) / std


def minmax_normalize(series: "pd.Series") -> "pd.Series":  # noqa: F821
    """Normalise une Series dans [0, 1]. Retourne la série inchangée si constante."""
    import pandas as pd
    vmin, vmax = series.min(), series.max()
    if vmax > vmin:
        return (series - vmin) / (vmax - vmin)
    return series
