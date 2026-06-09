"""
src/analysis/kohonen.py
────────────────────────
Carte Auto-Organisatrice de Kohonen (SOM) — implémentation pure NumPy.

Fonctions publiques
───────────────────
run_kohonen(df, output_dir, ...) → dict[str, pd.DataFrame]

Sorties
───────
  *_bmu.csv       — terme → cellule BMU
  *_umatrix.csv   — distances neuronales
  *_weights.csv   — poids de la grille
  *.(svg|png)     — carte principale (U-matrix + étiquettes)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.matrix.utils import get_logger, normalize_rows

_LOG = get_logger("analysis.kohonen")

# ── Valeurs par défaut ─────────────────────────────────────────────────────────
_DEFAULTS = dict(
    grid_shape     = "auto",
    sigma          = None,
    learning_rate  = 0.5,
    n_iterations   = 5_000,
    topology       = "rectangular",
    normalize_data = True,
    umatrix_cmap   = "Blues",
    top_n_labels   = 3,
    figsize        = (16, 12),
    target_word    = None,
)


# ── API publique ───────────────────────────────────────────────────────────────

def run_kohonen(
    df: pd.DataFrame,
    output_dir: Path,
    label: str                  = "kohonen",
    target_word: str | None     = _DEFAULTS["target_word"],
    grid_shape: tuple | str     = _DEFAULTS["grid_shape"],
    sigma: float | None         = _DEFAULTS["sigma"],
    learning_rate: float        = _DEFAULTS["learning_rate"],
    n_iterations: int           = _DEFAULTS["n_iterations"],
    topology: str               = _DEFAULTS["topology"],
    normalize_data: bool        = _DEFAULTS["normalize_data"],
    umatrix_cmap: str           = _DEFAULTS["umatrix_cmap"],
    top_n_labels: int           = _DEFAULTS["top_n_labels"],
    figsize: tuple              = _DEFAULTS["figsize"],
) -> dict[str, pd.DataFrame]:
    """
    Entraîne une SOM et génère la carte principale avec U-matrix.

    Parameters
    ----------
    df            : sous-matrice (lignes = termes, colonnes = contextes)
    output_dir    : répertoire de sortie
    label         : préfixe des fichiers de sortie
    target_word   : mot-pôle (mis en évidence sur la carte)
    grid_shape    : (rows, cols) ou 'auto' (√5√n arrondi)
    sigma         : rayon initial du voisinage (défaut : grid_rows / 2)
    learning_rate : taux d'apprentissage initial
    n_iterations  : nombre d'itérations d'entraînement
    topology      : 'rectangular' ou 'hexagonal'
    normalize_data: normalisation L2 des vecteurs avant entraînement
    umatrix_cmap  : colormap pour l'U-matrix (ex: 'Blues', 'viridis')
    top_n_labels  : nombre max de termes affichés par cellule
    figsize       : dimensions de la figure en pouces

    Returns
    -------
    dict avec clés : 'bmu_map', 'umatrix', 'weights'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    words             = df.index.tolist()
    data              = df.values.astype(float)
    n_terms, n_features = data.shape

    # ── Normalisation ──────────────────────────────────────────────
    data_som = normalize_rows(data) if normalize_data else data.copy()

    # ── Taille de grille ───────────────────────────────────────────
    grid_rows, grid_cols = _resolve_grid(grid_shape, n_terms)
    if sigma is None:
        sigma = grid_rows / 2.0

    _LOG.info(
        f"Grille {grid_rows}×{grid_cols} | σ={sigma:.1f} | "
        f"lr={learning_rate} | {n_iterations} it. | topo={topology}"
    )

    # ── Entraînement ───────────────────────────────────────────────
    som = _SOM(grid_rows, grid_cols, n_features, sigma, learning_rate, topology)
    som.fit(data_som, n_iterations)

    bmus = [som.bmu(v) for v in data_som]
    df_bmu = pd.DataFrame({
        "term":    words,
        "bmu_row": [b[0] for b in bmus],
        "bmu_col": [b[1] for b in bmus],
    })
    umatrix  = som.umatrix()
    feature_names = df.columns.tolist()

    # ── Exports CSV ────────────────────────────────────────────────
    df_bmu.to_csv(output_dir / f"{label}_bmu.csv", sep=";", index=False)
    pd.DataFrame(umatrix).to_csv(output_dir / f"{label}_umatrix.csv", sep=";")
    pd.DataFrame(
        som.weights.reshape(grid_rows * grid_cols, n_features),
        columns=feature_names,
    ).to_csv(output_dir / f"{label}_weights.csv", sep=";")
    _LOG.info(f"CSV → {output_dir}")

    # ── Figure principale ──────────────────────────────────────────
    _plot_kohonen(
        df_bmu=df_bmu,
        umatrix=umatrix,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        topology=topology,
        target_word=target_word,
        label=label,
        output_dir=output_dir,
        umatrix_cmap=umatrix_cmap,
        top_n_labels=top_n_labels,
        figsize=figsize,
        data=data_som,
        words=words,
    )

    return {
        "bmu_map": df_bmu,
        "umatrix": pd.DataFrame(umatrix),
        "weights": pd.DataFrame(
            som.weights.reshape(-1, n_features), columns=feature_names
        ),
    }


# ── SOM ────────────────────────────────────────────────────────────────────────

class _SOM:
    """Self-Organizing Map minimaliste — pure NumPy, zéro dépendance externe."""

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        n_features: int,
        sigma: float,
        learning_rate: float,
        topology: str,
    ) -> None:
        self.rows, self.cols = grid_rows, grid_cols
        self.sigma0  = sigma
        self.lr0     = learning_rate
        self.topo    = topology

        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal((grid_rows, grid_cols, n_features))
        self._neuron_coords = np.array(
            [[r, c] for r in range(grid_rows) for c in range(grid_cols)],
            dtype=float,
        )

    def fit(self, data: np.ndarray, n_iterations: int) -> None:
        n   = len(data)
        rng = np.random.default_rng(42)
        try:
            self._init_pca(data)
        except Exception:
            pass

        tau = n_iterations / np.log(max(self.sigma0, 1e-6))

        for t in range(n_iterations):
            x            = data[rng.integers(0, n)]
            sigma_t      = self.sigma0 * np.exp(-t / tau)
            lr_t         = self.lr0    * np.exp(-t / n_iterations)
            bmu_r, bmu_c = self.bmu(x)
            self._update(x, bmu_r, bmu_c, sigma_t, lr_t)
            if (t + 1) % max(n_iterations // 5, 1) == 0:
                print(f"[kohonen]   {(t+1)/n_iterations*100:.0f}%", end="\r")

        print(f"[kohonen] Entraînement terminé ({n_iterations} it.)    ")

    def _init_pca(self, data: np.ndarray) -> None:
        """Initialisation des poids par PCA (convergence plus rapide)."""
        centered = data - data.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = Vt[0]
        pc2 = Vt[1] if Vt.shape[0] > 1 else Vt[0]
        lr  = np.linspace(-1, 1, self.rows)
        lc  = np.linspace(-1, 1, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                self.weights[r, c] = lr[r] * pc1 + lc[c] * pc2

    def _update(self, x, bmu_r, bmu_c, sigma, lr):
        bmu_pos = np.array([[bmu_r, bmu_c]], dtype=float)
        coords  = self._hex_coords() if self.topo == "hexagonal" else self._neuron_coords
        dist2   = np.sum((coords - bmu_pos) ** 2, axis=1)
        h       = np.exp(-dist2 / (2 * sigma ** 2)).reshape(self.rows, self.cols, 1)
        self.weights += lr * h * (x - self.weights)

    def _hex_coords(self) -> np.ndarray:
        coords = []
        for r in range(self.rows):
            for c in range(self.cols):
                coords.append([r * np.sqrt(3) / 2, c + 0.5 * (r % 2)])
        return np.array(coords, dtype=float)

    def bmu(self, x: np.ndarray) -> tuple[int, int]:
        dists = np.sum((self.weights - x) ** 2, axis=2)
        idx   = np.unravel_index(np.argmin(dists), dists.shape)
        return int(idx[0]), int(idx[1])

    def umatrix(self) -> np.ndarray:
        U = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                nbrs = [
                    np.linalg.norm(self.weights[r, c] - self.weights[r+dr, c+dc])
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols
                ]
                U[r, c] = np.mean(nbrs) if nbrs else 0.0
        return U


# ── Utilitaires géométriques ───────────────────────────────────────────────────

def _resolve_grid(grid_shape, n_terms: int) -> tuple[int, int]:
    if grid_shape == "auto":
        side = max(3, int(math.ceil(math.sqrt(5 * math.sqrt(n_terms)))))
        return side, side
    return int(grid_shape[0]), int(grid_shape[1])


def _cell_xy(r: int, c: int, hex_mode: bool) -> tuple[float, float]:
    if hex_mode:
        return c + 0.5 * (r % 2), r * (np.sqrt(3) / 2)
    return float(c), float(r)


def _cell_label_priority(
    terms: list[str],
    data: np.ndarray,
    words: list[str],
    top_n: int,
    target_word: str | None,
) -> list[str]:
    word_idx = {w: i for i, w in enumerate(words)}
    norms    = {t: np.linalg.norm(data[word_idx[t]]) for t in terms if t in word_idx}
    sorted_t = sorted(norms, key=norms.get, reverse=True)
    if target_word and target_word in sorted_t:
        sorted_t.remove(target_word)
        sorted_t = [target_word] + sorted_t
    return sorted_t[:top_n]


# ── Figure principale ──────────────────────────────────────────────────────────

def _plot_kohonen(
    df_bmu, umatrix, grid_rows, grid_cols,
    topology, target_word, label, output_dir,
    umatrix_cmap, top_n_labels, figsize, data, words,
):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    hex_mode  = topology == "hexagonal"
    umat_norm = (umatrix - umatrix.min()) / max(umatrix.max() - umatrix.min(), 1e-9)
    cm_u      = plt.get_cmap(umatrix_cmap)

    # ── Fond U-Matrix ──────────────────────────────────────────────
    for r in range(grid_rows):
        for c in range(grid_cols):
            xc, yc = _cell_xy(r, c, hex_mode)
            color  = cm_u(umat_norm[r, c])
            _draw_cell(ax, xc, yc, hex_mode, color, "white", 0.4)

    # ── Termes par cellule ─────────────────────────────────────────
    cell_terms: dict[tuple, list[str]] = {}
    for _, row in df_bmu.iterrows():
        cell_terms.setdefault(
            (int(row["bmu_row"]), int(row["bmu_col"])), []
        ).append(row["term"])

    for (r, c), terms in cell_terms.items():
        xc, yc = _cell_xy(r, c, hex_mode)
        lw     = min(0.4 + len(terms) * 0.3, 2.5)
        _draw_cell(ax, xc, yc, hex_mode, "none", "#555555", lw)

        priority = _cell_label_priority(terms, data, words, top_n_labels, target_word)
        for k, term in enumerate(priority):
            is_pole = term == target_word
            y_off   = yc + (k - (len(priority) - 1) / 2) * 0.22
            if is_pole:
                ax.scatter(xc, yc,      s=260, c=["#F5C4B3"], alpha=0.35,
                           linewidths=0, zorder=4)
                ax.scatter(xc, yc+0.18, s=120, c=["#2C2C2A"],
                           marker="o", zorder=5, edgecolors="white", linewidths=0.5)
            txt = ax.text(
                xc, y_off, term,
                ha="center", va="center",
                fontsize=min(7.5 + len(terms) * 0.1, 10.5),
                fontweight="bold" if is_pole else "normal",
                color="#2C2C2A" if is_pole else "#1A1A1A",
                zorder=6,
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground="white"),
                pe.Normal(),
            ])

    # ── Cadrage ────────────────────────────────────────────────────
    all_x = [_cell_xy(r, c, hex_mode)[0] for r in range(grid_rows) for c in range(grid_cols)]
    all_y = [_cell_xy(r, c, hex_mode)[1] for r in range(grid_rows) for c in range(grid_cols)]
    ax.set_xlim(min(all_x) - 0.8, max(all_x) + 0.8)
    ax.set_ylim(min(all_y) - 0.8, max(all_y) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    title = (
        f"Carte de Kohonen — champ sémantique de « {target_word} »"
        if target_word else "Carte de Kohonen — vue globale du corpus"
    )
    ax.set_title(
        f"{title}\nGrille {grid_rows}×{grid_cols}  ·  {topology}",
        fontsize=11, pad=14,
    )

    # ── Colorbar U-Matrix ──────────────────────────────────────────
    sm = ScalarMappable(cmap=cm_u, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=35, shrink=0.6)
    cbar.set_label("U-Matrix (distance neuronale)", fontsize=8, labelpad=6)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["zone dense", "transition", "frontière"], fontsize=7)

    # ── Légende ────────────────────────────────────────────────────
    handles = []
    if target_word:
        handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="#2C2C2A", markersize=9,
            label=f"« {target_word} » (pôle)",
        ))
    handles.append(mpatches.Patch(
        facecolor="#DDDDDD", edgecolor="#555555",
        linewidth=1.5, label="bordure épaisse = cellule dense",
    ))
    if handles:
        ax.legend(handles=handles, fontsize=8, framealpha=0.85, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_dir / f"{label}.svg", bbox_inches="tight")
    plt.savefig(output_dir / f"{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info(f"Carte principale → {output_dir / label}.(svg|png)")


def _draw_cell(ax, xc, yc, hex_mode, facecolor, edgecolor, lw):
    if hex_mode:
        ax.add_patch(mpatches.RegularPolygon(
            (xc, yc), numVertices=6, radius=0.56, orientation=0,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        ))
    else:
        ax.add_patch(mpatches.FancyBboxPatch(
            (xc - 0.48, yc - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.04",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        ))
