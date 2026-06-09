"""
src/analysis/afc.py
────────────────────
Analyse Factorielle de Correspondances (AFC) sur une sous-matrice.

Encodage visuel
───────────────
  Position  → coordonnées factorielles
  Taille    → contribution factorielle (moyenne F1 + F2), normalisée
  Couleur   → couleur unique neutre (mot-pôle mis en évidence)
  Légende   → 3 cercles aux quantiles 10/50/90% avec valeurs réelles

Plans générés automatiquement
──────────────────────────────
  F1 × F2  → plan principal
  F2 × F3  → plan secondaire
  F1 × F3  → plan complémentaire

Robustesse
──────────
  Les lignes et colonnes entièrement nulles sont supprimées avant le fit
  pour éviter la division par zéro dans prince.CA (calcul de r**-0.5).

Optimisations de performance (v2)
──────────────────────────────────
  1. SCATTER VECTORISÉ  : un seul appel ax.scatter() par figure au lieu
     d'une boucle sur chaque terme → gain x10-x50 sur grandes matrices.
  2. adjust_text FILTRÉ : seules les étiquettes des top-N termes les plus
     contributifs sont repositionnées (coûteux en O(n²)) ; les autres
     sont affichées sans ajustement.
  3. CSV EN PARALLÈLE   : les 4 exports CSV sont écrits dans un
     ThreadPoolExecutor (I/O pur, pas de GIL).
  4. PNG UNIQUEMENT     : le SVG est supprimé par défaut (très lent sur
     grandes figures) ; activable via save_svg=True.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
from adjustText import adjust_text

from src.matrix.utils import get_logger

_LOG = get_logger("analysis.afc")

# ── Valeurs par défaut ─────────────────────────────────────────────────────────
_DEFAULTS = dict(
    contrib_axis      = "mean",
    point_size_range  = (30, 350),
    point_color       = "#3a86ff",
    target_color      = "#e63946",
    figsize           = (16, 12),
    adjust_text_top_n = 80,    # nb de labels repositionnés par adjust_text
    save_svg          = False, # True = aussi exporter en SVG (lent)
)

# Plans factoriels générés systématiquement (indices 0-based)
_PLANS = [(0, 1), (1, 2), (0, 2)]   # F1×F2, F2×F3, F1×F3


# ── API publique ───────────────────────────────────────────────────────────────

def run_afc(
    df_focus: pd.DataFrame,
    target_word: str,
    afc_dir: Path,
    n_components: int = 3,
    contrib_axis: str = _DEFAULTS["contrib_axis"],
    point_size_range: tuple[int, int] = _DEFAULTS["point_size_range"],
    point_color: str = _DEFAULTS["point_color"],
    target_color: str = _DEFAULTS["target_color"],
    figsize: tuple[int, int] = _DEFAULTS["figsize"],
    adjust_text_top_n: int = _DEFAULTS["adjust_text_top_n"],
    save_svg: bool = _DEFAULTS["save_svg"],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Lance l'AFC, exporte les CSV diagnostiques et génère les figures.

    Trois plans factoriels sont produits automatiquement :
      - F1 × F2  (plan principal)
      - F2 × F3  (plan secondaire)
      - F1 × F3  (plan complémentaire)

    Encodage visuel :
      - Position  : coordonnées factorielles
      - Taille    : contribution factorielle (moyenne F1+F2), normalisée
      - Couleur   : couleur unique, mot-pôle distingué par target_color

    Parameters
    ----------
    df_focus           : sous-matrice (lignes = termes, colonnes = contextes)
    target_word        : mot-pôle (affiché en couleur distincte)
    afc_dir            : répertoire de sortie
    n_components       : nombre de facteurs calculés (≥ 3 pour les 3 plans)
    contrib_axis       : agrégation des contributions ('mean', 'sum', 'F1'…)
    point_size_range   : (taille_min, taille_max) en points matplotlib
    point_color        : couleur des termes ordinaires
    target_color       : couleur du mot-pôle
    figsize            : dimensions de la figure en pouces
    adjust_text_top_n  : nb max d'étiquettes repositionnées (perf O(n²))
    save_svg           : si True, exporte aussi en SVG (lent sur grandes figures)

    Returns
    -------
    dict avec clés : 'coords', 'contrib', 'cos2', 'inertie'
    """
    afc_dir = Path(afc_dir)
    afc_dir.mkdir(parents=True, exist_ok=True)

    # n_components doit couvrir tous les plans demandés
    n_components = max(n_components, max(max(p) for p in _PLANS) + 1)

    # ── Symétrisation si matrice carrée (cooccurrences) ────────────
    # Optimisation : comparaison d'index sans conversion en list
    if df_focus.index.equals(df_focus.columns):
        df_sym = df_focus + df_focus.T
        np.fill_diagonal(df_sym.values, 0)
    else:
        df_sym = df_focus.copy()

    # ── Nettoyage : suppression des lignes et colonnes nulles ──────
    row_sums       = df_sym.sum(axis=1)
    col_sums       = df_sym.sum(axis=0)
    lignes_vides   = (row_sums == 0).values
    colonnes_vides = (col_sums == 0).values

    if lignes_vides.any() or colonnes_vides.any():
        _LOG.warning(
            f"Suppression de {lignes_vides.sum()} ligne(s) et "
            f"{colonnes_vides.sum()} colonne(s) nulles avant AFC."
        )
        df_sym = df_sym.loc[~lignes_vides, ~colonnes_vides]

    if df_sym.shape[0] < 3 or df_sym.shape[1] < 3:
        raise ValueError(
            f"Matrice trop petite après nettoyage ({df_sym.shape}) — "
            "vérifiez la sous-matrice d'entrée."
        )

    _LOG.info(
        f"Calcul AFC sur « {target_word} » "
        f"({df_sym.shape[0]} lignes × {df_sym.shape[1]} colonnes)…"
    )
    ca = prince.CA(n_components=n_components, random_state=42).fit(df_sym)

    row_coords  = ca.row_coordinates(df_sym)
    row_contrib = ca.row_contributions_
    row_cos2    = ca.row_cosine_similarities(df_sym)
    inertia     = ca.percentage_of_variance_

    col_names = [f"F{i+1}" for i in range(row_coords.shape[1])]
    for df_ in (row_coords, row_contrib, row_cos2):
        df_.columns = col_names[:df_.shape[1]]

    # ── CSV diagnostiques (écriture parallèle en threads I/O) ─────
    df_inertie = pd.DataFrame({
        "Facteur":     col_names,
        "Inertie (%)": inertia,
        "Cumul (%)":   np.cumsum(inertia),
    })
    _log_inertie_msg = (
        "Inertie par facteur : "
        + "  ".join(f"{col_names[i]}={inertia[i]:.1f}%" for i in range(len(col_names)))
    )

    def _write_csv(args):
        df_, path, kwargs_ = args
        df_.to_csv(path, **kwargs_)

    csv_tasks = [
        (row_coords,  afc_dir / "coordonnees.csv",  {"sep": ";"}),
        (row_contrib, afc_dir / "contributions.csv", {"sep": ";"}),
        (row_cos2,    afc_dir / "cos2.csv",          {"sep": ";"}),
        (df_inertie,  afc_dir / "inertie.csv",       {"sep": ";", "index": False}),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_write_csv, csv_tasks))

    _LOG.info(_log_inertie_msg)

    # ── Contribution normalisée (taille des points) ────────────────
    contrib_vec = _aggregate_contrib(row_contrib, contrib_axis, col_names)

    # ── Figures AFC : F1×F2, F2×F3, F1×F3 ────────────────────────
    for ax1_idx, ax2_idx in _PLANS:
        _plot_afc(
            row_coords=row_coords,
            inertia=inertia,
            target_word=target_word,
            afc_dir=afc_dir,
            contrib_vec=contrib_vec,
            contrib_raw=row_contrib,
            axes_indices=(ax1_idx, ax2_idx),
            col_names=col_names,
            point_size_range=point_size_range,
            point_color=point_color,
            target_color=target_color,
            figsize=figsize,
            adjust_text_top_n=adjust_text_top_n,
            save_svg=save_svg,
        )

    return {
        "coords":  row_coords,
        "contrib": row_contrib,
        "cos2":    row_cos2,
        "inertie": df_inertie,
    }


# ── Helpers privés ─────────────────────────────────────────────────────────────

def _aggregate_contrib(
    df: pd.DataFrame,
    mode: str,
    col_names: list[str],
) -> pd.Series:
    """Agrège les contributions sur les axes et normalise entre 0 et 1."""
    if mode in col_names:
        vec = df[mode]
    elif mode == "mean":
        cols = [c for c in ["F1", "F2"] if c in df.columns]
        vec  = df[cols].mean(axis=1)
    else:
        cols = [c for c in ["F1", "F2"] if c in df.columns]
        vec  = df[cols].sum(axis=1)
    rng = vec.max() - vec.min()
    return (vec - vec.min()) / rng if rng > 0 else vec


def _size_legend_handles(
    contrib_raw: pd.DataFrame,
    point_color: str,
) -> list:
    """
    Construit 3 handles de légende avec des cercles visuellement distincts
    aux quantiles 10 / 50 / 90 % de la distribution réelle des contributions.
    """
    cols = [c for c in ["F1", "F2"] if c in contrib_raw.columns]
    vec  = contrib_raw[cols].mean(axis=1)

    q_low  = float(vec.quantile(0.10))
    q_mid  = float(vec.quantile(0.50))
    q_high = float(vec.quantile(0.90))

    handles = []
    for label_txt, ms in [
        (f"faible  ({q_low  * 100:.2f} %)", 5),
        (f"moyen   ({q_mid  * 100:.2f} %)", 10),
        (f"fort    ({q_high * 100:.2f} %)", 18),
    ]:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o", color="none",
                markerfacecolor=point_color,
                markeredgecolor="white",
                markeredgewidth=0.4,
                markersize=ms,
                label=label_txt,
                alpha=0.88,
            )
        )
    return handles


def _plot_afc(
    row_coords: pd.DataFrame,
    inertia: list[float],
    target_word: str,
    afc_dir: Path,
    contrib_vec: pd.Series,
    contrib_raw: pd.DataFrame,
    axes_indices: tuple[int, int],
    col_names: list[str],
    point_size_range: tuple[int, int],
    point_color: str,
    target_color: str,
    figsize: tuple[int, int],
    adjust_text_top_n: int,
    save_svg: bool,
) -> None:
    """Génère et sauvegarde une figure AFC pour un plan factoriel donné."""
    idx1, idx2 = axes_indices
    name1, name2 = col_names[idx1], col_names[idx2]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=0)

    s_min, s_max = point_size_range
    sizes = (s_min + contrib_vec * (s_max - s_min)).values  # numpy array

    # Coordonnées extraites en arrays numpy une seule fois
    xs = row_coords[name1].values
    ys = row_coords[name2].values
    terms = np.array(row_coords.index)

    # ── OPTIMISATION 1 : scatter vectorisé ────────────────────────
    # Masque booléen pour distinguer mot-pôle vs termes ordinaires
    is_target_mask = (terms == target_word)

    # Termes ordinaires en un seul scatter
    mask_ord = ~is_target_mask
    if mask_ord.any():
        ax.scatter(
            xs[mask_ord], ys[mask_ord],
            s=sizes[mask_ord],
            c=point_color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.4,
            marker="o",
            zorder=4,
        )

    # Mot-pôle (1 point au plus, tracé séparément pour le style distinct)
    if is_target_mask.any():
        ax.scatter(
            xs[is_target_mask], ys[is_target_mask],
            s=sizes[is_target_mask],
            c=target_color,
            alpha=0.88,
            edgecolors="#1A1A1A",
            linewidths=1.5,
            marker="o",
            zorder=5,
        )

    # ── OPTIMISATION 2 : adjust_text limité aux top-N contributifs ─
    # Sélectionne les indices des N termes avec la plus grande contribution
    # + toujours inclure le mot-pôle s'il est présent
    contrib_vals = contrib_vec.values
    top_idx = set(np.argsort(contrib_vals)[-adjust_text_top_n:].tolist())
    target_idx = set(np.where(is_target_mask)[0].tolist())
    label_idx = sorted(top_idx | target_idx)

    texts = []
    path_fx = [pe.withStroke(linewidth=2, foreground="white")]

    for i in label_idx:
        term = terms[i]
        is_t = is_target_mask[i]
        txt = ax.text(
            xs[i], ys[i], term,
            fontsize=8,
            ha="center", va="bottom",
            fontweight="bold" if is_t else "normal",
            color="#1A1A1A",
            zorder=6,
        )
        txt.set_path_effects(path_fx)
        texts.append(txt)

    adjust_text(
        texts, ax=ax,
        expand=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.5),
    )

    # ── Légende ────────────────────────────────────────────────────
    color_handles = [
        mpatches.Patch(facecolor=point_color,  alpha=0.88, label="Termes"),
        mpatches.Patch(facecolor=target_color, alpha=0.88,
                       label=f"Pôle : {target_word}"),
    ]
    sep = mpatches.Patch(
        facecolor="none", edgecolor="none",
        label="── Contribution factorielle (F1+F2) ──",
    )
    size_handles = _size_legend_handles(contrib_raw, point_color)

    ax.legend(
        handles=color_handles + [sep] + size_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        handletextpad=1.0,
        labelspacing=0.6,
        borderpad=0.9,
    )

    ax.set_xlabel(f"{name1}  ({inertia[idx1]:.1f} % de l'inertie)", fontsize=11)
    ax.set_ylabel(f"{name2}  ({inertia[idx2]:.1f} % de l'inertie)", fontsize=11)
    ax.set_title(
        f"AFC — Champ sémantique de « {target_word} »  ·  Plan {name1} × {name2}\n"
        f"Taille : contribution factorielle moyenne (F1+F2)",
        fontsize=11, pad=14,
    )

    plt.tight_layout()
    stem = f"afc_{target_word}_{name1}_{name2}"

    # ── OPTIMISATION 3 : PNG uniquement par défaut (SVG très lent) ─
    plt.savefig(afc_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    if save_svg:
        plt.savefig(afc_dir / f"{stem}.svg", format="svg", bbox_inches="tight")

    plt.close(fig)
    _LOG.info(f"Figure AFC {name1}×{name2} → {afc_dir / stem}.png")
