"""
src/analysis/graph.py
──────────────────────
Construction du réseau sémantique, calcul des métriques et export Gephi.

Fonctions
─────────
build_graph(df_focus, target_word, network_dir, use_mst, hdbscan_csv) → nx.Graph
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

from src.matrix.utils import get_logger

_LOG = get_logger("analysis.graph")


def build_graph(
    df_focus: pd.DataFrame,
    target_word: str,
    network_dir: Path,
    use_mst: bool = True,
    hdbscan_csv: Path | str | None = None,
) -> nx.Graph:
    """
    Construit le réseau, calcule PageRank + Louvain, exporte GEXF + CSV.
    Version optimisée (chargement unique, vectorisation, graphe bipartite accéléré).

    Parameters
    ----------
    df_focus    : sous-matrice de cooccurrence.
    target_word : préfixe des fichiers de sortie.
    network_dir : répertoire de destination.
    use_mst     : True → Arbre Recouvrant Maximum (colonne vertébrale).
                  False → graphe complet.
    hdbscan_csv : chemin vers le CSV produit par Classification_Hdbscan.py
                  (colonnes attendues : 'terme', 'classe_umap', 'classe_tsne').
                  Si None, le script tente la détection automatique dans
                  Results/hdbscan/comparatif_structures_mots.csv.

    Returns
    -------
    nx.Graph avec attributs de nœuds :
        pagerank, community, degree,
        hdbscan_umap (cluster UMAP, -1 = bruit),
        hdbscan_tsne (cluster t-SNE, -1 = bruit).
    """
    network_dir = Path(network_dir)
    network_dir.mkdir(parents=True, exist_ok=True)

    # 1. OPTIMISATION : Construction de graphe accélérée
    if df_focus.index.equals(df_focus.columns):
        np.fill_diagonal(df_focus.values, 0)
        G_full = nx.from_pandas_adjacency(df_focus, create_using=nx.Graph())
    else:
        # Évite le parcours quadratique en utilisant la représentation "stack" (coordonnées)
        # On ne garde que les valeurs strictement positives d'un coup
        df_stacked = df_focus.stack()
        df_edges = df_stacked[df_stacked > 0]
        
        G_full = nx.Graph()
        G_full.add_nodes_from(df_focus.index, bipartite=0)
        G_full.add_nodes_from(df_focus.columns, bipartite=1)
        
        # Ajout des arêtes par lot (bulk)
        edges = [(str(row), str(col), float(w)) for (row, col), w in df_edges.items()]
        G_full.add_weighted_edges_from(edges)

    largest = max(nx.connected_components(G_full), key=len)
    G_main = G_full.subgraph(largest).copy()

    if use_mst:
        G = nx.maximum_spanning_tree(G_main, weight="weight")
        mode = "mst"
    else:
        G = G_main
        mode = "full"

    _LOG.info(f"Mode {mode.upper()} : {G.number_of_nodes()} nœuds, {G.number_of_edges()} liens")

    # ── Métriques ──────────────────────────────────────────────────
    pr = nx.pagerank(G)
    partition = community_louvain.best_partition(G)
    degrees = dict(G.degree())

    nx.set_node_attributes(G, pr, "pagerank")
    nx.set_node_attributes(G, partition, "community")
    nx.set_node_attributes(G, degrees, "degree")
    _LOG.info(f"{len(set(partition.values()))} communauté(s) détectée(s)")

    # ── Enrichissement HDBSCAN (Optimisé : 1 seule lecture) ────────
    _hdbscan_path: Path | None = None
    if hdbscan_csv is not None:
        p = Path(hdbscan_csv)
        _hdbscan_path = p if p.exists() else None
        if _hdbscan_path is None:
            _LOG.warning(f"hdbscan_csv fourni mais introuvable : {p}")
    else:
        for _folder in ["Results/embeddings", "Results/hdbscan"]:
            if Path(_folder).exists():
                candidates = sorted(
                    Path(_folder).glob("comparatif_structures_mots*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    _hdbscan_path = candidates[0]
                    break
        if _hdbscan_path:
            _LOG.info(f"CSV HDBSCAN détecté automatiquement : {_hdbscan_path}")
        else:
            _LOG.warning("Aucun fichier comparatif_structures_mots*.csv trouvé.")

    # Variable pivot pour stocker le DataFrame nettoyé afin d'éviter la ré-lecture plus tard
    df_hdb_clean = None

    if _hdbscan_path is not None:
        _LOG.info(f"Chargement des clusters HDBSCAN depuis : {_hdbscan_path}")
        try:
            # usecols permet de ne charger en mémoire QUE les colonnes nécessaires (très rapide)
            df_raw = pd.read_csv(_hdbscan_path, sep=";", engine="c")
            col_terme = df_raw.columns[0]
            
            if "classe_umap" not in df_raw.columns or "classe_tsne" not in df_raw.columns:
                raise ValueError(f"Colonnes absentes dans {_hdbscan_path}.")

            # Nettoyage unique
            df_hdb_clean = (
                df_raw[[col_terme, "classe_umap", "classe_tsne"]]
                .drop_duplicates(subset=col_terme, keep="first")
                .set_index(col_terme)
            )

            # Assignation rapide via dictionnaire (évite le reindex coûteux)
            nodes = list(G.nodes())
            umap_dict = df_hdb_clean["classe_umap"].to_dict()
            tsne_dict = df_hdb_clean["classe_tsne"].to_dict()

            cluster_umap = {n: int(umap_dict.get(n, -2)) for n in nodes}
            cluster_tsne = {n: int(tsne_dict.get(n, -2)) for n in nodes}
            
            n_matched = sum(1 for n in nodes if n in umap_dict)

            nx.set_node_attributes(G, cluster_umap, "hdbscan_umap")
            nx.set_node_attributes(G, cluster_tsne, "hdbscan_tsne")
            _LOG.info(f"Clusters HDBSCAN injectés : {n_matched}/{G.number_of_nodes()} nœuds appariés.")
            
        except Exception as exc:
            _LOG.error(f"Échec du chargement HDBSCAN : {exc}")
            df_hdb_clean = None

    # ── Export GEXF ────────────────────────────────────────────────
    gexf_path = network_dir / f"{mode}_{target_word}.gexf"
    nx.write_gexf(G, str(gexf_path))
    _LOG.info(f"Fichier Gephi → {gexf_path}")

    # ── Export CSV stats (Optimisé : Pas de re-lecture disque) ─────
    df_stats = pd.DataFrame({
        "pagerank": pr,
        "community": partition,
        "degree": degrees,
    }).sort_values("pagerank", ascending=False)

    if df_hdb_clean is not None:
        # On renomme simplement l'index et les colonnes du DF déjà en mémoire
        df_hdb_merge = df_hdb_clean.rename_axis("terme").rename(
            columns={"classe_umap": "hdbscan_umap", "classe_tsne": "hdbscan_tsne"}
        )
        df_stats = df_stats.join(df_hdb_merge, how="left")
        df_stats[["hdbscan_umap", "hdbscan_tsne"]] = (
            df_stats[["hdbscan_umap", "hdbscan_tsne"]].fillna(-2).astype(int)
        )

    stats_path = network_dir / f"{mode}_node_stats.csv"
    df_stats.to_csv(stats_path, sep=";")
    _LOG.info(f"Statistiques → {stats_path}")

    # ── Visualisation graphique ─────────────────────────────────────
    hdbscan_umap_attrs = nx.get_node_attributes(G, "hdbscan_umap")
    if hdbscan_umap_attrs:
        try:
            _plot_hdbscan_graph(G, pr, hdbscan_umap_attrs, target_word, mode, network_dir)
        except Exception as exc:
            _LOG.warning(f"Visualisation HDBSCAN échouée : {exc}")

    return G


# ── Visualisation privée ───────────────────────────────────────────────────────

def _plot_hdbscan_graph(
    G: "nx.Graph",
    pr: dict,
    hdbscan_umap: dict,
    target_word: str,
    mode: str,
    network_dir: Path,
    figsize: tuple = (22, 16),
    top_n_labels: int = 60,
) -> None:
    """
    Génère une visualisation du graphe colorée par cluster HDBSCAN (UMAP).

    Encodage visuel
    ───────────────
      Couleur  → cluster HDBSCAN UMAP (-2 = non apparié gris, -1 = bruit noir)
      Taille   → PageRank normalisé
      Labels   → top_n_labels nœuds par PageRank décroissant
    """
    # ── Layout ────────────────────────────────────────────────────
    # Ajout de iterations=30 pour accélérer la génération du layout
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.8 / (G.number_of_nodes() ** 0.5), iterations=30)

    # ── Palette de couleurs ────────────────────────────────────────
    cluster_ids = sorted(set(hdbscan_umap.values()))
    # Clusters réels (≥ 0) → colormap tab20 ; -1 bruit → noir ; -2 absent → gris
    real_clusters = [c for c in cluster_ids if c >= 0]
    cmap = plt.cm.get_cmap("tab20", max(len(real_clusters), 1))
    color_map_clusters = {c: cmap(i) for i, c in enumerate(real_clusters)}
    color_map_clusters[-1] = (0.15, 0.15, 0.15, 0.6)   # bruit → gris foncé
    color_map_clusters[-2] = (0.75, 0.75, 0.75, 0.4)   # non apparié → gris clair

    node_colors = [color_map_clusters.get(hdbscan_umap.get(n, -2), (0.75, 0.75, 0.75, 0.4))
                   for n in G.nodes()]

    # ── Taille des nœuds proportionnelle au PageRank ───────────────
    pr_vals = np.array([pr.get(n, 0.0) for n in G.nodes()])
    pr_norm = (pr_vals - pr_vals.min()) / (pr_vals.max() - pr_vals.min() + 1e-9)
    node_sizes = 80 + pr_norm * 1200

    # ── Figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Arêtes (pondérées → épaisseur)
    weights = np.array([G[u][v].get("weight", 1.0) for u, v in G.edges()])
    if weights.max() > weights.min():
        w_norm = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        w_norm = np.ones(len(weights))
    edge_widths = 0.3 + w_norm * 1.5

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.25,
        edge_color="#AAAAAA",
    )

    # Nœuds
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.90,
        linewidths=0.4,
        edgecolors="white",
    )

    # Labels : top_n_labels par PageRank
    top_nodes = sorted(pr, key=pr.get, reverse=True)[:top_n_labels]
    labels = {n: n for n in top_nodes if n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=6,
        font_color="#111111",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.55, lw=0),
    )

    # ── Légende clusters ──────────────────────────────────────────
    handles = []
    for cid in sorted(color_map_clusters):
        if cid == -2:
            label = "Non apparié"
        elif cid == -1:
            label = "Bruit HDBSCAN"
        else:
            label = f"Cluster {cid}"
        handles.append(mpatches.Patch(color=color_map_clusters[cid], label=label))

    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        ncol=max(1, len(handles) // 20),
        title="Clusters HDBSCAN (UMAP)",
        title_fontsize=8,
    )

    titre_ligne1 = f"Réseau sémantique — {target_word}  ·  Mode {mode.upper()}"
    titre_ligne2 = (
        f"Couleur : cluster HDBSCAN UMAP  ·  Taille : PageRank  ·  "
        f"{G.number_of_nodes()} nœuds, {G.number_of_edges()} liens"
    )
    ax.set_title(titre_ligne1 + "\n" + titre_ligne2, fontsize=11, pad=14)

    plt.tight_layout()
    out_path = network_dir / f"{mode}_{target_word}_hdbscan.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info(f"Visualisation HDBSCAN → {out_path}")

