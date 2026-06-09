"""
src/analysis/Hdbscan.py
────────────────────────
Classification topologique et clustering HDBSCAN sur la sous-matrice de cooccurrence.

API publique
────────────
  run_hdbscan(cfg, poles_label) → pd.DataFrame

  Tous les paramètres sont reçus via l'objet Config ; aucune variable globale,
  aucune résolution de chemin au niveau module — même interface que graph.py.

Sorties (dans cfg.embeddings_dir)
──────────────────────────────────
  comparatif_structures_mots_{poles_label}.csv   ← clé pour graph.py
  rapport_thematiques_detaille_{poles_label}.txt
  comparaison_topologique_umap_tsne_{poles_label}.png

Utilisation standalone
──────────────────────
  python Hdbscan.py          ← charge config.py automatiquement
"""

from __future__ import annotations

import gc
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")


# ── API publique ──────────────────────────────────────────────────────────────

def run_hdbscan(cfg, poles_label: str) -> pd.DataFrame:
    """
    Exécute le pipeline UMAP + t-SNE + HDBSCAN et exporte les résultats.

    Parameters
    ----------
    cfg         : objet Config (doit exposer submatrix_path(), submatrix_dir,
                  embeddings_dir, umap_neighbors, tsne_perplexity)
    poles_label : étiquette courte des mots-pôles (utilisée dans les noms de fichiers)

    Returns
    -------
    pd.DataFrame avec colonnes :
        terme, freq_totale,
        umap_x, umap_y, classe_umap,
        tsne_x, tsne_y, classe_tsne
    """
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Résolution de la sous-matrice ──────────────────────────
    chemin = _resoudre_submatrix(cfg)

    # ── 2. Chargement et nettoyage ────────────────────────────────
    data_brute, noms_mots, noms_docs = _charger_et_nettoyer(chemin)

    # ── 3. Préparation des profils ────────────────────────────────
    X_words            = data_brute.T                        # (n_mots × n_docs)
    freq_absolues_mots = data_brute.sum(axis=0)              # volume global par mot
    X_words_norm       = X_words / (X_words.sum(axis=1, keepdims=True) + 1e-9)

    N_TERMES = len(noms_mots)
    TAILLE_MIN_CLUSTER, ECHANTILLONS_MIN = _regle_echelle(N_TERMES)

    del data_brute, X_words
    gc.collect()

    # ── 4. Projection UMAP ────────────────────────────────────────
    print(f"[*] Calcul UMAP (n_neighbors={cfg.umap_neighbors}, metric=cosine)…")
    reducer_umap = umap.UMAP(
        n_neighbors=cfg.umap_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    embedding_umap = reducer_umap.fit_transform(X_words_norm)

    # ── 5. Projection t-SNE ───────────────────────────────────────
    print(f"[*] Calcul t-SNE (perplexity={cfg.tsne_perplexity})…")
    X_l2 = X_words_norm / (np.linalg.norm(X_words_norm, axis=1, keepdims=True) + 1e-9)
    if X_l2.shape[1] > 50:
        X_l2 = TruncatedSVD(n_components=50, random_state=42).fit_transform(X_l2)
    embedding_tsne = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        random_state=42,
        init="pca",
    ).fit_transform(X_l2)

    del X_words_norm, X_l2
    gc.collect()

    # ── 6. Clustering HDBSCAN ─────────────────────────────────────
    print(f"[*] HDBSCAN (min_cluster_size={TAILLE_MIN_CLUSTER}, min_samples={ECHANTILLONS_MIN})…")
    labels_umap = hdbscan.HDBSCAN(
        min_cluster_size=TAILLE_MIN_CLUSTER,
        min_samples=ECHANTILLONS_MIN,
        core_dist_n_jobs=-1,
    ).fit_predict(embedding_umap)

    labels_tsne = hdbscan.HDBSCAN(
        min_cluster_size=TAILLE_MIN_CLUSTER,
        min_samples=ECHANTILLONS_MIN,
        core_dist_n_jobs=-1,
    ).fit_predict(embedding_tsne)

    # ── 7. DataFrame résultat ─────────────────────────────────────
    df = pd.DataFrame({
        "terme":        noms_mots,
        "freq_totale":  freq_absolues_mots,
        "umap_x":       embedding_umap[:, 0],
        "umap_y":       embedding_umap[:, 1],
        "classe_umap":  labels_umap,
        "tsne_x":       embedding_tsne[:, 0],
        "tsne_y":       embedding_tsne[:, 1],
        "classe_tsne":  labels_tsne,
    })

    # ── 8. Exports ────────────────────────────────────────────────
    _exporter(df, cfg.embeddings_dir, poles_label)

    return df


# ── Helpers privés ────────────────────────────────────────────────────────────

def _resoudre_submatrix(cfg) -> Path:
    """Retourne le chemin de la sous-matrice avec fallback sur le fichier le plus récent."""
    chemin = Path(cfg.submatrix_path())
    if chemin.exists():
        print(f"[+] Sous-matrice trouvée → {chemin}")
        return chemin

    print(f"[!] Fichier attendu introuvable : {chemin}")
    print(f"[~] Recherche du fichier le plus récent dans {cfg.submatrix_dir}…")
    candidats = sorted(
        Path(cfg.submatrix_dir).glob("themes_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidats:
        raise FileNotFoundError(
            f"Aucun fichier themes_*.csv dans {cfg.submatrix_dir}. "
            "Lancez d'abord : python main.py --steps submatrix"
        )
    chemin = candidats[0]
    print(f"[~] Fallback → {chemin}")
    return chemin


def _charger_et_nettoyer(chemin: Path) -> tuple[np.ndarray, list, list]:
    """Charge le CSV et convertit tout en float32 (anti-bug valeurs parasites)."""
    print(f"[*] Chargement de la sous-matrice : {chemin}…")
    df = pd.read_csv(chemin, sep=";", index_col=0, engine="c")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    noms_mots = df.columns.tolist()
    noms_docs = df.index.tolist()
    data = df.values.astype(np.float32)
    print(f"[+] Données nettoyées : {len(noms_docs)} documents, {len(noms_mots)} termes.")
    return data, noms_mots, noms_docs


def _regle_echelle(n_termes: int, k_cible: int = 25, taux_bruit: float = 0.40) -> tuple[int, int]:
    """Calibre min_cluster_size et min_samples dynamiquement selon la taille du vocabulaire."""
    taille_min = max(20, int((n_termes * (1 - taux_bruit)) / k_cible))
    echantillons_min = max(5, int(taille_min * 0.10))
    print(f"[i] Règle d'échelle ({n_termes} termes) : "
          f"min_cluster_size={taille_min}, min_samples={echantillons_min}")
    return taille_min, echantillons_min


def _exporter(df: pd.DataFrame, output_dir: Path, poles_label: str) -> None:
    """Écrit le CSV, le rapport textuel et le graphique comparatif."""
    csv_out     = output_dir / f"comparatif_structures_mots_{poles_label}.csv"
    rapport_out = output_dir / f"rapport_thematiques_detaille_{poles_label}.txt"
    img_out     = output_dir / f"comparaison_topologique_umap_tsne_{poles_label}.png"

    # CSV principal (clé pour graph.py)
    print(f"[*] Sauvegarde CSV → {csv_out}")
    df.to_csv(csv_out, sep=";", index=False, encoding="utf-8")

    # Rapport textuel
    print(f"[*] Rapport sémantique → {rapport_out}")
    classes_u = sorted(c for c in df["classe_umap"].unique() if c != -1)
    bruit     = int((df["classe_umap"] == -1).sum())
    with open(rapport_out, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("EXPLORATION SÉMANTIQUE DES CLUSTERS OPTIMISÉS (UMAP)\n")
        f.write("==================================================\n\n")
        f.write(f"Mots isolés (bruit, -1) : {bruit} / {len(df)}\n")
        f.write(f"Thématiques robustes    : {len(classes_u)}\n\n")
        for cid in classes_u:
            sous = df[df["classe_umap"] == cid].sort_values("freq_totale", ascending=False)
            top  = ", ".join(sous["terme"].tolist()[:15])
            f.write(f"CLUSTER UMAP {cid} ({len(sous)} mots) :\n")
            f.write(f"  > {top}\n\n")

    # Graphique comparatif
    print(f"[*] Graphique comparatif → {img_out}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(df["umap_x"], df["umap_y"], c=df["classe_umap"], cmap="Spectral", s=4, alpha=0.5)
    ax1.set_title("Topologie UMAP + HDBSCAN", fontsize=12)
    ax1.set_xlabel("UMAP 1"); ax1.set_ylabel("UMAP 2")
    ax2.scatter(df["tsne_x"], df["tsne_y"], c=df["classe_tsne"], cmap="Spectral", s=4, alpha=0.5)
    ax2.set_title("Topologie t-SNE + HDBSCAN", fontsize=12)
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(img_out, dpi=150)
    plt.close(fig)
    print("[+] Exports terminés.")


# ── Utilisation standalone ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from config import Config
    cfg = Config()
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    poles_label = cfg._poles_label()
    df_result = run_hdbscan(cfg, poles_label)
    print(f"\nTerminé. Résultats dans : {cfg.embeddings_dir}/")
