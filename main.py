"""
main.py
───────
Pipeline d'analyse sémantique par cooccurrence.

Auteur : Stéphane Lamassé
Documentation : Rédigée automatiquement à l'aide d'un grand modèle de langage (LLM).

Pourquoi ce pipeline basé sur la cooccurrence ?
────────────────────────────────────────────────
La cooccurrence part du principe distribué en linguistique : "les mots qui apparaissent
dans des contextes similaires ont des significations similaires". En mesurant la fréquence
à laquelle deux mots apparaissent ensemble dans une même fenêtre de texte, ce script permet 
de cartographier mathématiquement les relations sémantiques. 

Contrairement aux modèles de Deep Learning "boîte noire" (comme Word2Vec ou les LLM), 
cette approche par matrice de cooccurrence offre plusieurs avantages critiques :
1. Interprétabilité totale : Chaque score (fréquence, PMI, spécificité) est calculable 
   et vérifiable mathématiquement.
2. Détection des attractions et répulsions : Grâce aux corrections de la v4, le calcul de 
   spécificité permet de distinguer les mots qui s'attirent (cooccurrence supérieure au hasard) 
   et les mots qui s'évitent activement (cooccurrence inférieure au hasard).

L'intérêt technique du format .npz (Matrices Creuses)
─────────────────────────────────────────────────────
Avec un vocabulaire de 66 000+ mots, une matrice de cooccurrence classique 
(dite "dense") devrait stocker un tableau carré de 66 000 x 66 000 cases, 
soit plus de 4,3 milliards de valeurs. En mémoire RAM, un tel tableau 
représenterait environ 17,4 Go de données, ce qui ferait planter le script.

Le format .npz (NumPy Zip) résout ce problème de trois manières :

1. Stockage Creux (Sparse CSR) :
   Dans un texte, la majorité des mots ne cooccurrent jamais ensemble. La 
   matrice est donc remplie de zéros à plus de 95%. Plutôt que de stocker 
   des milliards de zéros inutiles, le format de compression de SciPy 
   enregistre uniquement les coordonnées (Ligne, Colonne) des cases non nulles.

2. Rapidité d'exécution (I/O) :
   Le format .npz est un fichier binaire compressé (une archive ZIP native 
   non décodée). Contrairement à un fichier texte (CSV, JSON), Python n'a 
   pas besoin de parser les lignes ni de convertir du texte en nombres. 
   Le chargement se fait par copie binaire directe du disque vers la mémoire 
   RAM, ce qui prend généralement moins d'une seconde.

3. Encapsulation Totale (Correction v4 - BUG-1) :
   Un fichier .npz peut contenir plusieurs sous-tableaux de données. La 
   version v4 du script utilise la fonction personnalisée `to_npz()` pour 
   sceller dans le même fichier binaire :
   - Les données structurelles de la matrice creuse (`data`, `indices`, `indptr`).
   - La liste brute des mots (`words`) préservant l'alignement exact des index.
   Cela élimine définitivement tout risque de décalage de vocabulaire lors 
   des phases d'extraction ultérieures (ex: ExtractWords.py).

Corrections et Évolutions de la version v4
──────────────────────────────────────────
BUG-1 : sp.save_npz(spec_path, mat_spec) ne stockait PAS le vocabulaire (words) 
  dans le fichier. Correction effectuée via l'usage de to_npz().
BUG-2 : sp.save_npz pouvait éliminer certains zéros de manière agressive selon la 
  version de SciPy. to_npz() sécurise la persistance brute.
BUG-3 : Ajustement des conditions logiques de réécriture pour éviter de bloquer 
  le recalcul si la matrice de spécificité existante provenait d'une ancienne version.
EVOL-1 : Remplacement du module d'embeddings classique par le pipeline dynamique 
  Classification_hdbscan (UMAP + t-SNE combiné à un clustering HDBSCAN auto-calibré).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import scipy.sparse as sp

from config import Config
from src.matrix.builder      import build_vocabulary, build_cooccurrence_matrix
from src.matrix.specificity  import compute_specificity
from src.matrix.export       import to_npz, load_npz
from src.matrix.proximity    import ProximityEngine
from src.analysis.submatrix  import extract_submatrix, load_submatrix
from src.analysis.afc         import run_afc
from src.analysis.graph      import build_graph
from src.analysis.kohonen    import run_kohonen
from src.matrix.utils import get_logger

# Liste brute des étapes disponibles dans le pipeline pour validation CLI.
ALL_STEPS: list[str] = ["corpus", "matrix", "submatrix", "afc", "hdbscan", "graph", "kohonen"]

# Système de journalisation (logger) propre à ce module.
_LOG = get_logger("main")


# ── Chemins dérivés ───────────────────────────────────────

def _npz_path(cfg: Config) -> Path:
    """Génère le chemin d'accès attendu pour la matrice de cooccurrence brute."""
    return Path(cfg.matrix_path).with_suffix(".npz")


def _spec_path(cfg: Config) -> Path:
    """Génère le chemin d'accès pour la matrice de spécificité (attractions/répulsions)."""
    return Path(cfg.matrix_path).parent / "matrix_specificity.npz"


def _freq_path(cfg: Config) -> Path:
    """Génère le chemin d'accès pour la matrice de fréquences brutes filtrées."""
    return Path(cfg.matrix_path).parent / "matrix_freq.npz"


# ── Classification & Clustering HDBSCAN (Remplacement de embeddings.py) ──────

def run_hdbscan_classification(cfg: Config, poles_label: str) -> None:
    """Exécute le pipeline de classification topologique et clustering HDBSCAN

    Calcule dynamiquement la taille des clusters selon la dimension de la matrice,
    puis exporte les projections comparatives, graphiques et rapports sémantiques.
    """
    import gc
    import glob
    import pandas as pd
    import umap
    import hdbscan
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import TruncatedSVD

    # ── Résolution robuste du chemin de la sous-matrice ──────────────────────
    chemin = cfg.submatrix_path()
    print(f"[*] [hdbscan] Chemin attendu de la sous-matrice : {chemin}")

    if not chemin.exists():
        print(f"[!] [hdbscan] Fichier attendu introuvable, recherche du fichier le plus récent dans {cfg.submatrix_dir}...")
        candidats = sorted(
            glob.glob(str(cfg.submatrix_dir / "themes_*.csv")),
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        if not candidats:
            raise FileNotFoundError(
                f"[hdbscan] Aucun fichier CSV trouvé dans {cfg.submatrix_dir}. "
                "Lancez d'abord : python main.py --steps submatrix"
            )
        chemin = Path(candidats[0])
        print(f"[~] [hdbscan] Fallback : utilisation du fichier le plus récent → {chemin}")
    else:
        print(f"[+] [hdbscan] Fichier trouvé → {chemin}")
    
    try:
        # Chargement via le moteur C ultra-rapide et nettoyage anti-bug 'barbarian'
        df = pd.read_csv(chemin, sep=';', index_col=0, engine='c')
        df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        noms_mots = df_numeric.columns.tolist()
        noms_docs = df_numeric.index.tolist()
        data_brute = df_numeric.values.astype(np.float32)
        
        print(f"[+] Données nettoyées : {len(noms_docs)} documents, {len(noms_mots)} termes.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du traitement du CSV : {e}")

    # Extraction des profils des mots (m_mots × n_docs)
    X_words = data_brute.T
    freq_absolues_mots = data_brute.sum(axis=0)
    
    # Normalisation relative des profils
    X_words_norm = X_words / (X_words.sum(axis=1, keepdims=True) + 1e-9)
    N_TERMES = len(noms_mots)
    
    # Règle d'agrégation dynamique selon la taille de la matrice
    K_CIBLE = 25              
    TAUX_BRUIT_ESTIME = 0.40  
    TAILLE_MIN_CLUSTER = max(20, int((N_TERMES * (1 - TAUX_BRUIT_ESTIME)) / K_CIBLE))
    ECHANTILLONS_MIN = max(5, int(TAILLE_MIN_CLUSTER * 0.10))
    
    print(f"[i] RÈGLE D'ÉCHELLE APPLIQUÉE ({N_TERMES} termes détectés) :")
    print(f"    -> 'min_cluster_size' calibré à : {TAILLE_MIN_CLUSTER} mots minimum par thème.")
    print(f"    -> 'min_samples' calibré à : {ECHANTILLONS_MIN} pour stabiliser les densités.")

    del data_brute, X_words
    gc.collect()

    # ──── A. PROJECTION UMAP ────
    print("[*] Calcul de la projection UMAP (Métrique Cosine)...")
    reducer_umap = umap.UMAP(n_neighbors=cfg.umap_neighbors, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embedding_umap = reducer_umap.fit_transform(X_words_norm)
    
    # ──── B. PROJECTION t-SNE OPTIMISÉE ────
    print("[*] Calcul de la projection t-SNE (Optimisée pour grands volumes)...")
    X_words_l2 = X_words_norm / (np.linalg.norm(X_words_norm, axis=1, keepdims=True) + 1e-9)
    
    if X_words_l2.shape[1] > 50:
        X_reduced = TruncatedSVD(n_components=50, random_state=42).fit_transform(X_words_l2)
    else:
        X_reduced = X_words_l2
        
    tsne = TSNE(n_components=2, perplexity=cfg.tsne_perplexity, random_state=42, init='pca')
    embedding_tsne = tsne.fit_transform(X_reduced)
    
    del X_words_norm, X_words_l2, X_reduced
    gc.collect()

    # ──── C. CLUSTERING HDBSCAN DYNAMIQUE ────
    print("[*] Exécution des deux clusterings HDBSCAN avec les règles d'échelle...")
    clusterer_umap = hdbscan.HDBSCAN(min_cluster_size=TAILLE_MIN_CLUSTER, min_samples=ECHANTILLONS_MIN, core_dist_n_jobs=-1)
    labels_umap = clusterer_umap.fit_predict(embedding_umap)
    
    clusterer_tsne = hdbscan.HDBSCAN(min_cluster_size=TAILLE_MIN_CLUSTER, min_samples=ECHANTILLONS_MIN, core_dist_n_jobs=-1)
    labels_tsne = clusterer_tsne.fit_predict(embedding_tsne)
    
    df_final = pd.DataFrame({
        'terme': noms_mots,
        'freq_totale': freq_absolues_mots,
        'umap_x': embedding_umap[:, 0],
        'umap_y': embedding_umap[:, 1],
        'classe_umap': labels_umap,
        'tsne_x': embedding_tsne[:, 0],
        'tsne_y': embedding_tsne[:, 1],
        'classe_tsne': labels_tsne
    })
    
    # ──── D. EXPORTS ET RAPPORTS SÉMANTIQUES ────
    csv_out = cfg.embeddings_dir / f"comparatif_structures_mots_{poles_label}.csv"
    print(f"[*] Sauvegarde de la base de données → {csv_out}")
    df_final.to_csv(csv_out, sep=';', index=False, encoding='utf-8')
    
    report_out = cfg.embeddings_dir / f"rapport_thematiques_detaille_{poles_label}.txt"
    print(f"[*] Rédaction du rapport sémantique → {report_out}")
    with open(report_out, "w", encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("EXPLORATION SÉMANTIQUE DES CLUSTERS OPTIMISÉS (UMAP)\n")
        f.write("==================================================\n\n")
        
        bruit_umap = len(df_final[df_final['classe_umap'] == -1])
        f.write(f"Mots isolés reclassés en bruit (-1) : {bruit_umap} / {len(df_final)}\n\n")
        
        classes_u = sorted([c for c in df_final['classe_umap'].unique() if c != -1])
        f.write(f"Nombre total de thématiques robustes générées : {len(classes_u)}\n\n")
        
        for cid in classes_u:
            mots_cluster = df_final[df_final['classe_umap'] == cid]
            mots_tries = mots_cluster.sort_values(by='freq_totale', ascending=False)['terme'].tolist()
            
            f.write(f"THEMATIQUE (CLUSTER UMAP) {cid} -- {len(mots_cluster)} mots :\n")
            f.write(f"  > Top mots-clés : {', '.join(mots_tries[:15])}\n\n")
            
    img_out = cfg.embeddings_dir / f"comparaison_topologique_umap_tsne_{poles_label}.png"
    print(f"[#] Génération du graphique comparatif côte-à-côte → {img_out}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    ax1.scatter(df_final['umap_x'], df_final['umap_y'], c=df_final['classe_umap'], cmap='Spectral', s=4, alpha=0.5)
    ax1.set_title("Topologie des termes via UMAP + HDBSCAN (Agrégé)", fontsize=12)
    ax1.set_xlabel("Dimension UMAP 1")
    ax1.set_ylabel("Dimension UMAP 2")
    
    ax2.scatter(df_final['tsne_x'], df_final['tsne_y'], c=df_final['classe_tsne'], cmap='Spectral', s=4, alpha=0.5)
    ax2.set_title("Topologie des termes via t-SNE + HDBSCAN (Agrégé)", fontsize=12)
    ax2.set_xlabel("Dimension t-SNE 1")
    ax2.set_ylabel("Dimension t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(img_out, dpi=150)
    plt.close()


# ── CLI ──────────────────────────────────────────

def parse_args(cfg: Config) -> argparse.Namespace:
    """Configure l'analyseur d'arguments en ligne de commande (CLI)."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse sémantique par cooccurrence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
                        metavar="STEP",
                        help=f"Étapes : {', '.join(ALL_STEPS)}")
    parser.add_argument("--words", nargs="+", default=cfg.target_words,
                        metavar="WORD",
                        help="Un ou plusieurs mots-pôles (ex: --words barbarian savage hun)")
    parser.add_argument("--top-n",             type=int, default=cfg.top_n)
    parser.add_argument("--min-occ",           type=int, default=cfg.min_occ)
    parser.add_argument("--force",             action="store_true",
                        help="Recalcule la matrice même si elle existe déjà")
    parser.add_argument("--metric",            choices=["cosine","pmi","raw","specificity"],
                        default=cfg.proximity_metric)
    parser.add_argument("--cosine-batch-size", type=int, default=cfg.cosine_batch_size)
    parser.add_argument("--pmi-min-count",     type=int, default=cfg.pmi_min_count)
    parser.add_argument("--no-cpp",            action="store_true",
                        help="Désactive l'extension C++ pour la spécificité")
    return parser.parse_args()


# ── Corpus ────────────────────────────────────────────────────────────────────

def load_tokens(cfg: Config) -> list[str]:
    """Charge le corpus textuel et extrait une liste propre de tokens (lemmes)."""
    from lexploreur.corpus import lexical_view
    corpus_path = Path(cfg.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus introuvable : {corpus_path}")

    print(f"[corpus] Chargement : {corpus_path}")
    df_lemmes = lexical_view(
        corpus_file=str(corpus_path),
        feature_to_extract="lemma",
        lowercase=True,
        exclude_pos=cfg.exclude_pos,
        exclude_lemmas=cfg.exclude_lemmes,
    )
    tokens = [lemme for liste in df_lemmes["lemma"] for lemme in liste]
    if not tokens:
        raise ValueError("Aucun token extrait après filtrage POS.")
    print(f"[corpus] {len(tokens):,} tokens extraits.")
    return tokens


# ── Chargement sparse de la matrice globale ───────────────────────────────────

def load_global_matrix(cfg: Config) -> tuple[sp.csr_matrix, list[str]]:
    """Charge en mémoire la matrice de cooccurrence compressée et son vocabulaire."""
    npz = _npz_path(cfg)
    if not npz.exists():
        raise FileNotFoundError(
            f"Matrice NPZ introuvable : {npz}\n"
            "Lancez d'abord : python main.py --steps matrix"
        )
    mat, words = load_npz(npz)
    print(f"[matrix] Chargement NPZ : {mat.shape[0]:,} mots, {mat.nnz:,} non-nuls")
    return mat, words


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal qui orchestre le cycle de vie complet du pipeline."""
    start_total = time.perf_counter()

    cfg  = Config()
    args = parse_args(cfg)

    cfg.target_words        = args.words
    cfg.top_n               = args.top_n
    cfg.min_occ             = args.min_occ
    cfg.proximity_metric    = args.metric
    cfg.cosine_batch_size   = args.cosine_batch_size
    cfg.pmi_min_count       = args.pmi_min_count
    cfg.use_cpp_specificity = not args.no_cpp

    cfg.makedirs()
    steps = args.steps

    poles_label = cfg._poles_label()

    sep = "─" * 60
    print(f"\n{sep}")
    print(f" Pipeline — étapes : {', '.join(steps)}")
    print(f" Mot(s)-pôle(s) : {cfg.target_words}  top_n={cfg.top_n}  min_occ={cfg.min_occ}")
    print(f" metric={cfg.proximity_metric}  cpp={'non' if args.no_cpp else 'oui'}")
    print(f"{sep}\n")

    # ── 1. Corpus ─────────────────────────────────────────────────
    tokens = None
    if "corpus" in steps:
        t0     = time.perf_counter()
        tokens = load_tokens(cfg)
        print(f"   [chrono] Corpus traité en {time.perf_counter() - t0:.2f}s")

    # ── 2. Matrice globale ────────────────────────────────────────
    if "matrix" in steps:
        t0        = time.perf_counter()
        npz       = _npz_path(cfg)
        spec_path = _spec_path(cfg)

        if npz.exists() and spec_path.exists() and not args.force:
            print(
                "[matrix] Matrices existantes trouvées — utilisez --force pour recalculer.\n"
                "         (nécessaire si specificity.py a été mis à jour)"
            )
        else:
            if tokens is None:
                tokens = load_tokens(cfg)

            word_to_id = build_vocabulary(tokens, min_freq=cfg.min_freq)
            words      = list(word_to_id.keys())
            mat_sparse = build_cooccurrence_matrix(tokens, word_to_id, window=cfg.window)

            # ── Matrice brute ─────────────────────────────────────
            npz.parent.mkdir(parents=True, exist_ok=True)
            to_npz(mat_sparse, npz, words)
            print(f"[matrix] Matrice brute NPZ → {npz} ({mat_sparse.nnz:,} non-nuls)")

            # ── Spécificité (v4) ──────────────────────────────────
            mat_spec = compute_specificity(
                mat_sparse, word_to_id,
                clip_max=cfg.specificity_clip_max,
                use_cpp=cfg.use_cpp_specificity,
                k_min=cfg.specificity_k_min,
                spec_threshold=0.0,   
            )

            n_pos = int((mat_spec.data > 0).sum())
            n_neg = int((mat_spec.data < 0).sum())
            print(
                f"[matrix] Spécificités calculées : "
                f"{n_pos:,} positives, {n_neg:,} négatives"
            )

            to_npz(mat_spec, spec_path, words)
            print(f"[matrix] Spécificité NPZ (avec vocabulaire) → {spec_path}")

            # ── Fréquences brutes filtrées sur le masque spec ─────
            freq_path = _freq_path(cfg)
            mask                = mat_spec.copy()
            mask.data           = np.ones_like(mask.data, dtype=np.float32)
            mat_freq_filtered   = sp.csr_matrix(mat_sparse.multiply(mask))
            to_npz(mat_freq_filtered, freq_path, words)
            print(f"[matrix] Fréquences (masque spec, {mat_freq_filtered.nnz:,} non-nuls) → {freq_path}")

        print(f"   [chrono] Étape Matrice terminée en {time.perf_counter() - t0:.2f}s")

    # ── 3. Sous-matrice ───────────────────────────────────────────
    df_focus = None
    if "submatrix" in steps:
        t0                = time.perf_counter()
        mat_global, words = load_global_matrix(cfg)

        spec_matrix = None
        if cfg.proximity_metric == "specificity":
            spec_path = _spec_path(cfg)
            if not spec_path.exists():
                raise FileNotFoundError(f"Matrice de spécificité introuvable : {spec_path}")
            spec_matrix, spec_words = load_npz(spec_path)
            if spec_words != words:
                raise RuntimeError("Désalignement du vocabulaire détecté entre les fichiers NPZ.")

        freq_mat    = None
        freq_path   = _freq_path(cfg)
        stat_matrix = getattr(cfg, "stat_matrix", "freq")
        if freq_path.exists():
            freq_mat, _ = load_npz(freq_path)

        df_focus = extract_submatrix(
            mat_global,
            words=words,
            target_word=cfg.target_words,
            output_path=cfg.submatrix_path(),
            top_n=cfg.top_n,
            min_occ=cfg.min_occ,
            metric=cfg.proximity_metric,
            batch_size=cfg.cosine_batch_size,
            pmi_min_count=cfg.pmi_min_count,
            spec_matrix=spec_matrix,
            spec_threshold=getattr(cfg, "spec_threshold", 2.0),
            freq_mat=freq_mat,
            stat_matrix=stat_matrix,
        )
        print(f"   [chrono] Sous-matrice extraite en {time.perf_counter() - t0:.2f}s")

    # ── 4. AFC ────────────────────────────────────────────────────
    if "afc" in steps:
        t0 = time.perf_counter()
        mat_global_afc, words_afc = load_global_matrix(cfg)
        spec_matrix_afc, _ = load_npz(_spec_path(cfg))
        freq_mat_afc        = None
        if _freq_path(cfg).exists():
            freq_mat_afc, _ = load_npz(_freq_path(cfg))

        stat_matrix_afc = getattr(cfg, "stat_matrix", "freq")
        if freq_mat_afc is None:
            stat_matrix_afc = "spec"

        afc_submatrix_path = cfg.afc_dir / f"submatrix_afc_{poles_label}.csv"

        df_afc = extract_submatrix(
            mat_global_afc,
            words=words_afc,
            target_word=cfg.target_words,
            output_path=afc_submatrix_path,
            top_n=cfg.top_n,
            min_occ=cfg.min_occ,
            metric=cfg.proximity_metric,
            batch_size=cfg.cosine_batch_size,
            pmi_min_count=cfg.pmi_min_count,
            spec_matrix=spec_matrix_afc,
            spec_threshold=getattr(cfg, "spec_threshold", 2.0),
            freq_mat=freq_mat_afc,
            stat_matrix=stat_matrix_afc,
            afc_mode=True,
        )

        run_afc(
            df_focus=df_afc,
            target_word=poles_label,
            afc_dir=cfg.afc_dir,
            n_components=cfg.n_components,
            cluster_cmap=cfg.afc_cluster_cmap,
        )
        print(f"   [chrono] AFC terminée en {time.perf_counter() - t0:.2f}s")

    # ── 5. Classification HDBSCAN (UMAP + t-SNE) ────────────────
    if "hdbscan" in steps:
        t0 = time.perf_counter()
        from src.analysis.Hdbscan import run_hdbscan
        run_hdbscan(cfg, poles_label)
        print(f"   [chrono] Classification HDBSCAN terminée en {time.perf_counter() - t0:.2f}s")

    # ── 6. Graphe ─────────────────────────────────────────────────
    if "graph" in steps:
        t0 = time.perf_counter()
        if df_focus is None:
            df_focus = load_submatrix(cfg.submatrix_path())
        hdbscan_csv_path = cfg.embeddings_dir / f"comparatif_structures_mots_{poles_label}.csv"
        build_graph(
            df_focus,
            target_word=poles_label,
            network_dir=cfg.network_dir,
            hdbscan_csv=hdbscan_csv_path if hdbscan_csv_path.exists() else None,
        )
        print(f"   [chrono] Graphe généré en {time.perf_counter() - t0:.2f}s")

    # ── 7. Kohonen ────────────────────────────────────────────────
    if "kohonen" in steps:
        t0 = time.perf_counter()
        if df_focus is None:
            df_focus = load_submatrix(cfg.submatrix_path())
        run_kohonen(
            df_focus,
            output_dir=cfg.kohonen_dir,
            label=f"kohonen_{poles_label}",
            target_word=poles_label,
            grid_shape=cfg.kohonen_grid_shape,
            sigma=cfg.kohonen_sigma,
            learning_rate=cfg.kohonen_learning_rate,
            n_iterations=cfg.kohonen_n_iterations,
            topology=cfg.kohonen_topology,
            normalize_data=cfg.kohonen_normalize,
            umatrix_cmap=cfg.kohonen_umatrix_cmap,
            top_n_labels=cfg.kohonen_top_n_labels,
            figsize=cfg.kohonen_figsize,
        )
        print(f"   [chrono] Carte de Kohonen terminée en {time.perf_counter() - t0:.2f}s")

    # ── Rapport final ─────────────────────────────────────────────
    end_total        = time.perf_counter()
    duration_sec     = end_total - start_total
    readable_duration = str(timedelta(seconds=int(duration_sec)))

    print(f"\n{sep}")
    print(f" Pipeline terminé en {readable_duration} ({duration_sec:.2f}s total).")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
