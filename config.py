"""
config.py
─────────
Configuration centralisée du pipeline d'analyse sémantique.
Tous les paramètres exposés ici sont surchargés par la CLI (main.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Répertoires ────────────────────────────────────────────────
    results_dir:   Path = Path("Results")
    submatrix_dir: Path = Path("Results/submatrix")
    afc_dir:       Path = Path("Results/AFC")
    network_dir:   Path = Path("Results/Network")
    embeddings_dir: Path = Path("Results/embeddings")
    kohonen_dir:   Path = Path("Results/Kohonen")

    # ── Fichiers racine ────────────────────────────────────────────
#    corpus_path: Path = Path("Corpus/Corpus_Sarkozy.json")
#    corpus_path: Path = Path("Corpus/debats_ivg_1974_Cleaned.json")
#    corpus_path: Path = Path("Corpus/corpus_barbar.json")
    corpus_path: Path = Path("Corpus/corpus_propre_barbar.json")
   
#    corpus_path: Path = Path("Corpus/corpus_filtre_lemmes.json")
    matrix_path: Path = Path("Results/matrix_general.csv")

    # ── Paramètres corpus ──────────────────────────────────────────
 #   min_freq: int = 5
    min_freq: int = 5
    window:   int = 40

    # ── Filtrage lexical ───────────────────────────────────────────
    exclude_pos: list[str] = field(default_factory=lambda: [
#        "PUNCT", "CCONJ", "DET", "ADP", "PRON", "PART",  "ADV",   "SCONJ", "PROPN"
        "PUNCT", "CCONJ", "DET", "ADP", "PRON", "PART",  "ADV",   "SCONJ"
    ])

#    exclude_lemmes: list[str] = field(default_factory=lambda: [
#        "être", "avoir", "faire", "ne", "à", "t", "m", "-", "oe","l", "p.", "iv-2", "lb","—","_", "\n", "\t", "«", "»", "—", "–", """, """,
#    "'", "'", "…", "|", "§", "¶", "°", "~", "„", "•", "r"
#    ])

    exclude_lemmes: list[str] = field(default_factory=lambda: [
	"me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now","historique","romain","lettre","ancien","études","et","est","qui","se","essai","si","romains","mythologie","pas","histoire","de","le","sur","politique","l'histoire","âge","nord","société","contre","ancienne","revue","gaule","française","vie","paris","chez","archéologique","quelques","rapport","documents","naissance","pierre","par","barbares","sociale","littérature","roi","droit","grande","mort","sous","français","societe","henri","à","french","pour","il","el","ne","cette","van","bibliothèque","copie","leur","papier","électronique","ces","d'un","être","conservation","haut","guizot","nous","roland","publie","tous","même","normandie","juridique","privé","chronique","fois","manuscrits","numérisation","gestion","francophone","données","comité","siècle","bibliothèque","sa","chanson","l'original","comité","recherches","médiévale","en","origines","recherche","lieux","michel","barbare","des","la","monde","au","du","les","moyen","romaine","temps","deux","langue","belge","d'un","que","sont","nationale","d'une","comme","f.m.","compte","québec","preuve","cet","bresc","dictionnaire","été","droz","peut","siècle","actes","livre","aux","d'une","à","une","y","numérique","âge","mais","authentique","d'apres","juin","danois","dans","nouvelle","xiie","siecle","pratique","entre","colloque","internationale","un","xve","travaux","mémoire","scientifique","littéraire","france","jean","ou","ce","dossier","xiiie","avec","premiere","mélanges","sicile","ils","forme","het","bibliotheque","archivistique","loi","javols","l'histoire","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",">","®",".","*","©","fol","/","<","í","]","gv","7v","î","p.","pp","ed","ii","ml","o","¿","em","m","ï","ii","-","'","fig","ge'ez","[","]","ml","9v","°","eld","em","m","»","ii","«","»","—","–","\u201c","\u201d","\u2018","\u2019","\u2026","|","§","¶","°","~","„","•","r","brill","press","edit","university","berkeley","oxford","cambridge","To","psychotherapist","mindfulness","gruyter","2001","archeologii","uniwersytetu","legutko","instytut","sestertii","wiadomoqci","kaczanowski","numizmatyczne","warszawskiego","einer","der","zum","krapivina","numismatic","perspektywy","legutko","instytut","kaczanowski","wiadomoqci","numizmatyczne","warszawskiego","rzymskich","rzymskiego","2012","25","ausdruck","polsce""aleksander","piotr","profesora","nad","polsce","kasimierza","warszawa","über","zu","von","al", "aus","ihre","ruth","schmidt","clausdieter","vch", "sprache","wörter", "jena","acta", "av","dumbarton","studien","nowak","princeton","τὸ","tutti","3/4","%","wörter","wortbildung","bezeichnungen","okresu","arbeiten","eine","ein","die","humaniora","hartwig","handbuch","zur","ernst","schott","heinrich","althochdeutschene", "wiegand","buddhist","buddhism","regulis","frans","donati","brugensis","metalogicon","gd","century","plr","goddas","martins","tommaso","carpegna","pedro","cédrik","turnhout","rich6","idealistic","csel","3rd"


])

    # ── Paramètres sous-matrice ────────────────────────────────────
    #
    # target_words accepte un ou plusieurs mots-pôles.
    # Chaque pôle sélectionne indépendamment ses top_n cooccurrents
    # les plus significatifs ; la sous-matrice est construite sur
    # l'union de toutes ces sélections individuelles.
#    target_words: list[str] = field(default_factory=lambda: ["barbarian"])
    target_words: list[str] = field(default_factory=lambda: [
#    "barbarian", "barbarians",  "barbarism","barbarous","barbaricum","barbarorum","barbarismus"
#    "barbarian", "barbarians",  "barbarism","barbarous"
    "barbarian", "barbarians","barbarians-","barbarians,”cause","barbarians,¹to","barbarians?promoter","barbarians.1","barbarians.118","barbarians.17","barbarians.2","barbarians.4","barbarians.41","barbarians.6","barbarians.68","barbarians.8","barbarians.91","barbarians.97","barbarians').25","barbarians').29","barbarians’(胡hu","barbarians”.16","barbarians”.23","barbarians”25","barbarians37"
    ])
#    target_words: list[str] = field(default_factory=lambda: ["femme"])

    top_n:    int = 300
    min_occ:  int = 45

    # ── Paramètres sous-matrice
    stat_matrix: str = "freq"   # 'freq' → fréquences brutes  (recommandé AFC)
                                              # 'spec' → spécificité       


    #   'specificity' : spécificité de Lafon (-log10 p-valeur hypergéométrique)
    proximity_metric:  str   = "specificity"    # 'cosine' | 'dice' | 'pmi' | 'raw' | 'specificity'
    cosine_batch_size: int   = 512
    pmi_min_count:     int   = 2
    spec_threshold:    float = 2         # seuil spécificité (si proximity_metric='specificity')

    # ── Spécificité ────────────────────────────────────────────────
    specificity_method:   str   = "hypergeom"  # 'hypergeom' | 'pmi'
    specificity_clip_max: float = 100.0
    specificity_k_min:    int   = 4           # paires avec k < k_min ignorées (k=1,2 → jamais spec ≥ 2.0)
    use_cpp_specificity:  bool  = True         # tente _specificity.so si True
    spec_threshold:    float = 8.0         # seuil spécificité (si proximity_metric='specificity')
    spec_threshold_auto:  bool  = False   # True → seuil déterminé par GMM sur la matrice

    # ── UMAP / t-SNE ──────────────────────────────────────────────
    umap_neighbors:  int   = 10     # n_neighbors pour UMAP (15 = bon équilibre local/global)
    tsne_perplexity: float = 30.0   # perplexité t-SNE (5–50 ; 30 est la valeur standard)

    # ── AFC────────────────────────────────────────
    n_components:           int   = 4
    afc_show_cos2_polarity: bool  = False
    afc_cos2_min_total:     float = 0.4        # 0.1 permissif · 0.4 strict
    afc_cluster_cmap: str = "tab10"  # si on veut plus de 10 couleurs ou peut écrire tab20 

    # ── Kohonen ─────────────────────────────────────────────────
    kohonen_grid_shape:    object = "auto"
    kohonen_sigma:         object = None
    kohonen_learning_rate: float  = 0.5
    kohonen_n_iterations:  int    = 10_000
    kohonen_topology:      str    = "rectangular"
    kohonen_normalize:     bool   = True
    kohonen_umatrix_cmap:  str    = "Blues"
    kohonen_top_n_labels:  int    = 3
    kohonen_figsize:       tuple  = field(default_factory=lambda: (16, 12))

    # ── Utilitaires ────────────────────────────────────────────────

    def makedirs(self) -> None:
        """Crée tous les répertoires de sortie si absents."""
        for d in (
            self.results_dir, self.submatrix_dir, self.afc_dir,
            self.network_dir, self.embeddings_dir, self.kohonen_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _poles_label(self) -> str:
        """Étiquette courte et sûre pour les noms de fichiers.

        Utilise uniquement le premier mot-pôle (ex: 'barbarian') pour éviter
        les noms de fichiers tronqués en milieu de token ou contenant des
        caractères illégaux issus des variantes typographiques.
        """
        import re
        # Premier mot-pôle uniquement, nettoyé des caractères non-alphanumériques
        label = re.sub(r"[^a-zA-Z0-9_-]", "", self.target_words[0])
        return label[:50] if label else "poles"

    def submatrix_path(self) -> Path:
        return (
            self.submatrix_dir
            / f"themes_{self._poles_label()}_n{self.top_n}_occ{self.min_occ}.csv"
        )

    def projection_path(self) -> Path:
        return self.embeddings_dir / f"projections_{self._poles_label()}.csv"

    def proximity_memory_estimate_mb(self, vocab_size: int) -> float:
        """RAM estimée (Mo) pour le calcul de proximité."""
        if self.proximity_metric == "cosine":
            return (self.cosine_batch_size * vocab_size * 4) / (1024 ** 2)
        return (2 * vocab_size * 8) / (1024 ** 2)
