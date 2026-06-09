"""
src/analysis/submatrix.py
──────────────────────────
Extraction d'une sous-matrice focalisée autour d'un mot-pôle
ou d'un vecteur de mots-pôles.

Métriques disponibles (paramètre metric)
─────────────────────────────────────────
'cosine'      : similarité de profil contextuel
'dice'        : Dice 2*cooc/(fi+fj)
'pmi'         : information mutuelle ponctuelle
'raw'         : cooccurrences brutes
'specificity' : spécificité de Lafon → matrice RECTANGULAIRE L1 × L2

Vecteur de pôles
─────────────────
target_word accepte indifféremment un str ou une list[str].
Quand plusieurs pôles sont fournis, chacun contribue indépendamment
ses top_n cooccurrents les plus significatifs ; la sous-matrice finale
est construite sur l'UNION de ces sélections individuelles.
Les pôles eux-mêmes sont exclus du résultat final.

Choix de la matrice projetée dans les analyses statistiques
────────────────────────────────────────────────────────────
Le paramètre `stat_matrix` contrôle quelle matrice est écrite comme
fichier principal (celui que l'AFC et les autres analyses consomment) :

  "spec"  → scores de spécificité -log10(p) signés (comportement historique)
  "freq"  → cooccurrences brutes entières           (recommandé pour l'AFC)

Dans les deux cas, la matrice complémentaire est sauvegardée en parallèle
sous le nom  <stem>_freq.csv  ou  <stem>_spec.csv  selon le choix fait.

Quand stat_matrix="freq", freq_mat est obligatoire (ValueError sinon).
Quand stat_matrix="spec",  freq_mat est optionnel (sauvegardée si fournie).

Seuil de spécificité automatique
──────────────────────────────────
Quand spec_threshold_auto=True, le seuil est déterminé automatiquement par
une GMM à 2 composantes sur la distribution des scores POSITIFS (attractions).
Le seuil sépare bruit et signal côté sur-représentation.
Le seuil_négatif symétrique est -spec_threshold pour les répulsions.

Mode AFC (afc_mode)
────────────────────
Quand afc_mode=True, le niveau 2 est ignoré et les colonnes sont
restreintes au niveau 1 uniquement.

Corrections v4
──────────────
- auto_spec_threshold : GMM sur scores > 0 uniquement (correct, la GMM
  porte sur les attractions ; les répulsions sont gérées par symétrie).
- _extract_specificity : la sélection du niveau 1 inclut désormais les
  répulsions (|score| >= spec_threshold, score positif OU négatif).
  Un paramètre include_repulsions contrôle ce comportement (défaut True).
- _extract_specificity : niveau 2 étendu aux répulsions des termes N1.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.matrix.proximity import ProximityEngine
from src.matrix.utils     import get_logger

_LOG = get_logger("analysis.submatrix")

# Valeurs autorisées pour stat_matrix
_STAT_MATRIX_VALUES = ("spec", "freq")


# ── Normalisation du/des pôle(s) ──────────────────────────────────────────────

def _as_poles(target_word: str | list[str]) -> list[str]:
    """Retourne toujours une liste de pôles (même pour un seul mot)."""
    if isinstance(target_word, str):
        return [target_word]
    return list(target_word)


# ── Seuil automatique par GMM ─────────────────────────────────────────────────

def auto_spec_threshold(spec_matrix: sp.csr_matrix) -> float:
    """
    Détermine le seuil de spécificité optimal par GMM à 2 composantes
    sur la distribution des scores POSITIFS (sur-représentation).

    Principe
    --------
    La distribution des scores positifs de Lafon est bimodale :
      composante basse  = bruit / cooccurrences faibles
      composante haute  = attractions réelles / signal

    Seuil retenu = μ_bruit + σ_bruit (frontière à 1 écart-type).
    Ce seuil s'applique symétriquement : positifs ≥ seuil et négatifs ≤ -seuil.

    Note : la GMM porte uniquement sur les scores > 0 car les attractions
    et les répulsions ont des distributions différentes. Le seuil négatif
    est obtenu par symétrie (-spec_threshold).

    Parameters
    ----------
    spec_matrix : matrice de spécificité globale (sparse, valeurs signées)

    Returns
    -------
    float — seuil positif arrondi à 2 décimales, loggé pour traçabilité
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        _LOG.warning(
            "scikit-learn non disponible pour GMM ; "
            "spec_threshold_auto ignoré, seuil par défaut 2.0 utilisé."
        )
        return 2.0

    # GMM sur les scores positifs uniquement (attractions)
    scores = np.asarray(spec_matrix.data, dtype=np.float32)
    scores = scores[scores > 0]   # correct : la GMM porte sur les attractions

    if len(scores) < 100:
        _LOG.warning(
            f"Trop peu de scores positifs ({len(scores)}) pour GMM. "
            "Seuil par défaut 2.0 utilisé."
        )
        return 2.0

    if len(scores) > 500_000:
        rng    = np.random.default_rng(42)
        scores = rng.choice(scores, size=500_000, replace=False)

    scores_2d = scores.reshape(-1, 1)
    gmm       = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(scores_2d)

    means              = gmm.means_.flatten()
    stds               = np.sqrt(gmm.covariances_.flatten())
    low_idx, high_idx  = np.argsort(means)

    mu_low  = float(means[low_idx])
    sd_low  = float(stds[low_idx])
    mu_high = float(means[high_idx])
    sd_high = float(stds[high_idx])

    threshold = round(mu_low + sd_low, 2)

    _LOG.info(
        f"GMM spec_threshold auto (scores positifs) :\n"
        f"  bruit  : μ={mu_low:.3f}  σ={sd_low:.3f}\n"
        f"  signal : μ={mu_high:.3f}  σ={sd_high:.3f}\n"
        f"  → seuil retenu = ±{threshold}  "
        f"(équivalent p < {10**(-threshold):.4f})"
    )
    return threshold


# ── API publique ───────────────────────────────────────────────────────────────

def extract_submatrix(
    mat_global: sp.csr_matrix | pd.DataFrame,
    target_word: str | list[str],
    output_path: Path,
    top_n: int = 50,
    min_occ: int = 2,
    metric: str = "cosine",
    batch_size: int = 512,
    pmi_min_count: int = 2,
    words: list[str] | None = None,
    spec_matrix: sp.csr_matrix | None = None,
    spec_threshold: float = 2.0,
    spec_threshold_auto: bool = False,
    freq_mat: sp.csr_matrix | None = None,
    stat_matrix: str = "freq",
    afc_mode: bool = False,
    include_repulsions: bool = True,
) -> pd.DataFrame:
    """
    Extrait la sous-matrice focalisée et la sauvegarde.

    target_word peut être un str ou une list[str].
    Quand c'est une liste, chaque pôle sélectionne indépendamment ses
    top_n cooccurrents ; la sous-matrice finale est l'union de ces sélections.

    Pour metric='specificity' :
      afc_mode=False (défaut) → matrice RECTANGULAIRE (pôles+N1) × (N1+N2)
      afc_mode=True           → matrice RÉDUITE       (pôles+N1) × N1

    Parameters
    ----------
    mat_global          : matrice globale de cooccurrence (sparse ou DataFrame)
    target_word         : mot-pôle ou liste de mots-pôles
    output_path         : chemin du fichier CSV principal
    top_n               : nombre max de termes de niveau 1 par pôle
    min_occ             : fréquence brute minimale pour les termes de niveau 2
    metric              : métrique de proximité
    batch_size          : taille des lots pour le calcul cosine sparse
    pmi_min_count       : seuil minimal pour le PMI
    words               : liste des mots (requis si mat_global est sparse)
    spec_matrix         : matrice de spécificité (requis si metric='specificity')
    spec_threshold      : seuil absolu — sélectionne spec ≥ +seuil ET ≤ -seuil
    spec_threshold_auto : si True, seuil calculé par GMM sur spec_matrix
    freq_mat            : matrice de fréquences brutes globale
    stat_matrix         : matrice principale — 'spec' ou 'freq'
    afc_mode            : si True, colonnes restreintes au niveau 1
    include_repulsions  : si True (défaut), inclut les termes avec score ≤ -seuil

    Returns
    -------
    pd.DataFrame — la matrice choisie par stat_matrix
    """
    if stat_matrix not in _STAT_MATRIX_VALUES:
        raise ValueError(
            f"stat_matrix doit être parmi {_STAT_MATRIX_VALUES} "
            f"(reçu : {stat_matrix!r})"
        )
    if stat_matrix == "freq" and freq_mat is None:
        raise ValueError(
            "stat_matrix='freq' requiert freq_mat. "
            "Fournissez la matrice de fréquences brutes ou "
            "utilisez stat_matrix='spec'."
        )

    # ── Normalisation de l'entrée ────────────────────────────────────
    if isinstance(mat_global, pd.DataFrame):
        words      = mat_global.index.tolist()
        mat_sparse = sp.csr_matrix(mat_global.values.astype(np.float32))
    else:
        if words is None:
            raise ValueError("words est requis quand mat_global est une csr_matrix")
        mat_sparse = mat_global

    word_to_idx = {w: i for i, w in enumerate(words)}
    poles       = _as_poles(target_word)

    poles_absents = [p for p in poles if p not in word_to_idx]
    if poles_absents:
        _LOG.warning(f"Pôle(s) absent(s) du vocabulaire, ignoré(s) : {poles_absents}")
    poles_valides = [p for p in poles if p in word_to_idx]
    if not poles_valides:
        raise ValueError(
            f"Aucun des pôles fournis n'est présent dans le vocabulaire "
            f"({len(words):,} mots). Pôles demandés : {poles}"
        )

    # ── Résolution du seuil de spécificité ──────────────────────────
    if metric == "specificity" and spec_threshold_auto:
        if spec_matrix is None:
            _LOG.warning(
                "spec_threshold_auto=True mais spec_matrix absent. "
                f"Seuil manuel {spec_threshold} utilisé."
            )
        else:
            spec_threshold = auto_spec_threshold(spec_matrix)
    else:
        _LOG.info(f"spec_threshold manuel : ±{spec_threshold}")

    # ── Dispatch selon la métrique ───────────────────────────────────
    if metric == "specificity":
        return _extract_specificity(
            words=words,
            word_to_idx=word_to_idx,
            poles=poles_valides,
            output_path=Path(output_path),
            spec_matrix=spec_matrix,
            spec_threshold=spec_threshold,
            top_n=top_n,
            min_occ=min_occ,
            freq_mat=freq_mat,
            stat_matrix=stat_matrix,
            afc_mode=afc_mode,
            include_repulsions=include_repulsions,
        )

    return _extract_proximity(
        mat_sparse=mat_sparse,
        words=words,
        word_to_idx=word_to_idx,
        poles=poles_valides,
        output_path=Path(output_path),
        top_n=top_n,
        min_occ=min_occ,
        metric=metric,
        batch_size=batch_size,
        pmi_min_count=pmi_min_count,
        freq_mat=freq_mat,
        stat_matrix=stat_matrix,
    )


def load_submatrix(path: Path) -> pd.DataFrame:
    """Charge une sous-matrice depuis son CSV (séparateur ';')."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sous-matrice introuvable : {path}\n"
            "Lancez d'abord l'étape 'submatrix' via main.py."
        )
    _LOG.info(f"Chargement sous-matrice : {path}")
    return pd.read_csv(path, sep=";", index_col=0)


def load_submatrix_pair(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Charge la paire (matrice principale, matrice complémentaire).

    La matrice complémentaire est cherchée sous <stem>_freq.csv puis
    <stem>_spec.csv. Retourne (df_main, None) si elle est absente.
    """
    path    = Path(path)
    df_main = load_submatrix(path)

    for tag in ("_freq", "_spec"):
        companion = path.with_name(path.stem + tag + path.suffix)
        if companion.exists():
            _LOG.info(f"Chargement matrice complémentaire : {companion}")
            return df_main, load_submatrix(companion)

    _LOG.info("Aucune matrice complémentaire trouvée.")
    return df_main, None


# ── Extraction par proximité (cosine / dice / pmi / raw) ──────────────────────

def _extract_proximity(
    mat_sparse: sp.csr_matrix,
    words: list[str],
    word_to_idx: dict[str, int],
    poles: list[str],
    output_path: Path,
    top_n: int,
    min_occ: int,
    metric: str,
    batch_size: int,
    pmi_min_count: int,
    freq_mat: sp.csr_matrix | None,
    stat_matrix: str,
) -> pd.DataFrame:
    """
    Sélectionne les termes par proximité puis extrait les deux matrices
    (spécificité + fréquences) sur le même jeu de termes.
    """
    poles_set = set(poles)

    # ── Niveau 1 : top_n cooccurrents par pôle, puis union ──────────
    union_niveau1: list[str] = []
    seen: set[str]           = set(poles)

    for pole in poles:
        engine = ProximityEngine.from_sparse(mat_sparse, words, pole)
        scores = engine.compute(
            metric=metric,
            batch_size=batch_size,
            pmi_min_count=pmi_min_count,
        )
        scores_filtered = scores.drop(
            labels=[p for p in poles if p in scores.index],
            errors="ignore",
        )
        top_pole = (
            scores_filtered
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        nouveaux = [t for t in top_pole if t not in seen]
        union_niveau1.extend(nouveaux)
        seen.update(nouveaux)
        _LOG.info(
            f"Pôle {pole!r} → {len(top_pole)} cooccurrents "
            f"(+{len(nouveaux)} nouveaux, union N1={len(union_niveau1)})"
        )

    if not union_niveau1:
        raise ValueError(f"Aucun cooccurrent trouvé pour les pôles : {poles}.")

    _LOG.info(f"Niveau 1 (union) : {len(union_niveau1)} termes distincts")

    # ── Niveau 2 : cooccurrents des termes de niveau 1 ──────────────
    compteur: dict[str, int] = {}
    for terme in union_niveau1:
        if terme not in word_to_idx:
            continue
        i   = word_to_idx[terme]
        row = mat_sparse[i].toarray().flatten()
        col = mat_sparse[:, i].toarray().flatten()
        avg = (row + col) / 2.0
        for j in np.argsort(avg)[::-1][:top_n]:
            v = words[j]
            if v not in seen:
                compteur[v] = compteur.get(v, 0) + 1

    niveau2 = [t for t, c in compteur.items() if c >= min_occ]
    _LOG.info(f"Niveau 2 : {len(niveau2)} termes additionnels (min_occ={min_occ})")

    tous_termes = list(dict.fromkeys(poles + union_niveau1 + niveau2))
    tous_termes = [t for t in tous_termes if t in word_to_idx]

    if len(tous_termes) < 2:
        raise ValueError(f"Sous-matrice vide pour les pôles : {poles}.")

    idx_sel  = [word_to_idx[t] for t in tous_termes]
    spec_sub = mat_sparse[idx_sel, :][:, idx_sel].toarray().astype(np.float32)
    df_spec  = pd.DataFrame(spec_sub, index=tous_termes, columns=tous_termes)

    df_freq: pd.DataFrame | None = None
    if freq_mat is not None:
        freq_sub = freq_mat[idx_sel, :][:, idx_sel].toarray().astype(np.float32)
        df_freq  = pd.DataFrame(freq_sub, index=tous_termes, columns=tous_termes)

    df_main, df_companion, companion_tag = _resolve_outputs(df_spec, df_freq, stat_matrix)
    _write_pair(df_main, df_companion, output_path, companion_tag)
    return df_main


# ── Extraction par spécificité (matrice rectangulaire) ────────────────────────

def _extract_specificity(
    words: list[str],
    word_to_idx: dict[str, int],
    poles: list[str],
    output_path: Path,
    spec_matrix: sp.csr_matrix | None,
    spec_threshold: float,
    top_n: int,
    min_occ: int,
    freq_mat: sp.csr_matrix | None,
    stat_matrix: str,
    afc_mode: bool = False,
    include_repulsions: bool = True,
) -> pd.DataFrame:
    """
    Extraction par spécificité de Lafon — inclut attractions ET répulsions.

    Sélection du niveau 1
    ─────────────────────
    Attractions  (score ≥ +spec_threshold) : termes significativement
                 sur-représentés avec le pôle — toujours inclus.
    Répulsions   (score ≤ -spec_threshold) : termes significativement
                 sous-représentés avec le pôle — inclus si include_repulsions=True.

    Les top_n termes sont sélectionnés indépendamment par signe :
      - top_n/2 attractions (scores les plus positifs)
      - top_n/2 répulsions  (scores les plus négatifs)
    Si include_repulsions=False, top_n attractions uniquement.

    afc_mode=False (défaut) — matrice RECTANGULAIRE complète
      Lignes   = pôles + niveau 1
      Colonnes = niveau 1 + niveau 2

    afc_mode=True — matrice RÉDUITE pour l'AFC
      Lignes   = pôles + niveau 1
      Colonnes = niveau 1 UNIQUEMENT
    """
    if spec_matrix is None:
        raise ValueError(
            "metric='specificity' requiert spec_matrix. "
            "Vérifiez que spec_path est fourni dans main.py."
        )

    poles_set = set(poles)

    # Quota par signe quand les deux sont demandés
    top_n_pos = top_n // 2 if include_repulsions else top_n
    top_n_neg = top_n - top_n_pos if include_repulsions else 0

    # ── Niveau 1 : un top_n par pôle, puis union ─────────────────────
    union_niveau1: list[str] = []
    seen: set[str]           = set(poles)

    for pole in poles:
        pole_idx = word_to_idx[pole]
        pole_row = np.asarray(spec_matrix[pole_idx].todense()).flatten()

        nouveaux_pole: list[str] = []

        # ── Attractions : scores ≥ +spec_threshold ────────────────────
        idx_pos    = np.where(pole_row >= spec_threshold)[0]
        scores_pos = pole_row[idx_pos]
        order_pos  = np.argsort(scores_pos)[::-1][:top_n_pos]
        top_pos    = [
            words[idx_pos[k]] for k in order_pos
            if words[idx_pos[k]] not in poles_set
        ]
        nouveaux_pos = [t for t in top_pos if t not in seen]
        nouveaux_pole.extend(nouveaux_pos)

        # ── Répulsions : scores ≤ -spec_threshold ────────────────────
        n_rep = 0
        if include_repulsions and top_n_neg > 0:
            idx_neg    = np.where(pole_row <= -spec_threshold)[0]
            scores_neg = pole_row[idx_neg]
            order_neg  = np.argsort(scores_neg)[:top_n_neg]  # croissant → plus négatif d'abord
            top_neg    = [
                words[idx_neg[k]] for k in order_neg
                if words[idx_neg[k]] not in poles_set
            ]
            nouveaux_neg = [t for t in top_neg if t not in seen]
            nouveaux_pole.extend(nouveaux_neg)
            n_rep = len(nouveaux_neg)

        # Mise à jour de l'union
        for t in nouveaux_pole:
            if t not in seen:
                union_niveau1.append(t)
                seen.add(t)

        _LOG.info(
            f"Pôle {pole!r} → "
            f"{len(nouveaux_pos)} attractions (spec ≥ +{spec_threshold})"
            + (f", {n_rep} répulsions (spec ≤ -{spec_threshold})" if include_repulsions else "")
            + f" | union N1={len(union_niveau1)}"
        )

    if not union_niveau1:
        raise ValueError(
            f"Aucun cooccurrent spécifique pour les pôles {poles} "
            f"avec spec_threshold=±{spec_threshold}. "
            "Essayez un seuil plus bas (ex. 1.3 → p < 0.05) "
            "ou activez spec_threshold_auto=True."
        )

    _LOG.info(f"Niveau 1 (union) : {len(union_niveau1)} termes distincts")

    # ── Niveau 2 — ignoré si afc_mode=True ───────────────────────────
    if afc_mode:
        niveau2: list[str] = []
        _LOG.info(
            "afc_mode=True : niveau 2 ignoré — "
            f"colonnes restreintes aux {len(union_niveau1)} termes du niveau 1"
        )
    else:
        candidats: dict[str, int] = {}
        for terme in union_niveau1:
            i        = word_to_idx[terme]
            row_spec = np.asarray(spec_matrix[i].todense()).flatten()

            # Candidats N2 : attractions ET répulsions des termes N1
            idx_sig = np.where(
                (row_spec >= spec_threshold) |
                (include_repulsions & (row_spec <= -spec_threshold))
            )[0]
            for j in idx_sig:
                v = words[j]
                if v not in seen:
                    candidats[v] = candidats.get(v, 0) + 1

        if freq_mat is not None:
            niveau2 = []
            for t, _count in candidats.items():
                if t not in word_to_idx:
                    continue
                freq_t = int(freq_mat[word_to_idx[t]].sum())
                if freq_t >= min_occ:
                    niveau2.append(t)
            _LOG.info(
                f"Niveau 2 : {len(niveau2)} termes "
                f"(|spec| ≥ {spec_threshold} ET freq ≥ {min_occ})"
            )
        else:
            niveau2 = [t for t, c in candidats.items() if c >= min_occ]
            _LOG.info(
                f"Niveau 2 : {len(niveau2)} termes additionnels "
                f"(min_occ={min_occ} listes, freq_mat absent)"
            )

    # ── Index ligne / colonne ─────────────────────────────────────────
    lignes   = [t for t in poles + union_niveau1 if t in word_to_idx]
    colonnes = list(dict.fromkeys(
        [t for t in union_niveau1 + niveau2 if t in word_to_idx]
    ))

    if len(lignes) < 2 or not colonnes:
        raise ValueError(f"Sous-matrice vide pour les pôles : {poles}.")

    idx_lig = [word_to_idx[t] for t in lignes]
    idx_col = [word_to_idx[t] for t in colonnes]

    # ── Extraction des deux matrices sur la même sélection ───────────
    spec_sub = spec_matrix[idx_lig, :][:, idx_col].toarray().astype(np.float32)
    df_spec  = pd.DataFrame(spec_sub, index=lignes, columns=colonnes)

    df_freq: pd.DataFrame | None = None
    if freq_mat is not None:
        freq_sub = freq_mat[idx_lig, :][:, idx_col].toarray().astype(np.float32)
        df_freq  = pd.DataFrame(freq_sub, index=lignes, columns=colonnes)

    df_main, df_companion, companion_tag = _resolve_outputs(df_spec, df_freq, stat_matrix)
    _write_pair(df_main, df_companion, output_path, companion_tag)

    mode_label = "AFC (N1 seulement)" if afc_mode else "complète (N1+N2)"
    _LOG.info(
        f"Sous-matrice spécificité {mode_label} "
        f"({df_main.shape[0]} lignes × {df_main.shape[1]} colonnes) "
        f"→ {output_path}"
    )
    return df_main


# ── Helpers d'écriture ────────────────────────────────────────────────────────

def _resolve_outputs(
    df_spec: pd.DataFrame,
    df_freq: pd.DataFrame | None,
    stat_matrix: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """
    stat_matrix='freq' → principal=freq,  annexe=spec  (_spec.csv)
    stat_matrix='spec' → principal=spec,  annexe=freq  (_freq.csv)
    """
    if stat_matrix == "freq":
        return df_freq, df_spec, "_spec"   # type: ignore[return-value]
    else:
        return df_spec, df_freq, "_freq"


def _write_pair(
    df_main: pd.DataFrame,
    df_companion: pd.DataFrame | None,
    output_path: Path,
    companion_tag: str,
) -> None:
    """
    Écrit df_main → output_path  et  df_companion → <stem><tag>.csv.
    Séparateur ';' dans les deux cas.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_main.to_csv(output_path, sep=";")
    stat_label = "freq" if companion_tag == "_spec" else "spec"
    _LOG.info(
        f"Matrice principale [{stat_label}] "
        f"({df_main.shape[0]}×{df_main.shape[1]}) → {output_path}"
    )

    if df_companion is not None:
        companion_path = output_path.with_name(
            output_path.stem + companion_tag + output_path.suffix
        )
        df_companion.to_csv(companion_path, sep=";")
        companion_label = companion_tag.lstrip("_")
        _LOG.info(
            f"Matrice complémentaire [{companion_label}] "
            f"({df_companion.shape[0]}×{df_companion.shape[1]}) "
            f"→ {companion_path}"
        )
