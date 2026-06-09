import numpy as np
import scipy.sparse as sp
import argparse

def extraire_et_sauvegarder(fichier_matrice, fichier_vocabulaire, lemme, seuil, mode, top_n):
    try:
        print(f"Chargement des fichiers...")
        words = np.load(fichier_vocabulaire, allow_pickle=True)['words'].tolist()
        data_spec = np.load(fichier_matrice, allow_pickle=True)
        matrice = sp.csr_matrix(
            (data_spec['data'], data_spec['indices'], data_spec['indptr']),
            shape=tuple(data_spec['shape'])
        )

        if lemme not in words:
            print(f"Erreur : Lemme '{lemme}' introuvable.")
            return

        idx = words.index(lemme)
        colonne = matrice[:, idx].toarray().flatten()

        # Filtrage par seuil : positifs au-dessus, négatifs en-dessous
        if mode == 'positif':
            resultats = [
                (words[i], float(colonne[i]))
                for i in range(len(words))
                if colonne[i] >= seuil and i != idx        # ✅ >= +seuil
            ]
            resultats.sort(key=lambda x: x[1], reverse=True)

        else:  # mode negatif
            resultats = [
                (words[i], float(colonne[i]))
                for i in range(len(words))
                if colonne[i] <= -seuil and i != idx       # ✅ <= -seuil
            ]
            resultats.sort(key=lambda x: x[1])             # croissant (plus négatif en premier)

        if top_n:
            resultats = resultats[:top_n]

        fichier_sortie = f"resultats_{lemme}_{mode}.txt"
        with open(fichier_sortie, 'w', encoding='utf-8') as f:
            f.write(f"Lemme: {lemme} | Mode: {mode} | Seuil: ±{seuil}\n")
            f.write(f"{'Terme':<30} {'Score':>10}\n")
            f.write("-" * 42 + "\n")
            for mot, score in resultats:
                f.write(f"{mot:<30} {score:>10.4f}\n")

        print(f"Succès : {len(resultats)} termes enregistrés dans '{fichier_sortie}'")

    except Exception as e:
        print(f"Erreur : {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction de cooccurrents par spécificité")
    parser.add_argument("lemme", help="Le terme à rechercher")
    parser.add_argument("--seuil", type=float, default=1.0,
                        help="Seuil absolu : garde scores >= +seuil (positif) ou <= -seuil (negatif)")
    parser.add_argument("--mode", choices=['positif', 'negatif'],   # ← sans accent
                        default='positif', help="Filtrer les associations positives ou négatives")
    parser.add_argument("--top", type=int, default=None, help="Nombre max de résultats")

    args = parser.parse_args()
    extraire_et_sauvegarder(
        'Results/matrix_specificity.npz',
        'Results/matrix_general.npz',
        args.lemme, args.seuil, args.mode, args.top
    )
