from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


def top_features_tfidf(tfidf_df, top_n=10):
    """
    Finds the top n features (words) by TF-IDF score for each document in the given DataFrame.

    Parameters
    ----------
    top_n : int, optional
        The number of top features to retrieve for each document. Default is 10.
    tfidf_df : pandas.DataFrame
        A DataFrame containing the TF-IDF scores for each feature in each document, with the document as the rows and the feature names as the columns.

    Returns
    -------
    None
        The function prints the top n features and their corresponding TF-IDF scores for each document.
    """

    top_words = {}

    for i in range(tfidf_df.shape[0]):
        # On récupère les tfidf du document correspondant à l'itération
        doc_tfidf = tfidf_df.iloc[i]
    
        # On trie les features par tfidf et récupère les N premiers
        top_terms = doc_tfidf.nlargest(top_n)
    
        # On stocke les résultats avec l'index du DataFrame initial et les scores tfidf
        top_words[tfidf_df.index[i]] = top_terms.to_dict()

        # Afficher les résultats
    for index, terms in top_words.items():
        print(f"Document index {index}:")
        for word, score in terms.items():
            print(f"  {word}: {score:.4f}")


def collocations_freq(documents, collocations):
    """
    Count the frequency of collocations detected by gensim with Phrases().

    Parameters
    ----------
    documents : pandas.series
        A pandas serie containing list of tokens for each document.

    collocations : gensim.models.phrases.Phrases
        Object of class Phrases from gensim.

    Returns
    -------
    dict
        A dictionary where keys are collocations and values are their corr
    esponding frequencies.
    """

    collocations_list = []
    
    for doc in documents:
        collocations_doc = collocations[doc]
        for tok in collocations_doc:
            if "_" in tok:
                collocations_list.append(tok)

    collocation_freq = Counter(collocations_list)

    # print(collocation_freq)

    return dict(collocation_freq.most_common())


def dice_scores(coocs_mat, feature_names):
    # On convertit la matrice initiale en csr (format matrice creuse scipy)
    coocs_mat = csr_matrix(coocs_mat)
                           
    
    # On utilise la propriété .A1 de Numpy pour
    # obtenir un 1D array avec la somme de chaque ligne
    f = coocs_mat.sum(axis=1).A1
    
    # Récupération des indices non nuls et des valeurs associées.
    i_indices, j_indices = coocs_mat.nonzero()
    values = coocs_mat.data

    # Calcul vectorisé du dénominateur pour chaque non-zéro (f[i] + f[j])
    denom = f[i_indices] + f[j_indices]
    
    # Calcul vectorisé de l'indice de Dice score
    # On évite la division par zéro via l'argument "where" de np.divide()
    dice_data = np.divide(2 * values, denom, out=np.zeros_like(values, dtype=np.float64), where=denom != 0)

    # Création de la nouvelle matrice de cooccurrence avec les scores de Dice.
    dice_matrix = csr_matrix((dice_data, (i_indices, j_indices)), shape=coocs_mat.shape)

    # On convertit le résultat en dataframe avec les noms des mots en index
    # fonction dédiée de pandas au foramt sparse matrix
    dice_df = pd.DataFrame.sparse.from_spmatrix(dice_matrix, index=feature_names, columns=feature_names)
    
    return dice_df
