import matplotlib.pyplot as plt
import json
from IPython.display import Markdown


def plot_tm(lda_model, nwords=10, nrows=5, ncols=2):
    """
    Plot the top words for each topic in a Latent Dirichlet Allocation (LDA) model computed by gensim.

    Parameters
    ----------
    lda_model : gensim.models.LdaModel
        A trained LDA model from the Gensim library. The model should have been fitted to a corpus
        and should contain topics from which to extract words
    nwords : int, optional
        The number of top words to display for each topic. Default is 10.
    nrows : int, optional
        The number of rows in the subplot grid. Default is 5.
    ncols : int, optional
        The number of columns in the subplot grid. Default is 2.

    Returns
    -------
    None
        This function does not return any value. It displays a bar chart for each topic.

    Example
    -------
    To plot the top 15 words for each topic in a 3x5 grid:
    
    >>> plot_tm(nwords=15, nrows=3, ncols=5)
    """
    # Récupérer le nombre total de topics
    num_topics = lda_model.num_topics
    
    # Créer une figure avec une grille de subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 15))
    axes = axes.flatten()

    for topic_id in range(num_topics):
        # Récupérer les termes du topic
        terms = lda_model.get_topic_terms(topic_id, nwords)
        
        # Extraire les mots et leurs probabilités
        words = [lda_model.id2word[term[0]] for term in terms]
        probabilities = [term[1] for term in terms]
        
        # Créer un barplot
        axes[topic_id].barh(words, probabilities, color='darksalmon')
        axes[topic_id].set_title(f'Topic {topic_id}')
        axes[topic_id].invert_yaxis()  # Inverser l'axe y pour avoir le mot le plus probable en haut
        axes[topic_id].set_xlabel('Probabilité')
        axes[topic_id].set_ylabel('Mots')

    plt.tight_layout()
    plt.show()

# from gensim import corpora
# from gensim.models import LdaModel
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis

# def get_lda_topics(voc, num_topics, stopwords=None):
#     if stopwords is not None:
#         voc = voc.apply(lambda tokens: [token for token in tokens if token not in stopwords])

#     dictionary = corpora.Dictionary(voc)
#     corpus = [dictionary.doc2bow(tokens) for tokens in voc]
#     lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

#     return lda_model, corpus, dictionary


# def visualize_lda_model(lda_model, corpus, dictionary, output_path):
#     """
#     Fonction pour visualiser un modèle LDA avec PyLDAvis et enregistrer la visualisation dans un fichier HTML.
#     """
#     pyLDAvis.enable_notebook()
    
#     vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
#     pyLDAvis.save_html(vis_data, output_path)

#     return vis_data


def kwic_doc_topic(corpus_file, lda_model, id_doc, id_topic, dictionary):
    """
    Generate a Keyword in Context (KWIC) representation for a specific document and topic.

    This function highlights the tokens in a document that correspond to the top words of a specified topic
    from a Latent Dirichlet Allocation (LDA) model trained with gensim. The highlighted tokens are formatted in bold. Work only in Jupyter notebook.

    Parameters
    ----------
    corpus_file : str
        The path to a JSON file containing the corpus. The JSON should be structured such that each document
        can be accessed by its ID.

    lda_model : gensim.models.LdaModel
        A trained LDA model from the Gensim library. The model should contain topics from which to extract
        the top words.

    id_doc : int
        The ID of the document to be processed. This ID is used to retrieve the specific document from the corpus.

    id_topic : int
        The ID of the topic for which the top words are to be extracted. This ID corresponds to the topic in the LDA model.

    dictionary : gensim.corpora.Dictionary
        A Gensim dictionary mapping word IDs to words. This is used to convert topic word IDs to their corresponding
        string representations.

    Returns
    -------
    Markdown
        A Markdown object containing the document's tokens, with tokens corresponding to the topic's top words
        highlighted in bold.
    """

    # On définit la liste des 50 mots les plus fréquents d'un topic donné
    topic_words = lda_model.get_topic_terms(id_topic, 50)
    kw = [dictionary[word_id] for word_id, freq in topic_words]

    # On charge le json du corpus
    with open(corpus_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # On récupère le doc
    document = data[id_doc]
    
    # Tokens du document
    tokens = document['document']['lexical_features']
    
    # Initialiser une liste pour stocker les tokens formatés
    formatted_tokens = []
    
    # On parcourt chaque token
    for token_info in tokens:
        token = token_info['token']
        lemma = token_info['lemma']
        
        # Si le lemme du doc est dans la liste des lemmes du topic
        if lemma in kw:
            # On formate le token en gras
            formatted_tokens.append(f"**{token}**")
        else:
            # Sinon on Ajoute le token sans formatage
            formatted_tokens.append(token)
    
    # Joindre tous les tokens en une seule chaîne
    return Markdown(' '.join(formatted_tokens))
