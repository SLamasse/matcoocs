import json
import pandas as pd
import re

def export_tdm(corpus, dtm, part):
    dtm[part] = corpus[part]
    count_df_grouped = dtm.groupby(part).sum().T
    count_df_grouped.to_csv('tdm.csv', index=True)

    return count_df_grouped


def nothing(doc):
    return doc


def ngrams(tokens, n=2, sep=' ', stopwords=set()):
    return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])
        if len([t for t in ngram if t in stopwords])==0]


# def keywords_in_context(json_file, pivot, field, window_size_left=5, window_size_right=5, doc_desc=None, regex=False):
#     """
#     Extracts the context (left and right) of a given keyword or phrase from a corpus of documents stored in a JSON file.

#     Parameters
#     ----------
#     json_file : str
#         Path to the JSON file containing the corpus data.
#     pivot : str
#         The keyword or phrase to search for in the corpus.
#     field : str
#         The lexical feature to use for the search, can be 'token', 'lemma', or 'pos'.
#     window_size_left : int, optional
#         The number of tokens to include in the left context. Default is 5.
#     window_size_right : int, optional
#         The number of tokens to include in the right context. Default is 5.
#     doc_desc : str, optional
#         Metadata used in the json file to describe documents.
#     regex : bool, optional
#         If True, pivot is compiled as a regex.

#     Returns
#     -------
#     pandas.DataFrame
#         A DataFrame containing the left context, the keyword or phrase, and the right context for each occurrence of the pivot in the corpus.

#     Raises
#     ------
#     ValueError
#         If the `field` parameter is not 'token', 'lemma', or 'pos'.
#     """

#     if field not in ['token', 'lemma', 'pos']:
#         raise ValueError("Le champ doit être 'token', 'lemma' ou 'pos'")

#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     results = []

#     if regex:
#         pivot_patterns = [re.compile(pattern) for pattern in pivot_words]
    
#     for document in data:

#         lexical_features = document['document']['lexical_features']
#         pivot_words = pivot.split()
#         for i in range(len(lexical_features) - len(pivot_words) + 1):
#             if all(lexical_features[i+j][field] == pivot_words[j] for j in range(len(pivot_words))):
#                 left_context = ' '.join([f['token'] for f in lexical_features[max(0, i-window_size_left):i]])
#                 right_context = ' '.join([f['token'] for f in lexical_features[i+len(pivot_words):min(len(lexical_features), i+len(pivot_words)+window_size_right)]])

#                 result = {
#                         'contexte_gauche': left_context,
#                         'mot_pivot': ' '.join([f['token'] for f in lexical_features[i:i+len(pivot_words)]]),
#                         'contexte_droit': right_context
#                     }
#                 if doc_desc is not None:
#                     doc_value = document['document'][doc_desc]
#                     result[doc_desc] = doc_value

#                 results.append(result)
#     return pd.DataFrame(results)

def keywords_in_context(json_file, pivot, field, window_size_left=5, window_size_right=5, doc_desc=None, regex=False):
    """
    Extracts the context (left and right) of a given keyword or phrase from a corpus of documents stored in a JSON file.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing the corpus data.
    pivot : str
        The keyword or phrase to search for in the corpus.
    field : str
        The lexical feature to use for the search, can be 'token', 'lemma', or 'pos'.
    window_size_left : int, optional
        The number of tokens to include in the left context. Default is 5.
    window_size_right : int, optional
        The number of tokens to include in the right context. Default is 5.
    doc_desc : str, optional
        Metadata used in the json file to describe documents.
    regex : bool, optional
        If True, pivot is compiled as a regex.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the left context, the keyword or phrase, and the right context for each occurrence of the pivot in the corpus.

    Raises
    ------
    ValueError
        If the `field` parameter is not 'token', 'lemma', or 'pos'.
    """

    if field not in ['token', 'lemma', 'pos']:
        raise ValueError("Le champ doit être 'token', 'lemma' ou 'pos'")

    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []

    if regex:
        pivot_pattern = re.compile(pivot)

    for document in data:
        lexical_features = document['document']['lexical_features']
        pivot_words = pivot.split()

        if regex:
            for i in range(len(lexical_features)):
                match = pivot_pattern.match(lexical_features[i][field])
                if match:
                    left_context = ' '.join([f['token'] for f in lexical_features[max(0, i-window_size_left):i]])
                    right_context = ' '.join([f['token'] for f in lexical_features[i+1:min(len(lexical_features),
                                                                                           i+1+window_size_right)]])
                    
                    result = {
                        'contexte_gauche': left_context,
                        'mot_pivot': lexical_features[i][field],
                        'contexte_droit': right_context
                    }
                    if doc_desc is not None:
                        doc_value = document['document'][doc_desc]
                        result[doc_desc] = doc_value

                    results.append(result)
        else:
            for i in range(len(lexical_features) - len(pivot_words) + 1):
                if all(lexical_features[i+j][field] == pivot_words[j] for j in range(len(pivot_words))):
                    left_context = ' '.join([f['token'] for f in lexical_features[max(0, i-window_size_left):i]])
                    right_context = ' '.join([f['token'] for f in lexical_features[i+len(pivot_words):min(len(lexical_features), i+len(pivot_words)+window_size_right)]])

                    result = {
                        'contexte_gauche': left_context,
                        'mot_pivot': ' '.join([f['token'] for f in lexical_features[i:i+len(pivot_words)]]),
                        'contexte_droit': right_context
                    }
                    if doc_desc is not None:
                        doc_value = document['document'][doc_desc]
                        result[doc_desc] = doc_value

                    results.append(result)

    return pd.DataFrame(results)


def extract_ner(corpus_file, ner_tags, output="output_ner.csv"):
    """
    Extracts the named entitites detected in a corpus of documents made with lexploreur corpus() function.

    Parameters
    ----------
    corpus_file : str
        Path to the JSON file containing the corpus data.
    ner_tags : list or set
        List containing the NER labels to extract.
    output : str
        Name of the CSV file containing the NER data.

    Returns
    -------
    csv file
        A csv file containing the extracted named entitites.
    """

    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extracted = []

    for doc in data:
        doc_id = doc["document"]["id"]
        for feature in doc["document"]["lexical_features"]:
            if feature["pos"] in ner_tags:
                extracted.append({
                    "document_id": doc_id,
                    "type": feature["pos"],
                    "token": feature["token"]
                })

    df = pd.DataFrame(extracted)

    df.to_csv(output, index=False, encoding='utf-8')

    return (print(f"Extraction NERs terminée (fichier {output})."))
