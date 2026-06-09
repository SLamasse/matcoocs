# import spacy
# from tqdm.notebook import tqdm
# import pandas as pd
import spacy
from spacy.tokens import Doc
# from tqdm.notebook import tqdm
import pandas as pd
import json


def corpus(df, corpus_name, text_column, spacy_model, ner=None):
    """ Makes a corpus of texts using a spaCy model and generates a JSON file containing the lexical features of each document and its metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the documents to be processed.
    corpus_name : str
        Name of the JSON file to be generated.
    text_column : str
        Name of the column containing the text of the documents.
    spacy_model : str
        Name of the spaCy model to be used.
    ner : bool, optional
        If True, enables named entity recognition, otherwise disables named entity recognition.

    Returns
    -------
    None
        The function writes the JSON file containing the lexical features of each document.
    """

    # Chargement du modèle spaCy
    if ner is True:
        nlp = spacy.load(spacy_model, disable=["parser"])
    else:
        nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    # Initialisation de la liste des documents
    documents = []

    for index, row in df.iterrows():
        text = row[text_column]
        if pd.isna(text):
            print(f"⚠️ Le texte de la ligne {index}: est NaN.")
            continue
        else:
            print(f"Le texte de la ligne {index}. est en cours de traitement...")

        # Traitement du texte avec spaCy (découpage si > 100 000 caractères)
        if len(text) > 100000:
            split_text = [text[i:i+100000] for i in range(0, len(text), 100000)]
            docs = []
            for piece in split_text:
                piece = nlp(piece)
                docs.append(piece)
            doc = Doc.from_docs(docs)
        else:
            doc = nlp(text)

        document_data = {}

        # Copier toutes les colonnes du DataFrame sauf la colonne texte
        for col in df.columns:
            if col != text_column:
                document_data[col] = row[col]

        # Parcours des tokens du document
        lexical_features = []
        for token in doc:
            token_dict = {
                "token": token.text,
                "pos":   token.pos_,
                "lemma": token.lemma_
            }

            # Vérification si le token fait partie d'une entité nommée
            found_ent = False
            for ent in doc.ents:
                if token in ent:
                    token_dict["token"] = ent.text
                    token_dict["pos"]   = ent.label_
                    token_dict["lemma"] = ent.text.lower()
                    found_ent = True
                    break

            if not found_ent:
                lexical_features.append(token_dict)
            else:
                if (len(lexical_features) == 0
                        or lexical_features[-1]["token"] != token_dict["token"]):
                    lexical_features.append(token_dict)

        document_data['lexical_features'] = lexical_features
        documents.append({'document': document_data})

    # Écriture du fichier JSON
    with open(corpus_name, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)


def lexical_view(corpus_file, feature_to_extract='token',
                 lowercase=None, stopwords=None,
                 exclude_tokens=None, exclude_pos=None,
                 exclude_lemmas=None, group_by=None):
    """
    Extracts the specified lexical feature from a corpus of documents
    and returns a pandas DataFrame.

    Parameters
    ----------
    corpus_file : str
        Path to the JSON file containing the corpus data.
    feature_to_extract : str
        The lexical feature to extract: 'token', 'pos', or 'lemma'.
        Default is 'token'.
    lowercase : bool, optional
        If True, all features are lowercased before filtering and extraction.
        IMPORTANT : le filtre exclude_lemmas est appliqué APRÈS la mise en
        minuscules, garantissant que "France" est exclu si "france" figure
        dans la liste.
    stopwords : list, optional
        A list of stopwords to exclude from the extracted features.
    exclude_tokens : list, optional
        A list of tokens to exclude from the extracted features.
    exclude_pos : list, optional
        A list of part-of-speech tags to exclude from the extracted features.
    exclude_lemmas : list, optional
        A list of lemmas to exclude from the extracted features.
        La comparaison respecte le paramètre lowercase : si lowercase=True,
        les lemmes sont mis en minuscules avant comparaison.
    group_by : str, optional
        Name of the metadata to use to group corpus documents.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the extracted lexical features.
    """

    # Charger le fichier JSON
    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Construire un set pour la recherche en O(1)
    exclude_lemmas_set  = set(exclude_lemmas)  if exclude_lemmas  else set()
    exclude_tokens_set  = set(exclude_tokens)  if exclude_tokens  else set()
    stopwords_set       = set(stopwords)        if stopwords       else set()

    rows = []

    for doc in data:
        lexical_features = doc['document']['lexical_features']

        # ── Filtres indépendants de la casse ──────────────────────
        if exclude_pos:
            lexical_features = [
                f for f in lexical_features
                if f['pos'] not in exclude_pos
            ]

        # ── Normalisation lowercase AVANT les filtres textuels ────
        # On travaille sur une copie normalisée pour ne pas altérer
        # les données originales (utile si feature_to_extract='token').
        if lowercase:
            lexical_features = [
                {
                    "token": f["token"].lower(),
                    "pos":   f["pos"],
                    "lemma": f["lemma"].lower(),
                }
                for f in lexical_features
            ]

        # ── Filtres textuels (appliqués après normalisation) ──────
        if stopwords_set:
            lexical_features = [
                f for f in lexical_features
                if f[feature_to_extract] not in stopwords_set
            ]

        if exclude_tokens_set:
            lexical_features = [
                f for f in lexical_features
                if f[feature_to_extract] not in exclude_tokens_set
            ]

        if exclude_lemmas_set:
            lexical_features = [
                f for f in lexical_features
                if f['lemma'] not in exclude_lemmas_set
            ]

        # ── Extraction de la feature demandée ─────────────────────
        row = {
            feature_to_extract: [f[feature_to_extract] for f in lexical_features]
        }

        if group_by:
            row[group_by] = doc['document'][group_by]

        rows.append(row)

    df = pd.DataFrame(rows)

    if group_by:
        df = df.groupby(group_by)[feature_to_extract].sum().reset_index()

    return df


def lexical_features(corpus_file):
    """
    Retourne un DataFrame avec tous les tokens, POS et lemmes du corpus.
    """
    with open(corpus_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    tokens_data = []
    for doc in documents:
        document_data = doc['document']
        for token in document_data['lexical_features']:
            tokens_data.append({
                "token": token['token'],
                "pos":   token['pos'],
                "lemma": token['lemma'],
            })

    return pd.DataFrame(tokens_data)


def split_segments(df, colonne_texte, max_tokens=512):
    """
    Découpe les documents en segments de max_tokens tokens.
    """
    segments = []

    for _, row in df.iterrows():
        tokens = row[colonne_texte]
        for i in range(0, len(tokens), max_tokens):
            segment_tokens = tokens[i:i + max_tokens]
            segment_row = row.to_dict()
            segment_row[colonne_texte] = segment_tokens
            segments.append(segment_row)

    return pd.DataFrame(segments)
