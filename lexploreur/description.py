import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def features_frequencies(dtm):
    """
    Calculates the frequencies of the features in the given document-term matrix.

    Parameters
    ----------
    dtm : pandas.DataFrame
        The document-term matrix, where rows represent documents and columns represent features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the features and their corresponding frequencies, sorted in descending order by frequency.
    """

    total_occurrences = dtm.sum(axis=0)
    frequencies = pd.DataFrame(total_occurrences).reset_index()
    frequencies.columns = ['features', 'freq']
    frequencies.set_index('features', inplace=True)

    return frequencies.sort_values('freq', ascending=False)


def plot_pareto(feature_frequencies):
    """
    Plots the Pareto curve for the given feature frequencies.

    Parameters
    ----------
    feature_frequencies : pandas.DataFrame
        A pandas DataFrame containing the frequencies of the features, with the feature names as the index. Typicaly the object returned from `features_frequencies()`

    Returns
    -------
    None
        The function plots the Pareto curve and displays it.
    """

    plt.plot(feature_frequencies.index, feature_frequencies['freq'], marker='None')

    plt.xscale('log')
    plt.yscale('log')

    # Ajouter des labels et un titre
    plt.xlabel('Rang')
    plt.ylabel('Fréquence')
    plt.title('Courbe de Pareto')
    plt.show()


def corpus_size(dtm):
    """
    Calculates the total number of tokens and the total number of unique types (features) in the given document-term matrix.

    Parameters
    ----------
    dtm : pandas.DataFrame
        The document-term matrix, where rows represent documents and columns represent features.

    Returns
    -------
    None
        The function prints the total number of tokens and the total number of unique types in the corpus.
    """
    ntokens = dtm.sum().sum()
    ntypes = dtm.shape[1]

    return print(f"Total d'occurrences : {ntokens}\nTotal de formes : {ntypes}")


def group_sizes(dtm):
    """
    Calculates the number of features, unique types, and hapax for each doc from a dtm.

    Parameters
    ----------
    dtm : pandas.DataFrame
        The document-term matrix, where rows represent documents and columns represent features.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the number of features, unique types, and hapax legomena for each group in the corpus.
    """

    features = dtm.sum(axis=1)
    types = dtm.apply(lambda row: (row != 0).sum(), axis=1)
    hapax = dtm.apply(lambda row: (row == 1).sum(), axis=1)
    group_sizes = pd.DataFrame()
    group_sizes['features'] = features
    group_sizes['types'] = types
    group_sizes['hapax'] = hapax
    return group_sizes


def plot_group_sizes(groups, feature):
    """
    Plots a bar chart of the total count of the specified feature for each group in the corpus.

    Parameters
    ----------
    groups : pandas.DataFrame
        A DataFrame containing the group sizes, with the group names as the index.
    feature : str
        The name of the feature to plot, such as 'features', 'types', or 'hapax'.

    Returns
    -------
    None
        The function displays the bar chart.
    """

    groups[feature].plot(kind='bar')
    part = groups.index.name
    plt.title(f'Somme des {feature} par {part}')
    plt.xlabel(f'{part}')
    plt.ylabel(f'Nombre total de {feature}')
    plt.xticks(rotation="vertical")
    plt.show()


def plot_feature_distrib(corpus, dtm, features, part, plot_type):
    dtm[part] = corpus[part]
    features.append(str(part))
    selection = dtm[features]
    
    if plot_type == "frequency":
        count_df_grouped = selection.groupby(part).sum()
        count_df_grouped.plot(kind='bar')
        title_str = features[:-1]
        plt.title(f'{title_str}')
        plt.xlabel(f'{part}')
        plt.ylabel('Occurrences')
        plt.show()
    # elif plot_type == "cumulative":
    #     count_df_grouped = selection.sort_values(by=part).groupby(part)
    #     count_df_grouped.cumsum()
    #     # count_df_grouped = count_df_grouped.sort_values(by=part)
    #     print(count_df_grouped.head())
    #     count_df_grouped.plot(marker='None')
    #     title_str = features[:-1]
    #     plt.title(f'{title_str}')
    #     plt.xlabel(f'{part}')
    #     plt.ylabel('Occurrences')
    #     plt.show()
    elif plot_type == "density":
        count_df_grouped = selection.groupby(part).sum()
        total = dtm.groupby(part).sum()
        total = total.sum(axis=1)
        # percentages = count_df_grouped.div(total) * 100
        percentages = count_df_grouped.div(total, axis=0) * 1000
        percentages.plot(marker='None')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        title_str = features[:-1]
        plt.title(f'{title_str}')
        plt.xlabel(f'{part}')
        plt.ylabel('Occurrences')
        plt.show()
