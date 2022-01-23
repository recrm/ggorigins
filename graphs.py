import copy
import gzip
import itertools
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.plotting import table
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import jensenshannon


def graph_dendrogram(data, dates, title, output_file):
    """This function creates and saves the dendrogram chart to a file.

    Parameters:
        data : np.array
            matrix of shape (n_docs, n_topics) representing the document embeddings for each document.
        dates : list(datetime.datetime)
            Labels to use for each document (assumes datetime object)
        title : str
            Title of dataset
        output_file : str
            File to save dendrogram to.
    """
    # Generate distance matrix.
    distances = []
    for x, y in itertools.combinations(range(data.shape[0]), 2):
        distances.append(jensenshannon(data[x], data[y], 2))

    z = hierarchy.linkage(distances, 'average', optimal_ordering=True)
    labels = [f"{i.year}-{str(i.month).zfill(2)}-{str(i.day).zfill(2)}" for i in dates]

    # Create graph
    fig, ax = plt.subplots(figsize=(25,6))
    hierarchy.dendrogram(z, labels=labels, leaf_rotation=90, ax=ax)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_title(f"{title} tweet clustering")
    ax.set_xlabel("Week")
    ax.set_ylabel("Distance")
    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
    fig.savefig(output_file)


def graph_topics(data, dates, output_folder):
    """Graphs all topics as individual files.

    Parameters:
        data : np.array
            matrix of shape (n_docs, n_topics) representing the document embedings for each document.
        dates : list(datetime.datetime)
            Labels to use for each document (assumes datetime object)
        output_folder : str
            Folder to save topics to.
    """
    for i in range(data.shape[1]):
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        fig.tight_layout(h_pad=5)
        x, y = zip(*sorted(zip(dates, data[:, i])))
        axes.scatter(x, y)
        axes.plot(x, y)
        axes.set_ylim([0, 0.65])
        axes.tick_params(axis="x", rotation=45)
        axes.set_title(f"Topic {i}")
        axes.set_xlabel("Date")
        axes.set_ylabel("Percent Prevalence")
        fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)
        fig.savefig(f"{output_folder}/topic-{i}.png")


def graph_all_topics(data, dates, title, topics, output_file):
    """Graphs a number of topics on the same graph.

    Parameters:
        data : np.array
            matrix of shape (n_docs, n_topics) representing the document embedings for each document.
        dates : list(datetime.datetime)
            Labels to use for each document (assumes datetime object)
        title : str
            Name of group of topics
        topics : set
            Set of all of the topics to graph.
        output_file : str
            File to save graph to.
    """
    fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    fig.tight_layout(h_pad=7)
    for i in range(data.shape[1]):
        if i in topics:
            x, y = zip(*sorted(zip(dates, data[:, i])))
            axes.scatter(x, y)
            line = axes.plot(x, y)
            line[0].set_label(f"Topic {i}")

    axes.legend()
    axes.set_ylim([0, 0.60])
    axes.tick_params(axis="x", rotation=45)
    axes.set_title(f"Topics over time: {title}")
    axes.set_xlabel("Date")
    axes.set_ylabel("Percent Prevalence")
    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
    fig.savefig(output_file)


def topic_csv(input_folder, output_file):
    """Saves the document embeddings to file.

    Parameters:
        input_folder : str
            Folder containing mallet output
        output_file : str
            File to save csv to.
    """
    data = []

    topic_names = [f"Topic_{i}" for i in range(16)]

    with open(f"{input_folder}/doc_topics.txt") as f:
        for row in f:
            doc_id, date, *vector = row.split()

            date = date.split("/")[-1].split(".")[0]

            data.append({
                "doc_id": doc_id,
                "date": date,
                **{k: v for k, v in zip(topic_names, vector)}
            })

    df_data = pd.DataFrame(data)
    df_data.to_csv(output_file, index=False, float_format='%.15f')


def iterate_state(input_folder):
    """Iterate over lines in state
    This function allows us to efficiently iterate over the state file.
    Note: This file is huge and should not be read into memory all at once.
    """
    with gzip.open(f"{input_folder}/topic-state.gz", "r") as f:
        # The state file has a three line header that we are skipping.
        f.readline()
        f.readline()
        f.readline()

        for entry in f:
            yield entry.decode().split()


def get_word_frequencies(input_folder):
    """Gets the word frequencies for all topics over the entire vocabulary."""

    data = {}
    for i in range(16):
        data[i] = Counter()

    for doc, source, pos, typeindex, type_, topic in iterate_state(input_folder):
        data[int(topic)][type_] += 1

    return data


def examples(length, input_folder, output_folder):
    """Saves the document embeddings to file.

    Parameters:
        length : int
            Minimum example size to look for.
        input_folder : str
            Folder containing mallet output.
        output_folder : str
            Folder to save examples to.
    """
    data = {}

    current_doc = []
    current_index = []
    for doc, source, pos, typeindex, type_, topic in iterate_state(input_folder):
        if type_ == "newline":
            if len(set(current_index)) == 1:
                if len(current_doc) >= length:
                    if current_index[0] not in data:
                        data[current_index[0]] = []
                    data[current_index[0]].append(" ".join(current_doc))

            current_index = []
            current_doc = []
            continue

        current_doc.append(type_)
        current_index.append(topic)

    for key, items in data.items():
        with open(f"{output_folder}/topic-{key}.txt", "w+") as g:
            for line in items:
                print(line, file=g)


def graph_documents(data, output_folder, names):
    """Graph each document individually.

    Parameters:
        data : np.array
            matrix of shape (n_docs, n_topics) representing the document embedings for each document.
        output_folder : str
            Folder to save graphs to.
        names : list(str)
            Names of documents
    """
    for i in range(data.shape[0]):
        fig, axes = plt.subplots(1, 1, figsize=(10,5))
        fig.tight_layout(h_pad=5)
        axes.bar(range(16), data[i])
        axes.set_ylim([0,0.65])
        doc_name = names[i].split(".")[0]
        axes.set_title(f"Date-{doc_name}")
        axes.set_ylabel("Topic")
        axes.set_ylabel("Percent Prevalence")
        fig.subplots_adjust(bottom=0.1, top=0.90, left=0.1, right=0.95)
        fig.savefig(f"{output_folder}/document-{names[i]}.png")


def create_token_hist(data_raw, num_keys, output_folder, stopwords):
    """Graphs the keys of all topics.

    Parameters:
        data_raw : dict
            Dictionary containing all topics with another dictionary containing all words and counts.
        output_folder : str
            Folder to save graphs to.
        num_keys : int
            Number of keys to graph.
        stopwords : list(str)
            Tokens to remove from graph.
    """
    data = copy.copy(data_raw)

    for name, data_topic in data.items():
        keys = sorted(data_topic.keys(), key=lambda x: data_topic[x], reverse=True)
        keys = [i for i in keys if i not in stopwords]
        keys = [i for i in keys if "http://" not in i and "https://" not in i]
        keys = keys[:num_keys]

        total = sum(data_topic[i] for i in keys)

        fig, axes = plt.subplots(1, 1, figsize=(20,10))
        axes.bar(keys, [data_topic[i] / total for i in keys])
        axes.tick_params(axis="x", labelrotation=90)
        axes.set_title(f"Topic {name}")
        axes.set_ylabel("Keyword")
        axes.set_ylabel("Percent Prevalence")
        fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
        fig.savefig(f"{output_folder}/topic-{name}.png")


def keys_csv(words_raw, output_file, stopwords):
    """Creates a csv with all of the topic keys.

    Parameters:
        words_raw : dict
            Dictionary containing all topics with another dictionary containing all words and counts.
        output_file : str
            File to save csv to.
        stopwords : list(str)
            Tokens to remove from graph.
    """
    data = []
    for topic in words_raw:
        words = list(words_raw[topic].keys())
        words = [i for i in words if i not in stopwords]
        words = [i for i in words if "http://" not in i and "https://" not in i]

        keys = {w: words_raw[topic][w] for w in words}

        data.append({
            "#topic": topic,
            **keys
        })

    df_data = pd.DataFrame(data)
    df_data.to_csv(output_file, index=False)


def import_doc_topics(input_folder, date_format):
    """Imports data from mallet.

    Parameters:
        input_folder : str
            Path to Mallet folder.
        date_format : str
            Format string to interpret filenames.
    """
    # Import topics
    names = []
    topics_inner = []
    with open(f"{input_folder}/doc_topics.txt") as f:
        for row in f.readlines():
            index, name, *value = row.split()
            stack = np.array([float(i) for i in value])
            filtered = name.split("/")[-1]
            names.append(filtered)
            topics_inner.append(stack)

    dates_inner = [datetime.strptime(i, date_format) for i in names]
    topics_inner = np.row_stack(topics_inner)
    return dates_inner, topics_inner, names


def keys_by_order(words_raw, topics, stopwords, n_keys, output_file):
    fig, axes = plt.subplots(1, 1, figsize=(15, 2))

    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    data = {}
    for topic in topics:
        topic_dict = words_raw[topic]
        keys = sorted(topic_dict.keys(), key=lambda key: topic_dict[key] if key not in stopwords else 0, reverse=True)[:n_keys]
        data[topic] = keys

    df = pd.DataFrame(data)
    the_table = table(axes, df, loc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.subplots_adjust(top = 0.9, bottom = 0, right = 1, left = 0.01,
            hspace = 0, wspace = 0)
    fig.savefig(output_file)
