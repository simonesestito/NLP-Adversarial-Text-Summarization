"""
Implementation adapted from: 
"""

import editdistance
import itertools
import networkx as nx
import nltk
import os
from multiprocessing import Process


def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print('Completed resource downloads.')


def unique_everseen(iterable, key=None):
    """List unique elements in order of appearance.

    Examples:
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        def key(x): return x
    for element in iterable:
        k = key(element)
        if k not in seen:
            seen_add(k)
            yield element


def build_graph(nodes):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = editdistance.eval(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr


def extract_sentences(text, summary_length=100, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text.

    :param text: A string.
    """
    sent_detector = nltk.data.load('tokenizers/punkt/'+language+'.pickle')
    sentence_tokens = sent_detector.tokenize(text.strip())
    graph = build_graph(sentence_tokens)

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    # most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
                       reverse=True)

    # return a 100 word summary
    summary = ' '.join(sentences)
    summary_words = summary.split()
    summary_words = summary_words[0:summary_length]
    dot_indices = [idx for idx, word in enumerate(summary_words) if word.find('.') != -1]
    if clean_sentences and dot_indices:
        last_dot = max(dot_indices) + 1
        summary = ' '.join(summary_words[0:last_dot])
    else:
        summary = ' '.join(summary_words)

    return summary


def load_dataset(dataset_name: str):
    filename = f'{dataset_name}_input_texts.txt'
    with open(filename, 'r') as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() != ''
        ]


def summarize_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name)

    with open(f'{dataset_name}_textrank_summaries.txt', 'w') as f:
        for i, text in enumerate(dataset):
            summary = extract_sentences(text)
            if summary == '':
                print('\n>>>>>>>>>>>>>>>>>>')
                print('Empty summary at index:', i)
                print('Text:', f'"{text}"')
                print('<<<<<<<<<<<<<<<<<<\n', flush=True)
                continue

            assert '\n' not in summary, 'Summary contains newline characters'
            f.write(summary)
            f.write('\n')

    print(f'{dataset_name} done', flush=True)


if __name__ == '__main__':
    setup_environment()

    # List all datasets = all files in current directory with name *_input_texts.txt
    datasets = [f.replace('_input_texts.txt', '') for f in os.listdir('.') if f.endswith('_input_texts.txt')]
    print('Running on datasets:', datasets)

    # Run in parallel
    processes = []
    for dataset_name in datasets:
        p = Process(target=summarize_dataset, args=(dataset_name,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
