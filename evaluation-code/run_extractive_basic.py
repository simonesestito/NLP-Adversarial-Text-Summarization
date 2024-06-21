# From https://colab.research.google.com/drive/1kqu4VEbbQuRz0sIu5qt8DI2n_3YP-TPp

from collections import defaultdict
import itertools
from multiprocessing import Process
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')


def create_frequency_table(words_list: list[str]) -> dict[str, int]:
    frequencies = defaultdict(int)
    for word in words_list:
        # May be parallelizable in multiple threads and merge the results
        frequencies[word] += 1
    return frequencies


def create_word_frequencies(words_frequencies: dict[str, int]) -> dict[str, float]:
    most_occurred_word_frequency = max(words_frequencies.values())
    # print('most_occurred_word_frequency:', most_occurred_word_frequency)

    return {
        word: word_frequency / most_occurred_word_frequency
        for word, word_frequency in words_frequencies.items()
    }


def compute_sentence_score(sentence: str, relative_word_frequencies: dict[str, float]) -> float:
    # Split the sentence into words
    words = word_tokenize(sentence)
    return sum(relative_word_frequencies[word] for word in words)


def compute_sentences_score(sentences: list[str], relative_word_frequencies: dict[str, float]) -> dict[str, float]:
    return {
        sentence: compute_sentence_score(sentence, relative_word_frequencies)
        for sentence in sentences
    }


def compute_threshold(sentences_score: dict[str, float], scaling_factor: float = 1.0):
    average = sum(score for score in sentences_score.values()) / \
        len(sentences_score)

    # In this paper, they use the average as it is. Others use average*1.3 or similar
    # Make sure to always have at least 1 sentence
    return min(average * scaling_factor, max(sentences_score.values()))


def summarize_corpus(corpus: str) -> list[str]:
    sentences = sent_tokenize(corpus)
    all_words = itertools.chain.from_iterable(
        word_tokenize(sentence) for sentence in sentences)

    frequency_table = create_frequency_table(all_words)
    if not frequency_table:
        print('>>>>>>>>>>>>>>>>>>')
        print('Empty frequency table')
        print('Corpus:', f'"{corpus}"')
        print('Sentences:', sentences)
        print('<<<<<<<<<<<<<<<<<<')
        return []
    word_frequencies = create_word_frequencies(frequency_table)
    sentences_score = compute_sentences_score(sentences, word_frequencies)
    threshold = compute_threshold(sentences_score)

    return [
        sentence
        for sentence, score in sentences_score.items()
        if score >= threshold
    ]


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

    with open(f'{dataset_name}_extractivebasic_summaries.txt', 'w') as f:
        for i, text in enumerate(dataset):
            summary = ' '.join(summarize_corpus(text))
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
    datasets = ['cnndailymail', 'wikihow', 'xsum']

    print('Summarizing datasets:', datasets)
    processes = [
        Process(target=summarize_dataset, args=(dataset,))
        for dataset in datasets
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
