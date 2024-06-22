"""
From: https://github.com/dmmiller612/bert-extractive-summarizer
"""

from summarizer import Summarizer
from tqdm import tqdm

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
    model = Summarizer('distilbert-base-uncased')

    with open(f'{dataset_name}_bertextractive_summaries.txt', 'w') as f:
        for i, text in enumerate(tqdm(dataset, desc=dataset_name)):
            summary = model(text)
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

def main():
    datasets = ['cnndailymail', 'wikihow', 'xsum']

    print('Summarizing datasets:', datasets)

    for dataset in datasets:
        print('Summarizing', dataset, '...', end=' ', flush=True)
        summarize_dataset(dataset)
        print('done', flush=True)

if __name__ == '__main__':
    main()