"""
Standard PEGASUS model by Google
"""

from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("summarization", model="google/pegasus-xsum", device=0, truncation=True)
batch_size = 100

def load_dataset(dataset_name: str):
    filename = f'{dataset_name}_input_texts.txt'
    with open(filename, 'r') as f:
        # Read at most N lines at a time, yielding each batch
        while True:
          # Read N lines
          lines = [ f.readline().strip() for _ in range(batch_size) ]
          lines = [ line for line in lines if line != '' ]
          if len(lines) == 0:
              break

          yield lines


def summarize_dataset(dataset_name: str):
    with open(f'{dataset_name}_bart_summaries.txt', 'w') as f:
        for texts in tqdm(load_dataset(dataset_name), desc=dataset_name):
            summaries = pipe(
                texts, # It is a batch of documents
                max_length=100,  # Maximum length of the output text
                min_length=10,   # Minimum length of the output text
                return_text=True # Return the output text
            )

            assert len(texts) == len(summaries), 'Number of texts and summaries do not match'

            for text, summary in zip(texts, summaries):
                summary_text = summary['summary_text']
                
                if summary_text == '':
                    print('\n>>>>>>>>>>>>>>>>>>')
                    print('Empty summary for text:', text)
                    print('<<<<<<<<<<<<<<<<<<\n', flush=True)
                else:
                    assert '\n' not in summary_text, 'Summary contains newline characters'
                    f.write(summary_text)
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