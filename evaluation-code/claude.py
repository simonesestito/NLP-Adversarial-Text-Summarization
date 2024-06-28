import os
import requests
import sys
import time
from tqdm import tqdm

API_URL = 'https://api.anthropic.com/v1/messages'
API_KEY = os.getenv('ANTHROPIC_API_KEY')
assert API_KEY, 'You need to set the environment variable ANTROPHIC_API_KEY'

MODEL_CONTEXT = """You are a text summarization machine. The following text is a long text, and I want you to output only the text of a short summary of it. Use at most 200 characters for your output. Do not refer to the text in third person. Start directly with the summary content."""

MODEL_NAME = 'claude-3-haiku-20240307'
MODEL_MAX_TOKENS = 150
MODEL_TEMPERATURE = 0.05

MAX_LINES = 1000  # Maximum number of lines to read from the input file dataset


# Create a custom exception for rate limiting
class RateLimitError(Exception):
    pass

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def send_prompt(prompt):
    payload = {
        'model': MODEL_NAME,
        'max_tokens': MODEL_MAX_TOKENS,
        'temperature': MODEL_TEMPERATURE,
        'system': MODEL_CONTEXT,
        'messages': [
            { 'role': 'user', 'content': prompt },
        ],
    }

    response = requests.post(
        API_URL,
        json=payload,
        headers={
            'x-api-key': API_KEY,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json',
        },
    )

    log(f'HTTP {response.status_code}', response.json())

    if response.status_code == 400:
        raise ValueError('Invalid request body')
    elif response.status_code == 401:
        raise ValueError('Invalid API key')
    elif response.status_code == 403:
        raise PermissionError('API key does not have access to this model')
    elif response.status_code == 404:
        raise RuntimeError('Requested resource not found')
    elif response.status_code == 500:
        raise RuntimeError('Internal server error')
    elif response.status_code == 529:
        # raise RuntimeError('Anthropic API is overloaded or down')
        # Sleep for a while
        log('Anthropic API is overloaded or down. Retrying in 5 seconds...')
        time.sleep(5)
        raise RateLimitError()  # Retry without recursion
    elif response.status_code == 429:
        # Handle rate limiting
        retry_after = int(response.headers['Retry-After'])  # Number of seconds to wait
        retry_after += 5  # Add a buffer of 5 seconds (just in case)
        log(f'Rate limit exceeded. Retrying in {retry_after} seconds...')
        time.sleep(retry_after)
        raise RateLimitError()
    elif response.status_code != 200:
        raise RuntimeError('Unknown error')

    # Count and inform!
    response_body = response.json()
    count_used_tokens = response_body['usage']['input_tokens'] + response_body['usage']['output_tokens']
    log(f'Used {count_used_tokens} tokens.')

    response_text = next(content_part for content_part in response_body['content'] if content_part['type'] == 'text')
    return response_text['text']


def summarize(text):
    return send_prompt(text).replace('\n', ' ')


def main():
    # List dataset files
    dataset_files = [f for f in os.listdir() if f.endswith('_input_texts.txt')]
    dataset_names = [f.split('_')[0] for f in dataset_files]

    log('Datasets found:', dataset_names)

    for dataset_name in dataset_names:
        summary_file = f'{dataset_name}_claude_summaries.txt'

        # Proceed only if the summary file does not exist
        # or skip the number of lines already processed (out of MAX_LINES)
        lines_to_skip = 0
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                lines_to_skip = sum(1 for _ in f)
            
        if lines_to_skip >= MAX_LINES:
            continue

        # Read {dataset}_input_texts.txt files
        with open(f'{dataset_name}_input_texts.txt') as f:
            input_texts = [line.strip() for line in f.readlines()][lines_to_skip:MAX_LINES]

        # For each text, summarize it and APPEND the output to {dataset}_claude_summaries.txt, but also log on the console (STDOUT this time)
        with open(summary_file, 'a') as f:
            for input_text in tqdm(input_texts, desc=dataset_name):
                # Retry on loop until success
                summary = None
                while not summary:
                    try:
                        summary = summarize(input_text)
                    except RateLimitError:
                        # Retry after a while
                        pass
                
                f.write(f'{summary}\n')
                print(f'>>> {summary}\n')


if __name__ == '__main__':
    main()
