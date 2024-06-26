import os
import requests
import sys
import time

API_URL = 'https://api.anthropic.com/v1/messages'
API_KEY = os.getenv('ANTROPHIC_API_KEY')
assert API_KEY, 'You need to set the environment variable ANTROPHIC_API_KEY'

MODEL_CONTEXT = """You are a text summarization machine. The following text is a long text, and I want you to output only the text of a short summary of it. Use at most 200 characters for your output."""
PROMPT_PREFIX = "Here it is the long text to summarize:"

MODEL_NAME = 'claude-3-haiku-20240307'
MODEL_MAX_TOKENS = 150
MODEL_TEMPERATURE = 0.05

MAX_LINES = 2000  # Maximum number of lines to read from the input file dataset


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


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

    log(response.json())

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
        raise RuntimeError('Anthropic API is overloaded or down')
    elif response.status_code != 200:
        raise RuntimeError('Unknown error')
    
    # Handle rate limiting
    if response.status_code == 429:
        log('Rate limit exceeded')
        retry_after = int(response.headers['Retry-After'])  # Number of seconds to wait
        log(f'Retrying in {retry_after} seconds...')
        time.sleep(retry_after)
        return send_prompt(prompt)  # FIXME: do not use recursion

    # Count and inform!
    count_used_tokens = response['usage']['input_tokens'] + response['usage']['output_tokens']
    log(f'Used {count_used_tokens} tokens.')

    return response['content']['text']


def summarize(text):
    prompt = f"{PROMPT_PREFIX}\n\n{text}"
    return send_prompt(prompt)


def main():  # TODO
    # List dataset files

    # Proceed only if the summary file does not exist
    # or skip the number of lines already processed (out of MAX_LINES)

    # Read {dataset}_input_texts.txt files

    # For each text, summarize it and APPEND the output to {dataset}_claude_summaries.txt, but also log on the console (STDOUT this time)

    pass


if __name__ == '__main__':
    main()
