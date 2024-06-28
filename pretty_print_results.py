import sys
import json
from typing import Union
from transformers import AutoTokenizer


ANSI_RESET = '\033[0m'
ANSI_GREEN = '\033[32m'
ANSI_RED = '\033[31m'
ANSI_BOLD = '\033[1m'

space_token = '▁'


def pretty_print_results(results: list[dict], tokenizer: Union[str, AutoTokenizer]):
    if type(tokenizer) == str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    for result in results:
        original_text = result['original_text']
        original_output = result['original_output']
        adversarial_text = result['adversarial_text']
        adversarial_output = result['adversarial_output']

        # Print token differences
        original_text_tokens = tokenizer.tokenize(original_text, truncation=True)
        adversarial_text_tokens = tokenizer.tokenize(adversarial_text, truncation=True)
        # assert len(original_text_tokens) == len(adversarial_text_tokens), 'Length of original text and output tokens should be the same'

        for original_token, adversarial_token in zip(original_text_tokens, adversarial_text_tokens):
            original_token, adversarial_token = original_token.replace(space_token, ' '), adversarial_token.replace(space_token, ' ')
            if original_token == adversarial_token:
                print(original_token, end='')
            else:
                print(ANSI_BOLD, ANSI_RED, original_token, ANSI_GREEN, adversarial_token, ANSI_RESET, end='', sep='')
        print()

        # Print different outputs
        print(ANSI_GREEN, original_output, ANSI_RESET)
        print(ANSI_RED, adversarial_output, ANSI_RESET)
        print('\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python pretty_print_results.py <tokenizer_name> <results_filename>')
        sys.exit(1)

    tokenizer_name, results_filename = sys.argv[1:]
    with open(results_filename, 'r') as f:
        results = json.load(f)
    pretty_print_results(results, tokenizer_name)
