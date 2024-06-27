import torch
import argparse
import datetime
import json

from utils import *
from pretty_print_results import pretty_print_results


MAX_TESTING_NUM = 5

ANSI_RED_BOLD = '\033[1;31m'
ANSI_RESET = '\033[0m'


def main(task_id, attack_id, beam, resume_from_index=0, batch_size=10):
    # task_id = 0, attack_id = 0, beam = 1
    model_name = MODEL_NAME_LIST[task_id]
    # model_name = 'T5-small'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    model, tokenizer, space_token, dataset, src_lang, tgt_lang = load_model_dataset(model_name)
    print('load model %s successful' % model_name)

    beam = model.config.num_beams if beam is None else beam
    if beam > 1:
        print(ANSI_RED_BOLD, '[!] Using beam dimension:', beam, ANSI_RESET, flush=True)

    config = {
        'num_beams': beam,
        'num_beam_groups': model.config.num_beam_groups,
        'max_per': 3,
        'max_len': 100,
        'src': src_lang,
        'tgt': tgt_lang
    }
    attack_class = ATTACKLIST[attack_id]

    # Seq2SickAttack supports batch_size in the select_apperance_best function
    if attack_class == Seq2SickAttack:
        attack = attack_class(model, tokenizer, space_token, device, config, batch_size)
    else:
        attack = attack_class(model, tokenizer, space_token, device, config)

    results = []
    try:
        for i, src_text in enumerate(dataset[resume_from_index:]):
            if i >= MAX_TESTING_NUM:
                break
            src_text = src_text.replace('\n', '')
            original_text, original_output, adversarial_text, adversarial_output = attack.run_attack([src_text])
            result_dict = {
                'original_text': original_text,
                'original_output': original_output,
                'adversarial_text': adversarial_text,
                'adversarial_output': adversarial_output
            }
            results.append(result_dict)

            # Also, log the result to console
            pretty_print_results([result_dict], tokenizer)
    except KeyboardInterrupt:
        print(ANSI_RED_BOLD, 'Caught KeyboardInterrupt! Saving results so far...', ANSI_RESET, flush=True, sep='')
    finally:
        # Save result to JSON file
        current_timestamp = get_current_timestamp()
        result_filename = 'adversarial_result__%s.json' % current_timestamp
        with open(result_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print('Save result to %s' % result_filename)


def get_current_timestamp() -> str:
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--task-id', default=0, type=int, help='task id')
    parser.add_argument('--beam', default=2, type=int, help='beam size')
    parser.add_argument('--resume-from-index', default=0, type=int, help='Index of the dataset sample to resume from')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size for Seq2SickAttack')
    args = parser.parse_args()
    main(args.task_id, 0, args.beam, args.resume_from_index, args.batch_size)
    exit(0)
