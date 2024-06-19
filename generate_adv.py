import os
import sys
import torch
import argparse


from utils import *


if not os.path.isdir('adv'):
    os.mkdir('adv')

MAX_TESTING_NUM = 5


def main(task_id, attack_id, beam):
    # task_id = 0, attack_id = 0, beam = 1
    model_name = MODEL_NAME_LIST[task_id]
    # model_name = 'T5-small'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, tokenizer, space_token, dataset, src_lang, tgt_lang = load_model_dataset(model_name) # model = T5ForConditionalGeneration
    print('load model %s successful' % model_name, file=sys.stderr)
    beam = model.config.num_beams if beam is None else beam
    config = {
        'num_beams': beam,
        'num_beam_groups': model.config.num_beam_groups,
        'max_per': 3,
        'max_len': 100,
        'src': src_lang,
        'tgt': tgt_lang
    }
    attack_class = ATTACKLIST[attack_id]
    attack = attack_class(model, tokenizer, space_token, device, config)
    task_name = 'attack_type:' + str(attack_id) + '_' + 'model_type:' + str(task_id)

    results = []
    for i, src_text in enumerate(dataset):
        if i == 0:
            continue
        if i >= MAX_TESTING_NUM:
            break
        src_text = src_text.replace('\n', '')
        is_success, adv_his = attack.run_attack([src_text])
        if not is_success:
            print('error', file=sys.stderr)
            sys.exit(1)
        for tmp in adv_his:
            assert type(tmp[0]) == str
            assert type(tmp[1]) == int
            assert type(tmp[2]) == float

        if len(adv_his) != config['max_per'] + 1:
            delta = config['max_per'] + 1 - len(adv_his)
            for _ in range(delta):
                adv_his.append(adv_his[-1])

        assert len(adv_his) == config['max_per'] + 1
        results.append(adv_his)
        torch.save(results, 'adv/' + task_name + '_' + str(beam) + '.adv')
    torch.save(results, 'adv/' + task_name + '_' + str(beam) + '.adv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--data', default=0, type=int, help='experiment subjects')
    parser.add_argument('--attack', default=0, type=int, help='attack type')
    parser.add_argument('--beam', default=1, type=int, help='beam size')
    args = parser.parse_args()
    main(args.data, args.attack, args.beam)
    exit(0)


