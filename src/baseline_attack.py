import sys
import torch
import numpy as np
from copy import deepcopy

from .base_attack import BaselineAttack
from .TranslateAPI import translate

class Seq2SickAttack(BaselineAttack):
    def __init__(self, model, tokenizer, space_token, device, config, batch_size):
        super(Seq2SickAttack, self).__init__(model, tokenizer, space_token, device, config)

        # this batch_size will be used in the select_appearance_best function
        self.batch_size = batch_size

    @torch.no_grad()
    def select_apperance_best(self, new_strings, ori_trans: list, batch_size=10):
        seqs, scores = [], []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            input_token = self.tokenizer(new_strings[st:ed], return_tensors="pt", padding=True, truncation=True).input_ids
            input_token = input_token.to(self.device)
            trans_res = translate(
                self.model, input_token,
                early_stopping=False, num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, use_cache=True,
                max_length=self.max_len
            )
            new_trans_seqs = trans_res['sequences'].tolist()
            seqs.extend(new_trans_seqs)
            scores.extend([len(set(ori_trans) & set(s)) for s in new_trans_seqs])
        pred_len = np.array([self.compute_seq_len(torch.tensor(seq)) for seq in seqs])
        scores = np.array(scores)
        sel_index = scores.argmin()   #! FIXME: This is the only place where the score is minimized
        return [new_strings[sel_index]], scores[sel_index], pred_len[sel_index]

    def run_attack(self, text):
        assert len(text) == 1, 'Only support batch_size=1'
        ori_trans, ori_len = self.get_trans_string_len(text)     # int
        ori_trans = ori_trans.tolist()
        current_adv_text, _ = deepcopy(text), ori_len  # current_adv_text: List[str]
        # adv_his = [(deepcopy(current_adv_text[0]), current_len, 0.0)]
        pbar = range(1) # Enforce to run only once, it is enough in our experiments    tqdm(range((self.max_per)))
        modify_pos = []

        debug_log('Starting attack', is_first=True)

        for _ in pbar:
            debug_log('Computing loss')
            loss = self.compute_loss(text)
            debug_log('Doing backward')
            self.model.zero_grad()
            loss.backward()
            grad = self.embedding.grad
            debug_log('Finding token replace mutation')
            new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)

            debug_log('Selecting best appearance')
            current_adv_text, _, _ = self.select_apperance_best(new_strings, ori_trans, batch_size=self.batch_size)
            assert len(current_adv_text) == 1, 'current_adv_text should be a list of one string'
            current_adv_text = current_adv_text[0]

            debug_log('Found adversarial text!')

            with torch.no_grad():
                # self.model is T5ForConditionalGeneration
                debug_log('Tokenizing input')
                input_ids = self.tokenizer(text[0], return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
                debug_log('Generating original output')
                output = self.model.generate(input_ids=input_ids, max_length=100)

                original_text = text[0]
                original_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # print('ORI_TEXT:', original_text, '\n')
                # print('ORI_TEXT_OUTPUT:', original_output, '\n')

                debug_log('Tokenizing adv input')
                input_ids = self.tokenizer(current_adv_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
                debug_log('Generating adv output')
                output = self.model.generate(input_ids=input_ids, max_length=100)

                adversarial_text = current_adv_text.replace('</s>', '')
                adversarial_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # print('BEST_ADV_TEXT:', adversarial_text, '\n')
                # print('DECODED_ADV_OUTPUT:', adversarial_output)
                # print('\n-----\n', flush=True)

            debug_log('Returning from attack')
            return (original_text, original_output, adversarial_text, adversarial_output)

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.untarget_loss(scores, seqs, pred_len)
        return sum(loss_list)

    def token_replace_mutation(self, current_adv_text, grad, modify_pos):
        new_strings = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True, truncation=True).input_ids[0]
        base_tensor = current_tensor.clone()
        for pos in modify_pos:
            t = current_tensor[0][pos]
            grad_t = grad[t]
            score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
            index = score.argsort()
            for tgt_t in index:
                if tgt_t not in self.specical_token:
                    base_tensor[pos] = tgt_t
                    break
        score_list = []
        for pos, t in enumerate(current_tensor):
            if t not in self.specical_id:
                grad_t = grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        score_list.append(score[tgt_t])
                        new_base_tensor = base_tensor.clone()
                        new_base_tensor[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_tensor)
                        new_strings.append(candidate_s)
                        break
        return new_strings


debug_step_i = 0
def debug_log(message: str, is_first=False):
    global debug_step_i

    if is_first:
        debug_step_i = 0

    ANSI_YELLOW, ANSI_RESET = '\033[93m', '\033[0m'

    print(' ', ANSI_YELLOW, f'[{debug_step_i}] ', message, ANSI_RESET, ' '*10, flush=True, sep='', file=sys.stderr)
    debug_step_i += 1
