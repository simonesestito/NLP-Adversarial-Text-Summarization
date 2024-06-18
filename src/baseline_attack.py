import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .base_attack import BaselineAttack
from .TranslateAPI import translate


class Seq2SickAttack(BaselineAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(Seq2SickAttack, self).__init__(model, tokenizer, space_token, device, config)

    @torch.no_grad()
    def select_apperance_best(self, new_strings, ori_trans: list, batch_size=100):
        seqs, scores = [], []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1
        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            input_token = self.tokenizer(new_strings[st:ed], return_tensors="pt", padding=True).input_ids
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
        sel_index = scores.argmax()   #! FIXME: This is the only place where the score is minimized
        return [new_strings[sel_index]], scores[sel_index], pred_len[sel_index]

    def run_attack(self, text):
        assert len(text) == 1
        ori_trans, ori_len = self.get_trans_string_len(text)     # int
        ori_trans = ori_trans.tolist()
        best_adv_text, best_len, best_score = text.copy()[0], ori_len, len(ori_trans)
        current_adv_text, current_len = deepcopy(text), ori_len  # current_adv_text: List[str]
        adv_his = [(deepcopy(current_adv_text[0]), current_len, 0.0)]
        pbar = tqdm(range(self.max_per))
        modify_pos = []

        t1 = time.time()
        for it in pbar:
            loss = self.compute_loss(text)
            self.model.zero_grad()
            loss.backward()
            grad = self.embedding.grad
            new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)

            current_adv_text, current_score, current_len = self.select_apperance_best(new_strings, ori_trans)

            log_str = "%d, %d, %d, %.2f" % (it, len(new_strings), best_score, best_len / ori_len)
            pbar.set_description(log_str)
            if current_score < best_score:
                best_adv_text = current_adv_text[0]
                best_score = current_score

            with torch.no_grad():
                # self.model is T5ForConditionalGeneration
                input_ids = self.tokenizer(text[0], return_tensors="pt", padding=True).input_ids.to(self.device)
                output = self.model.generate(input_ids=input_ids, max_length=100)
                print('ORI_TEXT:', text[0], '\n')
                print('ORI_TEXT_OUTPUT:', self.tokenizer.decode(output[0], skip_special_tokens=True), '\n')
                
                input_ids = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids.to(self.device)
                output = self.model.generate(input_ids=input_ids, max_length=100)
                print('BEST_ADV_TEXT:', current_adv_text, '\n')
                print('DECODED_ADV_OUTPUT:', self.tokenizer.decode(output[0], skip_special_tokens=True))
                print('\n-----\n', flush=True)

            t2 = time.time()
            adv_his.append((best_adv_text, best_len, t2 - t1))
        return True, adv_his

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        loss_list = self.untarget_loss(scores, seqs, pred_len)
        return sum(loss_list)

    def token_replace_mutation(self, current_adv_text, grad, modify_pos):
        new_strings = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
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