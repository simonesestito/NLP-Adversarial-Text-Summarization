import sys
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.bleu_metric import BleuMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.rouge_metric import RougeMetric

AVAILABLE_METRICS = ['bert_score', 'blanc', 'bleu', 'mover_score', 'rouge', 'all']


def pre_download_nltk():
    # Required by BlancMetric
    import nltk
    nltk.download('punkt')


def pre_download_models():
    pre_download_nltk()
    run_evaluation(
        input_texts=["a"],
        summaries=["a"],
        references=["a"],
        device='cpu',  # TODO: use a GPU
        metric='all',
    )  # Ignore the results


def run_evaluation(input_texts: list[str], summaries: list[str], references: list[str], device: str, metric: str) -> dict:
    output_dict = dict()

    if metric == 'bert_score' or metric == 'all':
        print("Evaluating BERT Score...")
        bert_score = BertScoreMetric()
        bert_score_dict = bert_score.evaluate_batch(summaries, references)
        output_dict['bert_score'] = bert_score_dict
    
    if metric == 'blanc' or metric == 'all':
        print("Evaluating Blanc...")
        blanc = BlancMetric(device=device, use_tune=(device != 'cpu'))
        blanc_dict = blanc.evaluate_batch(summaries, input_texts)
        output_dict.update(blanc_dict)
    
    if metric == 'bleu' or metric == 'all':
        print("Evaluating BLEU...")
        bleu = BleuMetric()
        bleu_dict = bleu.evaluate_batch(summaries, references)
        output_dict.update(bleu_dict)

    # FIXME: MoverScoreMetric is not working without a GPU
    # if metric == 'mover_score' or metric == 'all':
    #     print("Evaluating Mover Score...")
    #     mover_score = MoverScoreMetric()
    #     mover_score_dict = mover_score.evaluate_batch(summaries, references)
    #     output_dict['mover_score'] = mover_score_dict

    if metric == 'rouge' or metric == 'all':
        print("Evaluating ROUGE...")
        rouge = RougeMetric()
        rouge_dict = rouge.evaluate_batch(summaries, references)
        output_dict.update(rouge_dict)

    return output_dict


if __name__ == "__main__":
    print("Precaching models...")
    pre_download_models()
    print("Models precached.")
