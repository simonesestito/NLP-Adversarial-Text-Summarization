import sys
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.bleu_metric import BleuMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.rouge_metric import RougeMetric


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
        device='cpu'  # TODO: use a GPU
    )  # Ignore the results


def run_evaluation(input_texts: list[str], summaries: list[str], references: list[str], device: str) -> dict:
    # Create the models
    bert_score = BertScoreMetric()
    blanc = BlancMetric(device=device, use_tune=(device != 'cpu'))
    bleu = BleuMetric()
    # mover_score = MoverScoreMetric()  FIXME: do with a GPU
    rouge = RougeMetric()

    # Evaluate the metrics
    print('Evaluating with BertScoreMetric...', file=sys.stderr)
    bert_score_dict = bert_score.evaluate_batch(summaries, references)

    print('Evaluating with BlancMetric...', file=sys.stderr)
    blanc_dict = blanc.evaluate_batch(summaries, input_texts)  #! BLANC works on input texts, not on references

    print('Evaluating with BleuMetric...', file=sys.stderr)
    bleu_dict = bleu.evaluate_batch(summaries, references)

    # print('Evaluating with MoverScoreMetric...')
    # mover_score_dict = mover_score.evaluate_batch(summaries, references)

    print('Evaluating with RougeMetric...', file=sys.stderr)
    rouge_dict = rouge.evaluate_batch(summaries, references)

    return {
        'bert_score': bert_score_dict,
        **blanc_dict,
        **bleu_dict,
        # 'mover_score': mover_score_dict,
        **rouge_dict,
    }


if __name__ == "__main__":
    print("Precaching models...")
    pre_download_models()
    print("Models precached.")
