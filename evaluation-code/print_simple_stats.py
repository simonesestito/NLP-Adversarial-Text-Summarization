"""
Print the average and std. deviation of the reduction in the summary length
"""

import os
import json

MAX_LINES = 2000

summaries_files = [f for f in os.listdir() if f.endswith('_summaries.txt')]

results_dict = dict()

for summary_file in summaries_files:
    with open(summary_file) as f:
        summaries = [line.strip() for line in f.readlines()][:MAX_LINES]

    dataset_name = summary_file.split('_')[0]
    with open(f"{dataset_name}_input_texts.txt") as f:
        input_texts = [line.strip() for line in f.readlines()][:MAX_LINES]

    model_name = summary_file.split('_')[1]
    with open(f"{dataset_name}_references.txt") as f:
        references = [line.strip() for line in f.readlines()][:MAX_LINES]

    # With respect to the input text
    input_perc_reduction = [ (len(input_text) - len(summary)) / len(input_text) for input_text, summary in zip(input_texts, summaries)]
    input_avg_reduction = sum(input_perc_reduction) / len(input_perc_reduction)
    input_std_reduction = (sum([(perc - input_avg_reduction) ** 2 for perc in input_perc_reduction]) / len(input_perc_reduction)) ** 0.5

    # With respect to the reference
    refs_perc_reduction = [ (len(ref) - len(summary)) / len(ref) for ref, summary in zip(references, summaries)]
    refs_avg_reduction = sum(refs_perc_reduction) / len(refs_perc_reduction)
    refs_std_reduction = (sum([(perc - refs_avg_reduction) ** 2 for perc in refs_perc_reduction]) / len(refs_perc_reduction)) ** 0.5

    if model_name not in results_dict:
        results_dict[model_name] = dict()

    assert dataset_name not in results_dict[model_name]

    results_dict[model_name][dataset_name] = {
        'input_avg_reduction': input_avg_reduction,
        'input_std_reduction': input_std_reduction,
        'refs_avg_reduction': refs_avg_reduction,
        'refs_std_reduction': refs_std_reduction,
    }

print(json.dumps(results_dict, indent=2))
