"""
Print the average and std. deviation of the reduction in the summary length
"""

import os
import json

MAX_LINES = 2000

summaries_files = [f for f in os.listdir() if f.endswith('_summaries.txt')]
dataset_names = ['cnndailymail', 'xsum', 'wikihow']
model_names = ['bart', 'bertextractive', 'claude', 'extractivebasic', 'pegasusstd', 'pegasuslarge', 'textrank']

def load_summary_file(summary_file):
    with open(summary_file) as f:
        summaries = [line.strip() for line in f.readlines()][:MAX_LINES]
        # Replace empty line with a dot
        summaries = [summary if summary else '.' for summary in summaries]

    dataset_name = summary_file.split('_')[0]
    with open(f"{dataset_name}_input_texts.txt") as f:
        input_texts = [line.strip() for line in f.readlines()][:MAX_LINES]

    with open(f"{dataset_name}_references.txt") as f:
        references = [line.strip() for line in f.readlines()][:MAX_LINES]

    model_name = summary_file.split('_')[1]

    return input_texts, summaries, references, model_name, dataset_name


def compute_stats(input_texts, summaries, references):
    # With respect to the input text
    input_perc_reduction = [ (len(input_text) - len(summary)) / len(input_text) for input_text, summary in zip(input_texts, summaries)]
    input_avg_reduction = sum(input_perc_reduction) / len(input_perc_reduction)
    input_std_reduction = (sum([(perc - input_avg_reduction) ** 2 for perc in input_perc_reduction]) / len(input_perc_reduction)) ** 0.5

    # With respect to the reference
    refs_perc_reduction = [ (len(ref) - len(summary)) / len(ref) for ref, summary in zip(references, summaries)]
    refs_avg_reduction = sum(refs_perc_reduction) / len(refs_perc_reduction)
    refs_std_reduction = (sum([(perc - refs_avg_reduction) ** 2 for perc in refs_perc_reduction]) / len(refs_perc_reduction)) ** 0.5

    return {
        'input_avg_reduction': input_avg_reduction,
        'input_std_reduction': input_std_reduction,
        'refs_avg_reduction': refs_avg_reduction,
        'refs_std_reduction': refs_std_reduction,
    }


results_dict = dict()

for summary_file in summaries_files:
    input_texts, summaries, references, model_name, dataset_name = load_summary_file(summary_file)

    if model_name not in model_names:
        print(f"Skipping {model_name} as it is not in the list of models")
        continue

    # print(f"Computing stats for {model_name} on {dataset_name}")

    if model_name not in results_dict:
        results_dict[model_name] = dict()

    assert dataset_name not in results_dict[model_name]

    results_dict[model_name][dataset_name] = compute_stats(input_texts, summaries, references)


# Compute over all the datasets
OVERALL_STATS_KEY = 'overall'
summaries, input_texts, references = {model_name: [] for model_name in model_names}, {model_name: [] for model_name in model_names}, {model_name: [] for model_name in model_names}
for summary_file in summaries_files:
    input_text, summary, reference, model_name, _ = load_summary_file(summary_file)
    if model_name in model_names:
        summaries[model_name].extend(summary)
        input_texts[model_name].extend(input_text)
        references[model_name].extend(reference)

for model_name in model_names:
    results_dict[model_name][OVERALL_STATS_KEY] = compute_stats(input_texts[model_name], summaries[model_name], references[model_name])

# Save the results to a json file
with open('summaries_stats.json', 'w') as f:
    json.dump(results_dict, f, indent=4)


def latex_print_stats(stats: dict):
    avg, std = stats['input_avg_reduction'], stats['input_std_reduction']
    # avg, std = stats['refs_avg_reduction'], stats['refs_std_reduction']
    avg, std = f"{avg*100:.1f}\\%", f"{std*100:.1f}\\%"
    print(f" & {avg} ± {std}", end='')

# Print as table
for model_name in model_names:
    print(f'{model_name:<15}:', end='')
    for dataset_name in dataset_names:
        latex_print_stats(results_dict[model_name][dataset_name])
    latex_print_stats(results_dict[model_name][OVERALL_STATS_KEY])
    print(' \\\\')