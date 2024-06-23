#!/bin/bash --
set -euo pipefail

# Metric name as first argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <metric_name>"
    exit 1
fi
METRIC_NAME="$1"

# List all files in this directory ending by _input_texts.txt (the input texts)
INPUT_TEXTS=$(ls *_input_texts.txt)

# Extract the dataset names from them
DATASETS=$(echo $INPUT_TEXTS | sed 's/_input_texts.txt//g')
echo "Datasets found: $DATASETS"
echo

# For every dataset, run the evaluation
for dataset in $DATASETS; do
    # Check if all other files are present (_references.txt, _summaries.txt)
    if [ ! -f "${dataset}_references.txt" ]; then
        echo "Skipping $dataset: ${dataset}_references.txt not found"
        continue
    fi

    # Extract the model names
    ALL_SUMMARIES=$(ls ${dataset}_*_summaries.txt)
    MODELS=$(echo $ALL_SUMMARIES | sed "s/${dataset}_//g" | sed 's/_summaries.txt//g')
    echo "Models found for $dataset: $MODELS"
    echo

    # For every model, run the evaluation
    for model in $MODELS; do
        OUTPUT_FILE="${dataset}_${model}_metrics_${METRIC_NAME}.json"

        # Ask if the metrics already exist and it is not an empty file
        if [ -s "$OUTPUT_FILE" ]; then
            echo "Metric $METRIC_NAME already exist for $dataset $model. Do you want to overwrite them? [y/N]"
            read -r answer
            if [ "$answer" != "y" ]; then
                echo "Skipping $dataset $model..."
                continue
            fi
        fi

        # Run the evaluation
        echo "Evaluating $dataset $model on metric $METRIC_NAME..."
        docker run summeval-evaluation:latest \
            --input-texts ${dataset}_input_texts.txt \
            --references ${dataset}_references.txt \
            --summaries ${dataset}_${model}_summaries.txt \
            --metric "$METRIC_NAME" \
            > "$OUTPUT_FILE"
        echo "Results saved to $OUTPUT_FILE"
        echo
    done
done
