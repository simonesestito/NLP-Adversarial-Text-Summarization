#!/bin/bash --
set -euo pipefail

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
        # Ask if the metrics already exist and it is not an empty file
        if [ -s "${dataset}_${model}_metrics.txt" ]; then
            echo "Metrics already exist for $dataset $model. Do you want to overwrite them? [y/N]"
            read -r answer
            if [ "$answer" != "y" ]; then
                echo "Skipping $dataset $model..."
                continue
            fi
        fi

        # Run the evaluation
        echo "Evaluating $dataset $model..."
        docker run summeval-evaluation:latest \
            --input-texts ${dataset}_input_texts.txt \
            --references ${dataset}_references.txt \
            --summaries ${dataset}_${model}_summaries.txt \
            > ${dataset}_${model}_metrics.txt
        echo "Results saved to ${dataset}_${model}_metrics.txt"
        echo
    done
done
