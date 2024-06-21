# Evaluation metrics

This can be considered a subproject of the original one.

This contains the code that is used in the section **6. Comparative Evaluation** of our paper.

It aims to compare multiple text summarization models and see how good are they with other scores and datasets.
This is the code to effectively use SummEval for our goal.

This is *NOT* part of Adversarial Attacks, but still in the same survey / project.

## How to use
1. Download the dataset, making sure that the file has one sample per line, as in the `download_*.py` files
2. Run the model, as in `run_*.py` files
3. Build the Docker image from [SummEval](https://github.com/simonesestito/SummEval/), naming it `summeval:latest`
4. Build the Docker image in this folder, calling it `summeval-evaluation:latest`
5. Run the latter, passing `--input-texts`, `--summaries` and `--references` arguments pointing to the correct files. If some files are missing in the Docker image, check their naming and also if you want to explicitly include them editing the `Dockerfile` here.
6. Save the output in a file of your choice
