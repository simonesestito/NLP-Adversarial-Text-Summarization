# This image must be built locally before this Dockerfile can be used
# Use the instructions on the README.md of https://github.com/simonesestito/SummEval
FROM summeval:latest

# Pre-cache the models
COPY evaluator.py .
RUN python evaluator.py

# Copy all other files
COPY *.py .
COPY *_input_texts.txt .
COPY *_summaries.txt .
COPY *_references.txt .

ENTRYPOINT ["python", "run_evaluation.py"]