from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import os


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def summarize_batch(batch, tokenizer, model, device):
    inputs = tokenizer(
        batch, max_length=1024, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    summary_ids = model.generate(**inputs, early_stopping=True)
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summaries


def process_file(input_file, output_file, model_name, batch_size=8):
    tokenizer, model, device = load_model(model_name)

    with open(input_file, "r", encoding="utf-8") as infile:
        total_paragraphs = sum(1 for line in infile if line.strip())

    last_processed = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            last_processed = sum(1 for _ in outfile)

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "a", encoding="utf-8"
    ) as outfile:

        for _ in range(last_processed):
            next(infile)

        pbar = tqdm(
            total=total_paragraphs,
            initial=last_processed,
            desc="Summarizing paragraphs",
            unit="paragraph",
        )

        batch = []
        try:
            for line in infile:
                paragraph = line.strip()
                if paragraph:
                    batch.append(paragraph)

                    if len(batch) == batch_size:
                        summaries = summarize_batch(batch, tokenizer, model, device)
                        for summary in summaries:
                            outfile.write(summary + "\n")
                        outfile.flush()
                        pbar.update(len(batch))
                        batch = []

            if batch:
                summaries = summarize_batch(batch, tokenizer, model, device)
                for summary in summaries:
                    outfile.write(summary + "\n")
                outfile.flush()
                pbar.update(len(batch))

        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved. You can resume later.")
        finally:
            pbar.close()


if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    model_name = "google/pegasus-large"
    batch_size = 6
    process_file(input_file, output_file, model_name, batch_size)
    print(f"Summarization complete. Results saved to {output_file}")
