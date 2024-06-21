from datasets import load_dataset
from download_datasets import download_dataset

ds = load_dataset("EdinburghNLP/xsum", split="test").data
download_dataset(
    dataset=ds,
    article_key="document",
    highlights_key="summary",
    filename_prefix='xsum',
)

