from datasets import load_dataset
from download_datasets import download_dataset

ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").data
download_dataset(
    dataset=ds,
    article_key="article",
    highlights_key="highlights",
    filename_prefix='cnndailymail',
)
