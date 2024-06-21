from datasets import load_dataset
from download_datasets import download_dataset

ds = load_dataset("wangwilliamyang/wikihow", 'all', split="test", trust_remote_code=True, data_dir=".").data

download_dataset(
    dataset=ds,
    article_key="text",
    highlights_key="headline",
    filename_prefix='wikihow',
)