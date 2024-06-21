

def iter_dataset(ds, article_key: str, highlights_key: str):
    for batch in ds.to_batches():
        for i in range(batch.num_rows):
            row = batch.slice(i, 1).to_pydict()
            article, highlights = row[article_key], row[highlights_key]

            assert len(article) == 1, f'Expected one article, got {len(article)}'
            assert len(highlights) == 1, f'Expected one highlight, got {len(highlights)}'

            article, highlights = article[0], highlights[0]
            article = article.replace('\n', ' ').replace('\r', '')
            assert '\n' not in article, 'Unexpected newline in article'

            highlights = highlights.replace('\n', ' ').replace('\r', '')
            assert '\n' not in highlights, 'Unexpected newline in highlights'

            yield article, highlights


def download_dataset(dataset, article_key: str, highlights_key: str, filename_prefix: str):
    with open(f'{filename_prefix}_input_texts.txt', 'w') as input_texts, \
        open(f'{filename_prefix}_references.txt', 'w') as references:
        for article, highlights in iter_dataset(dataset, article_key, highlights_key):
            input_texts.write(article + '\n')
            references.write(highlights + '\n')

