from collections import Counter
import pathlib
import os

from nltk.tokenize.casual import TweetTokenizer
from joblib import Parallel, delayed


punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'


def process(path):
    # import data
    with open(path) as f:
        data = f.read()

    data = data.lower()

    # process entire string
    tokenizer = TweetTokenizer()
    string_processed = tokenizer.tokenize(data)

    # process individual tokens
    string_processed = [i.strip(punctuation) for i in string_processed]
    string_processed = [i for i in string_processed if i != ""]

    # get dictionary
    count = Counter(string_processed)
    return string_processed, count, str(path)


def filter_single(tokens, dictionary):
    return [i for i in tokens if (dictionary[i] > 10)]


def load_data_vectors(path):
    # First pass processing
    with Parallel(n_jobs=-1, verbose=5) as p:
        generator = (delayed(process)(i) for i in pathlib.Path(path).glob("*.txt"))
        data = p(generator)

        counts = sum((i[1] for i in data), Counter())
        names = [i[2] for i in data]

        # filter out words that appear only once in the entire corpus.
        gen = (delayed(filter_single)(i, counts) for i, d, n in data)
        data = p(gen)

    return {k: v for k, v in zip(names, data)}


if __name__ == "__main__":
    """This function finishes processing of data cleaning by creating a file mallet can read."""

    mallet_path = "/home/ryan/local/bin/Mallet/bin/mallet"

    output_path = "output/mallet"
    input_path = "data/weekly"
    os.makedirs(f"{output_path}/data", exist_ok=True)

    # Tokenization
    word_tokens = load_data_vectors(input_path)

    # Save as new text format
    for key, tokens in word_tokens.items():
        name = key.split("/")[-1]
        with open(f"{output_path}/data/{name}", "w+") as f:
            f.write(" ".join(tokens))