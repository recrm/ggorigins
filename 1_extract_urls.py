import pathlib
import pandas as pd
import json

from nltk.tokenize.casual import TweetTokenizer
from collections import Counter
from nltk.tokenize.treebank import TreebankWordDetokenizer

if __name__ == "__main__":
    """This script will extract all twitter links out of the raw dataset."""
    tokenizer = TweetTokenizer()
    untokenizer = TreebankWordDetokenizer()

    urls = Counter()
    for i in pathlib.Path(f"raw_data/weekly_raw").glob("*.csv"):
        df = pd.read_csv(i)

        # Remove retweets
        dfFiltered = df[pd.isnull(df["Is a Retweet of"])]

        for tweet in dfFiltered["Tweet Text"]:
            tokenized = tokenizer.tokenize(tweet)

            # Some retweets aren't marked as retweets sadly.
            if tokenized[0] == "RT":
                continue

            for index, word in enumerate(tokenized):
                if "://t.co/" in word:
                    urls[word] += 1

    with open(f"output/all_urls.json", "w+") as f:
        json.dump(list(urls.keys()), f)
