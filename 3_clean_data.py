import pathlib
import pandas as pd
import os
import json
import html

# from utilities import name_generator
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

with open("output/url_dictionary.json") as f:
    url_data = json.load(f)


def process(tweet):
    tokenizer = TweetTokenizer()
    untokenizer = TreebankWordDetokenizer()

    # remove unmarked retweets
    if tweet[:2] == "RT":
        return None

    # Unescape html encoded sequences (can be nested up to 4 times).
    for _ in range(4):
        tweet = html.unescape(tweet)

    # Replace newlines with token.
    tweet = tweet.replace("\n", " _newline_ ")

    # Replace twitter links with tokens.
    # Also removes gamingjoboblin spam.
    tokenized = tokenizer.tokenize(tweet)
    for index, word in enumerate(tokenized):
        if "://t.co/" in word:
            clean = url_data[word]
            if "gamingjobonlin.com" in clean:
                return None

            tokenized[index] = "twitter_link"

        # elif word[0] == "@":
        #     tokenized[index] = name_generator(word)

    return untokenizer.detokenize(tokenized)


if __name__ == "__main__":
    """This script cleans the raw data into processable data files. It does a number of things.
    1) Detect and removes retweets.
    2) Detect and removes "gamingjobonlin.com" spam.
    3) Replace all twitter link with the "twitter_link" token.
    4) Replace all newlines (\n) with the "_newline_" token.
    5) Remove html escaped characters.
    """
    output_folder = "data/weekly"
    os.makedirs(output_folder, exist_ok=True)

    for i in pathlib.Path(f"raw_data/weekly_raw").glob("*.csv"):
        print(i)
        df = pd.read_csv(i)
        dfFiltered = df[pd.isnull(df["Is a Retweet of"])]
        name = i.parts[-1].split(".")[0]

        output = [process(i) for i in dfFiltered["Tweet Text"]]
        with open(f"{output_folder}/{name}.txt", "w+") as f:
            for t in output:
                if t is not None:
                    print(t, file=f)
