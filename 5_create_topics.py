import os


if __name__ == "__main__":
    """This actually generates the mallet trained model."""

    mallet_path = "/home/ryan/local/bin/Mallet/bin/mallet"

    output_path = "output/mallet"
    os.makedirs(f"{output_path}/data", exist_ok=True)

    # convert to mallet format
    os.system(f"{mallet_path} import-dir --input {output_path}/data "
              f"--output {output_path}/topic-input.mallet --keep-sequence --remove-stopwords")

    # train topic model
    hyper = {
        "num_topics": 16,
        "num_iterations": 2000,
        "optimize-interval": 10,
        "num-top-words": 100,
    }

    os.system(f"{mallet_path} train-topics --input {output_path}/topic-input.mallet "
              f"--output-state {output_path}/topic-state.gz "
              f"--output-doc-topics {output_path}/doc_topics.txt "
              f"--output-topic-keys {output_path}/topic_keys.txt "
              f"--inferencer-filename {output_path}/model.mallet "
              f"--num-topics {hyper['num_topics']} --num-iterations {hyper['num_iterations']} --optimize-interval 10 "
              f"--num-top-words {hyper['num-top-words']}")
