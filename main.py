import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def format_tsv(tsv_path):
    """
    Function to format the Reddit Hyperlink dataset from Tab-Separated format
    to Comma-Separated format. Following conversion, the CSV is reduced to contain
    only fields relevant to this project

    :param tsv_path: Path to TSV file
    :return: None
    """
    df = pd.read_table(tsv_path, sep="\t")
    df.to_csv("./data/soc-redditHyperlinks-body.csv")

    reddit_csv = "./data/soc-redditHyperlinks-body.csv"

    reddit_df = pd.read_csv(reddit_csv)

    reddit_df = reddit_df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]]

    reddit_df.to_csv("./data/redditHyperlinks-subredditsOnly.csv")


if __name__ == '__main__':
    pass
