import networkx
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import urllib.request as req
import os.path as path

from collections.abc import Iterable
from networkx.algorithms import community as comm


def _download_tsv():
    """
    Downloads the Reddit Hyperlink dataset TSV file and saves it to ./data
    :return: None
    """
    url = 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'
    body_save = "./data/soc-redditHyperlinks-body.tsv"
    print("Checking for 'soc-redditHyperlinks-body.tsv'...")
    if not path.exists(body_save):
        print("File not Found.")
        print(f"Downloading 'soc-redditHyperlinks-body.tsv' from URL: '{url}'...")
        req.urlretrieve(url, body_save)
    else:
        print(f"Found at Path: '{body_save}'!")

    url = 'https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv'
    title_save = "./data/soc-redditHyperlinks-title.tsv"
    print("Checking for 'soc-redditHyperlinks-title.tsv'...")
    if not path.exists(title_save):
        print("File not Found.")
        print(f"Downloading 'soc-redditHyperlinks-title.tsv' from URL: '{url}'...")
        req.urlretrieve(url, title_save)
        print("Done!")
    else:
        print(f"Found at Path: '{title_save}'!")


def format_tsv(title_path, body_path):
    """
    Function to format the Reddit Hyperlink dataset from Tab-Separated format
    to Comma-Separated format. Following conversion, the CSV is reduced to contain
    only fields relevant to this project

    :param title_path: Path to TSV file containing Hyperlinks in Title of posts
    :param body_path: Path to TSV file containing Hyperlinks in Body of posts
    :return: None
    """
    # Load the TSV files and convert to CSV files
    title_df = pd.read_table(title_path, sep="\t")
    title_csv = title_path[:-4] + ".csv"
    print("Checking for 'soc-redditHyperlinks-title.csv'...")
    if not path.exists(title_csv):
        print(f"Converting TSV to CSV file at: '{title_csv}'...")
        title_df.to_csv(title_csv)
        print("Converted!")
    else:
        print(f"Found at Path: {title_csv}")

    body_df = pd.read_table(body_path, sep='\t')
    body_csv = body_path[:-4] + ".csv"
    print("Checking for 'soc-redditHyperlinks-body.csv'...")
    if not path.exists(body_csv):
        print(f"Converting TSV to CSV file at: '{body_csv}'...")
        body_df.to_csv(body_csv)
        print("Converted!")
    else:
        print(f"Found at Path: {body_csv}")

    # Load CSV files into DataFrames, concatenate them and extract Source/Target nodes
    body_df = pd.read_csv(body_csv)
    title_df = pd.read_csv(title_csv)

    print("Creating edgelist DataFrame...")
    reddit_df = pd.concat([title_df, body_df]).reset_index(drop=True)
    reddit_df = reddit_df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]]

    reddit_df = reddit_df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).size().reset_index(name='WEIGHT')
    print("Done!")

    reddit_path = "./data/redditHyperlinks-subredditsOnly.csv"
    print(f"Saving edge details to CSV file at: '{reddit_path}'...")
    reddit_df.to_csv(reddit_path, index=False)
    print("Saved!")


def load_df(csv_path):
    """
    Loads the Edgelist CSV into a Dataframe
    :param csv_path: Path to the Edgelist CSV
    :return: Edgelist Dataframe
    """
    df = pd.read_csv(csv_path)

    return df


def _get_nodes_from_source(source, edgelist):
    if not isinstance(source, list):
        source = [source]

    df = edgelist[edgelist["SOURCE_SUBREDDIT"].isin(source)]

    return df


def _get_nodes_to_target(target, edgelist):
    if not isinstance(target, list):
        target = [target]

    df = edgelist[edgelist["TARGET_SUBREDDIT"].isin(target)]

    return df


def _get_nodes_combined(node, edgelist):
    out_edges = _get_nodes_from_source(node, edgelist)
    in_edges = _get_nodes_to_target(node, edgelist)

    combined = pd.concat([out_edges, in_edges]).sort_index()

    return combined


def make_unweighted(df):
    """
    Makes a NetworkX Unweighted graph from an Edgelist dataframe
    :param df: Edgelist Dataframe
    :return: NetworkX Graph
    """
    G = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT')

    return G


def make_weighted(df: pandas.DataFrame) -> networkx.classes.graph.Graph:
    G = nx.from_pandas_edgelist(df, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', ['WEIGHT'])

    return G


def perform_louvain(graph: networkx.classes.graph.Graph, resolution: float = 1.0) -> list:
    louvain = comm.louvain_communities(graph, weight='WEIGHT', resolution=resolution)

    return louvain


if __name__ == '__main__':
    _download_tsv()

    body_tsv = "./data/soc-redditHyperlinks-body.tsv"
    title_tsv = "./data/soc-redditHyperlinks-title.tsv"

    format_tsv(title_tsv, body_tsv)

    reddit_csv = "./data/redditHyperlinks-subredditsOnly.csv"
    reddit_df = load_df(reddit_csv)

    # destiny_df = _get_nodes_to_target("destinythegame", reddit_df)
    # destiny_df = _get_nodes_combined(["destinythegame", "raidsecrets"], reddit_df)

    weighted = make_weighted(reddit_df)

    all_neighbours = nx.all_neighbors(weighted, "destinythegame")
    neigh_list = []

    for neighbour in all_neighbours:
        neigh_list.append(neighbour)

    directed = weighted.to_directed()

    succ_list = []
    successors = directed.successors("destinythegame")

    for successor in successors:
        succ_list.append(successor)

    directed_df = nx.to_pandas_edgelist(directed)
    # destinygraph = make_unweighted(destiny_df)

    louvain = perform_louvain(weighted, resolution=0.1)

    # nx.draw_spring(destinygraph, with_labels=False)
    # plt.savefig("./out/destiny.png", dpi=1200)

    # nx.draw_shell(unweighted, with_labels=False)
    # plt.savefig("./network.png")
