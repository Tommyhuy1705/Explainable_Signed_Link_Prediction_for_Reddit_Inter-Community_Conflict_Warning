"""Phase 1 data preparation helpers for the Reddit signed network project."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from .data_loader import load_tsv, standardize_columns


REQUIRED_COLUMNS = [
    "source_subreddit",
    "target_subreddit",
    "post_id",
    "timestamp",
    "link_sentiment",
    "properties",
]


def load_raw_hyperlinks(path: str | Path) -> pd.DataFrame:
    """Load one raw Reddit hyperlink TSV file and standardize its schema."""
    frame = standardize_columns(load_tsv(path))
    rename_map = {
        "source_subreddit": "source_subreddit",
        "target_subreddit": "target_subreddit",
        "post_id": "post_id",
        "timestamp": "timestamp",
        "link_sentiment": "link_sentiment",
        "properties": "properties",
    }
    frame = frame.rename(columns=rename_map)
    frame = frame[[column for column in REQUIRED_COLUMNS if column in frame.columns]].copy()
    frame["source_subreddit"] = frame["source_subreddit"].astype("string").str.strip().str.lower()
    frame["target_subreddit"] = frame["target_subreddit"].astype("string").str.strip().str.lower()
    frame["post_id"] = frame["post_id"].astype("string").str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["link_sentiment"] = pd.to_numeric(frame["link_sentiment"], errors="coerce").astype("Int64")
    frame["properties"] = frame["properties"].astype("string")
    return frame


def combine_raw_datasets(body_path: str | Path, title_path: str | Path) -> pd.DataFrame:
    """Combine body and title hyperlink datasets into one cleaned frame."""
    body_frame = load_raw_hyperlinks(body_path)
    body_frame["dataset_source"] = "body"
    title_frame = load_raw_hyperlinks(title_path)
    title_frame["dataset_source"] = "title"
    combined = pd.concat([body_frame, title_frame], ignore_index=True)
    return clean_hyperlinks(combined)


def clean_hyperlinks(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows, drop duplicates, and sort by timestamp."""
    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=["source_subreddit", "target_subreddit", "timestamp", "link_sentiment"])
    cleaned = cleaned[cleaned["source_subreddit"] != cleaned["target_subreddit"]]
    cleaned = cleaned.drop_duplicates(subset=["source_subreddit", "target_subreddit", "post_id", "timestamp", "link_sentiment"])
    cleaned = cleaned.sort_values(["timestamp", "source_subreddit", "target_subreddit"]).reset_index(drop=True)
    return cleaned


def summarize_hyperlinks(frame: pd.DataFrame) -> dict[str, object]:
    """Build a compact summary for EDA and reporting."""
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    sentiment_counts = frame["link_sentiment"].value_counts(dropna=False).to_dict()
    top_sources = frame["source_subreddit"].value_counts().head(10)
    top_targets = frame["target_subreddit"].value_counts().head(10)
    return {
        "rows": int(len(frame)),
        "unique_source_subreddits": int(frame["source_subreddit"].nunique()),
        "unique_target_subreddits": int(frame["target_subreddit"].nunique()),
        "timestamp_min": timestamps.min(),
        "timestamp_max": timestamps.max(),
        "sentiment_counts": sentiment_counts,
        "top_source_subreddits": top_sources,
        "top_target_subreddits": top_targets,
    }


def build_undirected_graph(frame: pd.DataFrame) -> nx.Graph:
    """Build an undirected projection for k-core filtering."""
    graph = nx.Graph()
    edges = frame[["source_subreddit", "target_subreddit"]].dropna().itertuples(index=False, name=None)
    graph.add_edges_from(edges)
    return graph


def apply_k_core_filter(frame: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Keep only hyperlinks whose endpoints survive the k-core filter."""
    graph = build_undirected_graph(frame)
    if graph.number_of_nodes() == 0:
        return frame.iloc[0:0].copy()

    core_graph = nx.k_core(graph, k=k)
    core_nodes = set(core_graph.nodes())
    filtered = frame[
        frame["source_subreddit"].isin(core_nodes) & frame["target_subreddit"].isin(core_nodes)
    ].copy()
    return filtered.reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame,
    train_end: str = "2016-12-31 23:59:59",
    validation_end: str = "2017-02-28 23:59:59",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the cleaned frame into train, validation, and test partitions."""
    ordered = frame.sort_values("timestamp").copy()
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    train_frame = ordered[ordered["timestamp"] <= train_cutoff].copy()
    validation_frame = ordered[(ordered["timestamp"] > train_cutoff) & (ordered["timestamp"] <= validation_cutoff)].copy()
    test_frame = ordered[ordered["timestamp"] > validation_cutoff].copy()
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True), test_frame.reset_index(drop=True)