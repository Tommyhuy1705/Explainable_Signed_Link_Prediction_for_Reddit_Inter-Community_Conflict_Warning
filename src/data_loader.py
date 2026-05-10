"""Data loading and preprocessing utilities."""

from pathlib import Path

import pandas as pd


def load_tsv(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load a TSV file into a pandas DataFrame."""
    return pd.read_csv(path, sep="\t", nrows=nrows)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case for downstream processing."""
    normalized = frame.copy()
    normalized.columns = [column.strip().lower().replace(" ", "_") for column in normalized.columns]
    return normalized
