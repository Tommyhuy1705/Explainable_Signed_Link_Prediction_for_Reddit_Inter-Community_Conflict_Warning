"""Data loading helpers for the Streamlit research dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

try:  # Streamlit is optional for lightweight import checks.
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"
APP_DATA_DIR = ROOT / "reports" / "app_data"

DATASET_AUDIT_PATH = DATA_DIR / "dataset_audit.json"
METRICS_PATH = DATA_DIR / "phase3" / "phase3_model_metrics.csv"
SCORES_PATH = DATA_DIR / "phase3" / "phase3_prediction_scores.csv"
ERROR_CASES_PATH = DATA_DIR / "phase3" / "error_analysis_cases.csv"
ROBUSTNESS_PATH = DATA_DIR / "phase3" / "robustness_metrics.csv"
EDGE_FEATURES_PATH = DATA_DIR / "phase2" / "phase2_edge_features.csv"

APP_DATASET_AUDIT_PATH = APP_DATA_DIR / "dataset_audit.json"
APP_METRICS_PATH = APP_DATA_DIR / "model_metrics.csv"
APP_ERROR_CASES_PATH = APP_DATA_DIR / "error_analysis_cases.csv"
APP_ROBUSTNESS_PATH = APP_DATA_DIR / "robustness_metrics.csv"
APP_EDGE_FEATURES_PATH = APP_DATA_DIR / "edge_features.csv"
APP_THRESHOLD_SCAN_PATH = APP_DATA_DIR / "threshold_scan.csv"
APP_SUMMARY_PATH = APP_DATA_DIR / "summary.json"

_F = TypeVar("_F", bound=Callable)


def _cache_data(func: _F) -> _F:
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)  # type: ignore[return-value]


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""
    return ROOT.joinpath(*parts)


def figure_path(filename: str) -> Path:
    """Return a report figure path."""
    return FIGURE_DIR / filename


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@_cache_data
def load_dataset_audit() -> dict[str, object]:
    """Load the reproducible raw-dataset audit JSON."""
    path = _first_existing(DATASET_AUDIT_PATH, APP_DATASET_AUDIT_PATH)
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@_cache_data
def load_metrics() -> pd.DataFrame:
    """Load exported phase 3 model metrics."""
    path = _first_existing(METRICS_PATH, APP_METRICS_PATH)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@_cache_data
def load_error_cases() -> pd.DataFrame:
    """Load representative TP/FP/FN case-analysis rows."""
    path = _first_existing(ERROR_CASES_PATH, APP_ERROR_CASES_PATH)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@_cache_data
def load_robustness() -> pd.DataFrame:
    """Load sampled k-core robustness metrics."""
    path = _first_existing(ROBUSTNESS_PATH, APP_ROBUSTNESS_PATH)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


@_cache_data
def load_edge_features() -> pd.DataFrame:
    """Load signed subreddit-to-subreddit edge features for network exploration."""
    path = _first_existing(EDGE_FEATURES_PATH, APP_EDGE_FEATURES_PATH)
    if path is None:
        return pd.DataFrame()
    usecols = [
        "source_subreddit",
        "target_subreddit",
        "interaction_count",
        "positive_count",
        "negative_count",
        "negative_ratio",
        "reciprocal_edge",
    ]
    return pd.read_csv(path, usecols=usecols)


@_cache_data
def load_threshold_scan() -> pd.DataFrame:
    """Load precomputed threshold trade-off metrics for Cloud deployment."""
    if not APP_THRESHOLD_SCAN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(APP_THRESHOLD_SCAN_PATH)


@_cache_data
def load_prediction_scores(
    *,
    feature_set: str = "hybrid",
    model: str = "logistic_regression",
    split: str = "test",
) -> pd.DataFrame:
    """Load prediction scores for one model/feature-set/split.

    The score file is intentionally read in chunks because the full artifact has
    more than two million rows. The dashboard only needs one model slice for the
    threshold simulator.
    """
    if not SCORES_PATH.exists():
        return pd.DataFrame(columns=["y_true", "score"])

    usecols = ["split", "feature_set", "model", "y_true", "score"]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(SCORES_PATH, usecols=usecols, chunksize=250_000):
        selected = chunk[
            (chunk["split"] == split)
            & (chunk["feature_set"] == feature_set)
            & (chunk["model"] == model)
        ]
        if not selected.empty:
            frames.append(selected[["y_true", "score"]].copy())

    if not frames:
        return pd.DataFrame(columns=["y_true", "score"])
    return pd.concat(frames, ignore_index=True)


@_cache_data
def count_csv_rows(relative_path: str) -> int | None:
    """Count data rows in a CSV artifact without loading the whole file."""
    path = ROOT / relative_path
    if not path.exists():
        summary = load_app_summary()
        row_counts = summary.get("row_counts", {}) if isinstance(summary, dict) else {}
        if isinstance(row_counts, dict):
            return row_counts.get(relative_path.replace("\\", "/"))
        return None
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


@_cache_data
def load_app_summary() -> dict[str, object]:
    """Load compact app-only summary values used when large CSVs are absent."""
    if not APP_SUMMARY_PATH.exists():
        return {}
    return json.loads(APP_SUMMARY_PATH.read_text(encoding="utf-8"))


def artifact_status() -> list[tuple[str, bool]]:
    """Return core artifact availability for the app sidebar."""
    required = [
        ("Dataset audit", _first_existing(DATASET_AUDIT_PATH, APP_DATASET_AUDIT_PATH) or DATASET_AUDIT_PATH),
        ("Model metrics", _first_existing(METRICS_PATH, APP_METRICS_PATH) or METRICS_PATH),
        ("Threshold metrics", _first_existing(SCORES_PATH, APP_THRESHOLD_SCAN_PATH) or SCORES_PATH),
        ("Error cases", _first_existing(ERROR_CASES_PATH, APP_ERROR_CASES_PATH) or ERROR_CASES_PATH),
        ("Robustness metrics", _first_existing(ROBUSTNESS_PATH, APP_ROBUSTNESS_PATH) or ROBUSTNESS_PATH),
        ("Edge features", _first_existing(EDGE_FEATURES_PATH, APP_EDGE_FEATURES_PATH) or EDGE_FEATURES_PATH),
        ("Report figures", FIGURE_DIR),
    ]
    return [(label, path.exists()) for label, path in required]
