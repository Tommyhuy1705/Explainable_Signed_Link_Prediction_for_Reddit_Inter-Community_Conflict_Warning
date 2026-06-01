from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.phase2 import parse_property_matrix, property_feature_names
from src.phase3 import TARGET_COLUMN, build_strict_temporal_splits


def _properties(value: float = 0.0) -> str:
    return ",".join([str(value)] * 86)


def _synthetic_temporal_frame() -> pd.DataFrame:
    rows = []
    for pair_index, (source, target) in enumerate([("alpha", "beta"), ("gamma", "delta")]):
        for timestamp, sentiment in [
            ("2015-01-10", 1),
            ("2015-02-10", -1 if pair_index == 0 else 1),
            ("2016-02-10", -1 if pair_index == 0 else 1),
            ("2016-08-10", -1 if pair_index == 0 else 1),
            ("2017-02-10", -1 if pair_index == 0 else 1),
        ]:
            rows.append(
                {
                    "source_subreddit": source,
                    "target_subreddit": target,
                    "post_id": f"{source}_{timestamp}",
                    "timestamp": pd.Timestamp(timestamp),
                    "link_sentiment": sentiment,
                    "properties": _properties(float(pair_index)),
                    "dataset_source": "title",
                }
            )
    return pd.DataFrame(rows)


def test_property_parser_always_returns_86_columns() -> None:
    parsed = parse_property_matrix(pd.Series([_properties(1.0), "1,2,3", ""]))
    assert list(parsed.columns) == property_feature_names()
    assert parsed.shape == (3, 86)


def test_temporal_splits_use_disjoint_future_labels() -> None:
    splits = build_strict_temporal_splits(_synthetic_temporal_frame())
    assert len(splits.train) == 2
    assert len(splits.validation) == 2
    assert len(splits.test) == 2
    assert splits.train["future_start"].eq(pd.Timestamp("2015-12-31 23:59:59")).all()
    assert splits.train["future_end"].eq(pd.Timestamp("2016-06-30 23:59:59")).all()
    assert splits.validation["future_start"].eq(pd.Timestamp("2016-06-30 23:59:59")).all()
    assert splits.test["future_start"].eq(pd.Timestamp("2016-12-31 23:59:59")).all()
    assert set(splits.train[TARGET_COLUMN]) == {0, 1}


def test_metrics_schema_from_existing_artifact() -> None:
    metrics_path = Path("data/processed/phase3/phase3_model_metrics.csv")
    assert metrics_path.exists(), "Run phase 3 before validating the exported metric schema."
    metrics = pd.read_csv(metrics_path, nrows=5)
    expected = {
        "feature_set",
        "model",
        "validation_pr_auc",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "n_features",
    }
    assert expected.issubset(metrics.columns)


def test_smoke_cli_runs_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--stage", "smoke"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Smoke pipeline completed" in result.stdout
