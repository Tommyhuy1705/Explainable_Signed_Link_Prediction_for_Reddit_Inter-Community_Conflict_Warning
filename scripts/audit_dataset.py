"""Audit the raw Kaggle/SNAP Reddit Hyperlink files for reproducibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


EXPECTED_FILES = {
    "body": "soc-redditHyperlinks-body.tsv",
    "title": "soc-redditHyperlinks-title.tsv",
}
REQUIRED_COLUMNS = [
    "SOURCE_SUBREDDIT",
    "TARGET_SUBREDDIT",
    "POST_ID",
    "TIMESTAMP",
    "LINK_SENTIMENT",
    "PROPERTIES",
]
PROPERTY_COUNT = 86


def _property_length(value: object) -> int:
    parsed = np.fromstring(str(value), sep=",", dtype=np.float64)
    return int(parsed.size)


def audit_file(path: Path, dataset_source: str, *, chunksize: int = 100_000) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected raw file: {path}")

    header = pd.read_csv(path, sep="\t", nrows=0)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    rows = 0
    negative_rows = 0
    invalid_labels = 0
    invalid_timestamps = 0
    missing_core_fields = 0
    malformed_properties = 0
    timestamp_min = None
    timestamp_max = None

    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize):
        rows += len(chunk)
        if missing_columns:
            continue
        labels = pd.to_numeric(chunk["LINK_SENTIMENT"], errors="coerce")
        timestamps = pd.to_datetime(chunk["TIMESTAMP"], errors="coerce")
        property_lengths = chunk["PROPERTIES"].map(_property_length)

        negative_rows += int((labels == -1).sum())
        invalid_labels += int((~labels.isin([-1, 1])).sum())
        invalid_timestamps += int(timestamps.isna().sum())
        missing_core_fields += int(chunk[REQUIRED_COLUMNS[:-1]].isna().any(axis=1).sum())
        malformed_properties += int((property_lengths != PROPERTY_COUNT).sum())

        current_min = timestamps.min()
        current_max = timestamps.max()
        if pd.notna(current_min):
            timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
        if pd.notna(current_max):
            timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

    return {
        "dataset_source": dataset_source,
        "file": str(path),
        "rows": rows,
        "negative_rows": negative_rows,
        "negative_ratio": negative_rows / rows if rows else 0.0,
        "missing_columns": missing_columns,
        "missing_core_fields": missing_core_fields,
        "invalid_labels": invalid_labels,
        "invalid_timestamps": invalid_timestamps,
        "malformed_properties": malformed_properties,
        "timestamp_min": str(timestamp_min) if timestamp_min is not None else None,
        "timestamp_max": str(timestamp_max) if timestamp_max is not None else None,
    }


def run_audit(raw_dir: Path) -> dict[str, object]:
    file_results = [
        audit_file(raw_dir / filename, source)
        for source, filename in EXPECTED_FILES.items()
    ]
    total_rows = sum(result["rows"] for result in file_results)
    total_negative = sum(result["negative_rows"] for result in file_results)
    timestamp_values = [
        value
        for result in file_results
        for value in [result["timestamp_min"], result["timestamp_max"]]
        if value is not None
    ]
    return {
        "raw_dir": str(raw_dir),
        "kaggle_dataset": "https://www.kaggle.com/datasets/wolfram77/graphs-signed",
        "original_source": "https://snap.stanford.edu/data/soc-RedditHyperlinks.html",
        "files": file_results,
        "combined": {
            "rows": total_rows,
            "negative_rows": total_negative,
            "negative_ratio": total_negative / total_rows if total_rows else 0.0,
            "timestamp_min": min(timestamp_values) if timestamp_values else None,
            "timestamp_max": max(timestamp_values) if timestamp_values else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    audit = run_audit(args.raw_dir)
    text = json.dumps(audit, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")

    failing = []
    for result in audit["files"]:
        if result["missing_columns"]:
            failing.append(f"{result['dataset_source']}: missing columns")
        if result["missing_core_fields"]:
            failing.append(f"{result['dataset_source']}: missing core fields")
        if result["invalid_labels"]:
            failing.append(f"{result['dataset_source']}: invalid labels")
        if result["invalid_timestamps"]:
            failing.append(f"{result['dataset_source']}: invalid timestamps")
        if result["malformed_properties"]:
            failing.append(f"{result['dataset_source']}: malformed properties")
    if failing:
        print("Audit failed: " + "; ".join(failing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
