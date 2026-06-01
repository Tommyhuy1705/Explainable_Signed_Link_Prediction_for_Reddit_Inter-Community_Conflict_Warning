"""Run reproducible project stages from raw data to report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import phase3, visualization
from src.phase1 import apply_k_core_filter, combine_raw_datasets, temporal_split
from src.phase2 import export_phase2_tables, load_phase1_filtered
from src.reporting_artifacts import (
    build_error_analysis_cases,
    build_kcore_robustness,
    build_threshold_tradeoff,
)


def _raw_paths(raw_dir: Path) -> tuple[Path, Path]:
    return raw_dir / "soc-redditHyperlinks-body.tsv", raw_dir / "soc-redditHyperlinks-title.tsv"


def run_phase1(raw_dir: Path, processed_dir: Path, interim_dir: Path, *, k_core: int) -> dict[str, Path]:
    body_path, title_path = _raw_paths(raw_dir)
    combined = combine_raw_datasets(body_path, title_path)
    filtered = apply_k_core_filter(combined, k=k_core)
    train_df, validation_df, test_df = temporal_split(filtered)

    phase1_dir = processed_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "combined": interim_dir / "phase1_combined_clean.csv",
        "filtered": phase1_dir / "phase1_kcore_filtered.csv",
        "train": phase1_dir / "phase1_train.csv",
        "validation": phase1_dir / "phase1_validation.csv",
        "test": phase1_dir / "phase1_test.csv",
    }
    combined.to_csv(paths["combined"], index=False)
    filtered.to_csv(paths["filtered"], index=False)
    train_df.to_csv(paths["train"], index=False)
    validation_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def run_phase2(processed_dir: Path) -> dict[str, Path]:
    phase1_path = processed_dir / "phase1" / "phase1_kcore_filtered.csv"
    phase2_dir = processed_dir / "phase2"
    frame = load_phase1_filtered(phase1_path)
    return export_phase2_tables(frame, phase2_dir)


def run_phase3(processed_dir: Path) -> dict[str, Path]:
    phase1_path = processed_dir / "phase1" / "phase1_kcore_filtered.csv"
    phase3_dir = processed_dir / "phase3"
    interactions = phase3.load_phase1_interactions(phase1_path)
    metrics, importance, scores, _ = phase3.run_phase3_pipeline(interactions)
    return phase3.export_phase3_outputs(metrics, importance, phase3_dir, scores)


def run_figures(processed_dir: Path, figure_dir: Path) -> dict[str, object]:
    phase1_path = processed_dir / "phase1" / "phase1_kcore_filtered.csv"
    phase2_dir = processed_dir / "phase2"
    phase3_dir = processed_dir / "phase3"

    interactions = phase3.load_phase1_interactions(phase1_path)
    metrics = pd.read_csv(phase3_dir / "phase3_model_metrics.csv")
    importance = pd.read_csv(phase3_dir / "phase3_feature_importance.csv")
    scores = pd.read_csv(phase3_dir / "phase3_prediction_scores.csv")
    node_features = pd.read_csv(phase2_dir / "phase2_node_features.csv")

    try:
        figure_paths = visualization.export_report_figures(
            interactions,
            metrics,
            importance,
            figure_dir,
            scores,
            node_features,
            show=False,
        )
    except PermissionError as exc:
        print(f"Warning: existing figure appears locked; keeping current figure set. Details: {exc}")
        figure_paths = {path.stem: path for path in figure_dir.glob("*.png")}
    _, threshold_path = build_threshold_tradeoff(scores, figure_dir)
    cases = build_error_analysis_cases(interactions, scores, phase3_dir)
    figure_paths["threshold_tradeoff"] = threshold_path
    figure_paths["error_analysis_cases"] = phase3_dir / "error_analysis_cases.csv"
    figure_paths["error_analysis_case_count"] = len(cases)
    return figure_paths


def run_robustness(interim_dir: Path, processed_dir: Path, figure_dir: Path) -> dict[str, Path]:
    combined_path = interim_dir / "phase1_combined_clean.csv"
    if not combined_path.exists():
        raise FileNotFoundError("Run phase1 before robustness so phase1_combined_clean.csv exists.")
    combined = pd.read_csv(combined_path)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    robustness, figure_path = build_kcore_robustness(combined, processed_dir / "phase3")
    target_figure = figure_dir / figure_path.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    target_figure.write_bytes(figure_path.read_bytes())
    return {
        "metrics": processed_dir / "phase3" / "robustness_metrics.csv",
        "figure": target_figure,
        "rows": len(robustness),
    }


def _property_string(fill: float = 0.0) -> str:
    return ",".join([str(fill)] * 86)


def run_smoke() -> None:
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
                    "properties": _property_string(float(pair_index)),
                    "dataset_source": "title",
                }
            )
    frame = pd.DataFrame(rows)
    metrics, importance, scores, splits = phase3.run_phase3_pipeline(frame)
    assert not metrics.empty
    assert not importance.empty
    assert not scores.empty
    assert len(splits.train) == 2
    print("Smoke pipeline completed:", {"metrics": len(metrics), "scores": len(scores)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["phase1", "phase2", "phase3", "figures", "all", "smoke"], default="all")
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    if args.stage == "smoke":
        run_smoke()
        return 0

    stages = ["phase1", "phase2", "phase3", "figures"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"Running {stage}...")
        if stage == "phase1":
            print(run_phase1(args.raw_dir, args.output_dir, args.interim_dir, k_core=args.k_core))
        elif stage == "phase2":
            print(run_phase2(args.output_dir))
        elif stage == "phase3":
            print(run_phase3(args.output_dir))
        elif stage == "figures":
            print(run_figures(args.output_dir, args.figure_dir))
            print(run_robustness(args.interim_dir, args.output_dir, args.figure_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
