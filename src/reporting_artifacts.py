"""Report-facing robustness, threshold, and error-analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .phase1 import apply_k_core_filter
from .phase2 import aggregate_edge_table
from .phase3 import PAIR_HISTORY_FEATURES, TARGET_COLUMN, _aggregate_future_labels


FIGURE_DPI = 200
COLOR_MAIN = "#8F1D2C"
COLOR_BLUE = "#2F5D8C"
COLOR_GREEN = "#3D8B5B"
COLOR_MUTED = "#6B7280"
GRID = "#E6E8EB"


def _prepare_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _finalize(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def build_threshold_tradeoff(
    score_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_set: str = "hybrid",
    model: str = "logistic_regression",
) -> tuple[pd.DataFrame, Path]:
    """Export precision, recall, and F1 across thresholds for the best model."""
    output_path = _prepare_output_dir(output_dir)
    subset = score_frame[
        (score_frame["split"] == "test")
        & (score_frame["feature_set"] == feature_set)
        & (score_frame["model"] == model)
    ].copy()
    if subset.empty:
        raise ValueError(f"No test scores found for {feature_set} / {model}.")

    rows = []
    y_true = subset["y_true"].astype(int).to_numpy()
    scores = subset["score"].astype(float).to_numpy()
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (scores >= threshold).astype(int)
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "threshold": float(threshold),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )
    tradeoff = pd.DataFrame(rows)
    tradeoff_path = output_path / "threshold_tradeoff.csv"
    tradeoff.to_csv(tradeoff_path, index=False)

    best_idx = int(tradeoff["f1"].idxmax())
    best = tradeoff.loc[best_idx]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(tradeoff["threshold"], tradeoff["precision"], color=COLOR_BLUE, linewidth=2.2, label="Precision")
    ax.plot(tradeoff["threshold"], tradeoff["recall"], color=COLOR_GREEN, linewidth=2.2, label="Recall")
    ax.plot(tradeoff["threshold"], tradeoff["f1"], color=COLOR_MAIN, linewidth=2.8, label="F1")
    ax.axvline(best["threshold"], color=COLOR_MUTED, linestyle="--", linewidth=1.2)
    ax.scatter([best["threshold"]], [best["f1"]], color=COLOR_MAIN, zorder=5)
    ax.text(
        best["threshold"] + 0.015,
        best["f1"] + 0.015,
        f"best F1={best['f1']:.3f}\nthreshold={best['threshold']:.2f}",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D7DADE"},
    )
    ax.set_title("Threshold trade-off for the best temporal model")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID)
    ax.legend(loc="upper right")
    figure_path = _finalize(fig, output_path / "threshold_tradeoff.png")
    return tradeoff, figure_path


def _history_pair_table(frame: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    history = frame[frame["timestamp"] <= history_end].copy()
    table = aggregate_edge_table(history)
    return table.sort_values(["source_subreddit", "target_subreddit"]).reset_index(drop=True)


def _pair_split(frame: pd.DataFrame, history_end: pd.Timestamp, label_end: pd.Timestamp) -> pd.DataFrame:
    features = _history_pair_table(frame, history_end)
    labels = _aggregate_future_labels(frame, history_end, label_end)
    merged = features.merge(labels, on=["source_subreddit", "target_subreddit"], how="inner")
    return merged.reset_index(drop=True)


def build_error_analysis_cases(
    interactions: pd.DataFrame,
    score_frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_set: str = "hybrid",
    model: str = "logistic_regression",
    per_group: int = 10,
) -> pd.DataFrame:
    """Export representative TP/FP/FN cases with interpretable history signals."""
    output_path = _prepare_output_dir(output_dir)
    test_pairs = _pair_split(
        interactions,
        pd.Timestamp("2016-12-31 23:59:59"),
        pd.Timestamp("2017-04-30 23:59:59"),
    )
    scores = score_frame[
        (score_frame["split"] == "test")
        & (score_frame["feature_set"] == feature_set)
        & (score_frame["model"] == model)
    ].reset_index(drop=True)
    if len(scores) != len(test_pairs):
        raise ValueError(
            f"Score rows ({len(scores)}) and reconstructed test pairs ({len(test_pairs)}) do not align."
        )

    cases = pd.concat(
        [
            test_pairs[
                [
                    "source_subreddit",
                    "target_subreddit",
                    "interaction_count",
                    "positive_count",
                    "negative_count",
                    "negative_ratio",
                    "future_positive_count",
                    "future_negative_count",
                    "future_interaction_count",
                ]
            ],
            scores[["y_true", "score", "prediction", "threshold"]],
        ],
        axis=1,
    )
    cases["case_type"] = np.select(
        [
            (cases["y_true"] == 1) & (cases["prediction"] == 1),
            (cases["y_true"] == 0) & (cases["prediction"] == 1),
            (cases["y_true"] == 1) & (cases["prediction"] == 0),
        ],
        ["true_positive", "false_positive", "false_negative"],
        default="other",
    )
    cases["top_contributing_features"] = cases.apply(
        lambda row: (
            f"score={row.score:.3f}; prior_negative_ratio={row.negative_ratio:.3f}; "
            f"prior_negative_count={int(row.negative_count)}; "
            f"prior_positive_count={int(row.positive_count)}; "
            f"history_interactions={int(row.interaction_count)}"
        ),
        axis=1,
    )

    selected_frames = []
    for case_type, ascending in [
        ("true_positive", False),
        ("false_positive", False),
        ("false_negative", True),
    ]:
        selected_frames.append(
            cases[cases["case_type"] == case_type]
            .sort_values("score", ascending=ascending)
            .head(per_group)
        )
    selected = pd.concat(selected_frames, ignore_index=True)
    selected.to_csv(output_path / "error_analysis_cases.csv", index=False)
    return selected


def _history_safe_k_core(frame: pd.DataFrame, k: int, history_end: pd.Timestamp) -> pd.DataFrame:
    history = frame[frame["timestamp"] <= history_end].copy()
    graph = nx.Graph()
    graph.add_edges_from(history[["source_subreddit", "target_subreddit"]].itertuples(index=False, name=None))
    if graph.number_of_nodes() == 0:
        return frame.iloc[0:0].copy()
    core_nodes = set(nx.k_core(graph, k=k).nodes())
    return frame[
        frame["source_subreddit"].isin(core_nodes) & frame["target_subreddit"].isin(core_nodes)
    ].reset_index(drop=True)


def _evaluate_history_model(frame: pd.DataFrame) -> dict[str, float | int]:
    splits = {
        "train": _pair_split(frame, pd.Timestamp("2015-12-31 23:59:59"), pd.Timestamp("2016-06-30 23:59:59")),
        "validation": _pair_split(frame, pd.Timestamp("2016-06-30 23:59:59"), pd.Timestamp("2016-12-31 23:59:59")),
        "test": _pair_split(frame, pd.Timestamp("2016-12-31 23:59:59"), pd.Timestamp("2017-04-30 23:59:59")),
    }
    feature_columns = [column for column in PAIR_HISTORY_FEATURES if column in splits["train"].columns]
    if len(splits["train"]) == 0 or splits["train"][TARGET_COLUMN].nunique() < 2:
        return {
            "train_pairs": len(splits["train"]),
            "validation_pairs": len(splits["validation"]),
            "test_pairs": len(splits["test"]),
            "test_pr_auc": float("nan"),
            "test_roc_auc": float("nan"),
            "test_f1": float("nan"),
            "test_precision": float("nan"),
            "test_recall": float("nan"),
            "test_balanced_accuracy": float("nan"),
        }

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
        ]
    )
    x_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN].astype(int)
    x_test = splits["test"][feature_columns]
    y_test = splits["test"][TARGET_COLUMN].astype(int)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return {
        "train_pairs": len(splits["train"]),
        "validation_pairs": len(splits["validation"]),
        "test_pairs": len(splits["test"]),
        "test_pr_auc": average_precision_score(y_test, scores) if y_test.nunique() > 1 else float("nan"),
        "test_roc_auc": roc_auc_score(y_test, scores) if y_test.nunique() > 1 else float("nan"),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, predictions) if y_test.nunique() > 1 else float("nan"),
    }


def build_kcore_robustness(
    combined_interactions: pd.DataFrame,
    output_dir: str | Path,
    *,
    k_values: tuple[int, ...] = (3, 5, 10),
    max_rows: int | None = 100_000,
) -> tuple[pd.DataFrame, Path]:
    """Run a lightweight k-core sensitivity probe with pair-history logistic models."""
    output_path = _prepare_output_dir(output_dir)
    working = combined_interactions.sort_values("timestamp").reset_index(drop=True)
    sampled = False
    if max_rows is not None and len(working) > max_rows:
        positions = np.linspace(0, len(working) - 1, max_rows).astype(int)
        working = working.iloc[positions].reset_index(drop=True)
        sampled = True

    rows: list[dict[str, object]] = []
    history_end = pd.Timestamp("2015-12-31 23:59:59")
    for mode in ["global_k_core", "history_safe_k_core"]:
        for k in k_values:
            if mode == "global_k_core":
                filtered = apply_k_core_filter(working, k=k)
            else:
                filtered = _history_safe_k_core(working, k=k, history_end=history_end)
            metrics = _evaluate_history_model(filtered)
            rows.append(
                {
                    "filter_mode": mode,
                    "k_core": k,
                    "model": "history_logistic_regression",
                    "sampled_probe": sampled,
                    "source_rows": len(combined_interactions),
                    "probe_rows": len(working),
                    "interaction_rows": len(filtered),
                    **metrics,
                }
            )
    robustness = pd.DataFrame(rows)
    robustness_path = output_path / "robustness_metrics.csv"
    robustness.to_csv(robustness_path, index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for mode, color in [("global_k_core", COLOR_BLUE), ("history_safe_k_core", COLOR_MAIN)]:
        subset = robustness[robustness["filter_mode"] == mode].sort_values("k_core")
        label = mode.replace("_", " ")
        ax.plot(subset["k_core"], subset["test_pr_auc"], marker="o", linewidth=2.4, color=color, label=label)
        for row in subset.itertuples(index=False):
            if pd.notna(row.test_pr_auc):
                ax.text(row.k_core, row.test_pr_auc + 0.002, f"{row.test_pr_auc:.3f}", ha="center", fontsize=8)
    ax.set_title("K-core robustness probe using history-only logistic models")
    ax.set_xlabel("k-core threshold")
    ax.set_ylabel("Test PR-AUC")
    ax.set_xticks(list(k_values))
    ax.grid(color=GRID)
    ax.legend(loc="best")
    figure_path = _finalize(fig, output_path / "robustness_kcore_pr_auc.png")
    return robustness, figure_path
