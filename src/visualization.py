"""Reusable plotting helpers for report-ready project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


FIGURE_DPI = 180
PALETTE = {"positive": "#4C78A8", "negative": "#E45756", "neutral": "#72B7B2"}


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_current_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_label_distribution(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot positive/neutral versus negative hyperlink counts."""
    output_path = _prepare_output_dir(output_dir)
    label_counts = frame["link_sentiment"].map({1: "positive/neutral", -1: "negative"}).value_counts()

    plt.figure(figsize=(7, 4))
    colors = [PALETTE["negative"] if label == "negative" else PALETTE["positive"] for label in label_counts.index]
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette=colors, legend=False)
    plt.title("Distribution of Reddit hyperlink sentiment labels")
    plt.xlabel("Label")
    plt.ylabel("Number of hyperlinks")
    for index, value in enumerate(label_counts.values):
        plt.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    return _save_current_figure(output_path / "label_distribution.png")


def plot_monthly_negative_ratio(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot the monthly share of negative cross-community hyperlinks."""
    output_path = _prepare_output_dir(output_dir)
    dated = frame.copy()
    dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce")
    dated = dated.dropna(subset=["timestamp"])
    dated["month"] = dated["timestamp"].dt.to_period("M").astype(str)
    monthly = dated.groupby("month")["link_sentiment"].agg(
        total="size",
        negative=lambda values: int((values == -1).sum()),
    ).reset_index()
    monthly["negative_ratio"] = monthly["negative"] / monthly["total"].clip(lower=1)

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly, x="month", y="negative_ratio", marker="o", color=PALETTE["negative"])
    plt.title("Monthly negative hyperlink ratio")
    plt.xlabel("Month")
    plt.ylabel("Negative ratio")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    return _save_current_figure(output_path / "monthly_negative_ratio.png")


def plot_top_negative_subreddits(frame: pd.DataFrame, output_dir: str | Path, top_n: int = 15) -> list[Path]:
    """Plot top source and target subreddits by negative hyperlink count."""
    output_path = _prepare_output_dir(output_dir)
    negative_frame = frame[frame["link_sentiment"] == -1]
    paths = []
    for column, label, filename in [
        ("source_subreddit", "Source subreddit", "top_negative_sources.png"),
        ("target_subreddit", "Target subreddit", "top_negative_targets.png"),
    ]:
        counts = negative_frame[column].value_counts().head(top_n).sort_values()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.values, y=counts.index, color=PALETTE["negative"])
        plt.title(f"Top {top_n} {label.lower()}s by negative links")
        plt.xlabel("Negative hyperlink count")
        plt.ylabel(label)
        paths.append(_save_current_figure(output_path / filename))
    return paths


def plot_degree_distribution(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot source and target degree distributions on a log scale."""
    output_path = _prepare_output_dir(output_dir)
    out_degree = frame.groupby("source_subreddit")["target_subreddit"].nunique()
    in_degree = frame.groupby("target_subreddit")["source_subreddit"].nunique()
    degree_frame = pd.concat(
        [
            pd.DataFrame({"degree": out_degree, "type": "out-degree"}),
            pd.DataFrame({"degree": in_degree, "type": "in-degree"}),
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(8, 4))
    sns.histplot(data=degree_frame, x="degree", hue="type", bins=60, log_scale=(True, True), element="step")
    plt.title("Directed subreddit degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of subreddits")
    return _save_current_figure(output_path / "degree_distribution.png")


def plot_model_comparison(metrics_frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot test PR-AUC for each model and feature set."""
    output_path = _prepare_output_dir(output_dir)
    plot_frame = metrics_frame.copy()
    plot_frame["model_label"] = plot_frame["feature_set"] + " | " + plot_frame["model"]
    plot_frame = plot_frame.sort_values("test_pr_auc", ascending=True).tail(20)

    plt.figure(figsize=(9, max(5, 0.35 * len(plot_frame))))
    sns.barplot(data=plot_frame, x="test_pr_auc", y="model_label", color=PALETTE["positive"])
    plt.title("Model comparison on strict temporal test set")
    plt.xlabel("Test PR-AUC")
    plt.ylabel("Feature set | model")
    return _save_current_figure(output_path / "model_comparison_pr_auc.png")


def plot_best_confusion_matrix(metrics_frame: pd.DataFrame, output_dir: str | Path) -> Path:
    """Plot the test confusion matrix for the best PR-AUC model."""
    output_path = _prepare_output_dir(output_dir)
    best = metrics_frame.sort_values(["test_pr_auc", "test_f1"], ascending=False).iloc[0]
    matrix = [
        [int(best["test_tn"]), int(best["test_fp"])],
        [int(best["test_fn"]), int(best["test_tp"])],
    ]

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(f"Best model confusion matrix: {best['feature_set']} | {best['model']}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    return _save_current_figure(output_path / "best_confusion_matrix.png")


def plot_feature_importance(importance_frame: pd.DataFrame, metrics_frame: pd.DataFrame, output_dir: str | Path, top_n: int = 20) -> Path:
    """Plot feature importances for the best non-dummy model."""
    output_path = _prepare_output_dir(output_dir)
    ranked_models = metrics_frame[
        ~metrics_frame["model"].str.startswith("dummy")
        & (metrics_frame["model"] != "historical_negative_ratio")
    ].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    if ranked_models.empty:
        raise ValueError("No fitted model with feature importances is available.")
    best = ranked_models.iloc[0]
    filtered = importance_frame[
        (importance_frame["feature_set"] == best["feature_set"])
        & (importance_frame["model"] == best["model"])
    ].sort_values("importance", ascending=False).head(top_n).sort_values("importance")

    plt.figure(figsize=(8, max(5, 0.3 * len(filtered))))
    sns.barplot(data=filtered, x="importance", y="feature", color=PALETTE["neutral"])
    plt.title(f"Top {top_n} features: {best['feature_set']} | {best['model']}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    return _save_current_figure(output_path / "feature_importance_top20.png")


def _select_curve_models(metrics_frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Select a compact, informative set of models for curve plots."""
    ranked = metrics_frame[
        ~metrics_frame["model"].str.startswith("dummy")
    ].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    selected = ranked.head(top_n).copy()
    historical = ranked[ranked["model"] == "historical_negative_ratio"].head(1)
    if not historical.empty:
        selected = pd.concat([selected, historical], ignore_index=True).drop_duplicates(["feature_set", "model"])
    return selected


def plot_precision_recall_curve(score_frame: pd.DataFrame, metrics_frame: pd.DataFrame, output_dir: str | Path, top_n: int = 5) -> Path:
    """Plot test-set precision-recall curves for top models and a strong heuristic baseline."""
    output_path = _prepare_output_dir(output_dir)
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)

    plt.figure(figsize=(7, 5))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(subset["y_true"], subset["score"])
        label = f"{row.feature_set} | {row.model} (AP={row.test_pr_auc:.3f})"
        plt.plot(recall, precision, linewidth=1.8, label=label)

    prevalence = test_scores["y_true"].mean() if not test_scores.empty else 0.0
    plt.axhline(prevalence, color="gray", linestyle="--", linewidth=1.2, label=f"prevalence={prevalence:.3f}")
    plt.title("Precision-recall curves on strict temporal test set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    return _save_current_figure(output_path / "precision_recall_curve.png")


def plot_roc_curve(score_frame: pd.DataFrame, metrics_frame: pd.DataFrame, output_dir: str | Path, top_n: int = 5) -> Path:
    """Plot test-set ROC curves for top models and a strong heuristic baseline."""
    output_path = _prepare_output_dir(output_dir)
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)

    plt.figure(figsize=(7, 5))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(subset["y_true"], subset["score"])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        label = f"{row.feature_set} | {row.model} (AUC={roc_auc:.3f})"
        plt.plot(false_positive_rate, true_positive_rate, linewidth=1.8, label=label)

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="random")
    plt.title("ROC curves on strict temporal test set")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    return _save_current_figure(output_path / "roc_curve.png")


def export_report_figures(
    interactions: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    output_dir: str | Path,
    score_frame: pd.DataFrame | None = None,
) -> dict[str, Path | list[Path]]:
    """Export the main figures expected in the final report."""
    figures: dict[str, Path | list[Path]] = {
        "label_distribution": plot_label_distribution(interactions, output_dir),
        "monthly_negative_ratio": plot_monthly_negative_ratio(interactions, output_dir),
        "top_negative_subreddits": plot_top_negative_subreddits(interactions, output_dir),
        "degree_distribution": plot_degree_distribution(interactions, output_dir),
        "model_comparison": plot_model_comparison(metrics_frame, output_dir),
        "best_confusion_matrix": plot_best_confusion_matrix(metrics_frame, output_dir),
        "feature_importance": plot_feature_importance(importance_frame, metrics_frame, output_dir),
    }
    if score_frame is not None and not score_frame.empty:
        figures["precision_recall_curve"] = plot_precision_recall_curve(score_frame, metrics_frame, output_dir)
        figures["roc_curve"] = plot_roc_curve(score_frame, metrics_frame, output_dir)
    return figures
