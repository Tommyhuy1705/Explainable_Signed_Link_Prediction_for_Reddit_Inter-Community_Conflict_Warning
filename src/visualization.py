"""Reusable plotting helpers for notebook and report-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


FIGURE_DPI = 200
PALETTE = {
    "positive": "#2F5D8C",
    "negative": "#C84C4C",
    "negative_dark": "#8F1D2C",
    "neutral": "#5F9EA0",
    "muted": "#A7B0BA",
    "grid": "#E6E8EB",
    "text": "#222222",
}
MODEL_COLORS = {
    "logistic_regression": "#2F5D8C",
    "lightgbm": "#3D8B5B",
    "xgboost": "#D28B26",
    "random_forest": "#5F9EA0",
    "historical_negative_ratio": "#7A7A7A",
    "dummy_prior": "#B7BDC5",
    "dummy_most_frequent": "#D0D4DA",
}
FEATURE_COLORS = {
    "Text": "#8E6C8A",
    "Pair history": "#D28B26",
    "Node/network": "#2F5D8C",
    "Community": "#3D8B5B",
    "Structural balance": "#B85C38",
    "Other": "#8B949E",
}

sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={
        "axes.edgecolor": "#D7DADE",
        "axes.labelcolor": PALETTE["text"],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "figure.facecolor": "white",
        "font.family": "DejaVu Sans",
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    },
)


def _prepare_output_dir(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_figure(fig) -> None:
    """Display a matplotlib figure when running inside a notebook."""
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        fig.show()


def _finalize_figure(fig, path: Path | None = None, *, show: bool = False) -> Path | None:
    """Optionally display, save, and close a figure."""
    fig.tight_layout()
    if show:
        _display_figure(fig)
    if path is not None:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _output_path(output_dir: str | Path | None, filename: str) -> Path | None:
    output_path = _prepare_output_dir(output_dir)
    return output_path / filename if output_path is not None else None


def _despine(ax) -> None:
    sns.despine(ax=ax, left=False, bottom=False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _format_count(value: float) -> str:
    return f"{int(round(value)):,}"


def _model_display_name(model: str) -> str:
    names = {
        "dummy_most_frequent": "Dummy frequent",
        "dummy_prior": "Dummy prior",
        "historical_negative_ratio": "Historical ratio",
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    return names.get(model, model.replace("_", " ").title())


def _feature_set_display_name(feature_set: str) -> str:
    names = {
        "history_only": "History only",
        "text_only": "Text only",
        "graph_only": "Graph only",
        "graph_no_balance": "Graph no balance",
        "hybrid": "Hybrid",
        "hybrid_no_balance": "Hybrid no balance",
    }
    return names.get(feature_set, feature_set.replace("_", " ").title())


def _model_label(feature_set: str, model: str) -> str:
    return f"{_feature_set_display_name(feature_set)} | {_model_display_name(model)}"


def _feature_category(feature: str) -> str:
    if feature.startswith("text_property_") or feature.startswith("link_location_") or feature == "text_feature_count":
        return "Text"
    if feature in {"interaction_count", "positive_count", "negative_count", "negative_ratio", "sentiment_balance", "reciprocal_edge"}:
        return "Pair history"
    if "community" in feature or "clustering" in feature:
        return "Community"
    if feature.startswith("balance_") or feature == "common_neighbors":
        return "Structural balance"
    if any(token in feature for token in ["degree", "pagerank", "betweenness", "reciprocity"]):
        return "Node/network"
    return "Other"


def _feature_display_name(feature: str) -> str:
    if feature.startswith("text_property_"):
        return f"{feature} (SNAP text)"
    if feature.startswith("link_location_"):
        return feature.replace("link_location_", "link location: ")
    return feature.replace("_", " ")


def _community_size_map(node_features: pd.DataFrame) -> dict[int, int]:
    return node_features["community_id"].value_counts().astype(int).to_dict()


def _community_labels(node_features: pd.DataFrame, community_ids: list[int]) -> list[str]:
    sizes = _community_size_map(node_features)
    return [f"C{community}\nn={sizes.get(community, 0):,}" for community in community_ids]


def plot_label_distribution(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot positive/neutral versus negative hyperlink counts."""
    labels = frame["link_sentiment"].map({1: "positive/neutral", -1: "negative"})
    label_counts = labels.value_counts().reindex(["positive/neutral", "negative"]).fillna(0).astype(int)
    total = int(label_counts.sum())
    percentages = label_counts / max(total, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [PALETTE["positive"], PALETTE["negative"]]
    bars = ax.bar(label_counts.index, label_counts.values, color=colors, width=0.56)
    ax.set_title("Sentiment label distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of hyperlinks")
    ax.set_ylim(0, label_counts.max() * 1.18)
    ax.yaxis.set_major_formatter(lambda value, _: f"{int(value / 1000):,}K")
    ax.grid(axis="x", visible=False)

    for bar, count, percentage in zip(bars, label_counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_counts.max() * 0.025,
            f"{count:,}\n({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["text"],
        )
    ax.text(
        0.98,
        0.88,
        "Class imbalance is substantial,\nso PR-AUC is the main metric.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#4A4A4A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F6F7", "edgecolor": "#D7DADE"},
    )
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "label_distribution.png"), show=show)


def plot_monthly_negative_ratio(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot the monthly share of negative cross-community hyperlinks."""
    dated = frame.copy()
    dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce")
    dated = dated.dropna(subset=["timestamp"])
    dated["month"] = dated["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        dated.groupby("month")["link_sentiment"]
        .agg(total="size", negative=lambda values: int((values == -1).sum()))
        .reset_index()
        .sort_values("month")
    )
    monthly["negative_ratio"] = monthly["negative"] / monthly["total"].clip(lower=1)
    monthly["rolling_ratio"] = monthly["negative_ratio"].rolling(3, min_periods=1).mean()
    average_ratio = monthly["negative_ratio"].mean()

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(monthly["month"], monthly["negative_ratio"], color=PALETTE["negative"], alpha=0.35, linewidth=1.2, marker="o", markersize=3)
    ax.plot(monthly["month"], monthly["rolling_ratio"], color=PALETTE["negative_dark"], linewidth=2.5, label="3-month rolling average")
    ax.axhline(average_ratio, color="#6B7280", linestyle="--", linewidth=1.2, label=f"mean={average_ratio:.1%}")
    y_top = monthly["negative_ratio"].max() + 0.002
    for cutoff, label in [
        (pd.Timestamp("2015-12-31"), "train cutoff"),
        (pd.Timestamp("2016-06-30"), "validation cutoff"),
        (pd.Timestamp("2016-12-31"), "test cutoff"),
    ]:
        ax.axvline(cutoff, color="#B8BEC6", linestyle=":", linewidth=1.0)
        ax.annotate(
            label,
            xy=(cutoff, y_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#5A5F66",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D7DADE", "alpha": 0.92},
        )
    ax.set_title("Monthly negative hyperlink ratio over time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Negative ratio")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(monthly["negative_ratio"].min() - 0.002, y_top + 0.006)
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "monthly_negative_ratio.png"), show=show)


def plot_top_negative_subreddits(
    frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    show: bool = False,
) -> list[Path | None]:
    """Plot top source and target subreddits by negative hyperlink count."""
    negative_frame = frame[frame["link_sentiment"] == -1]
    total_negative = max(len(negative_frame), 1)
    paths: list[Path | None] = []
    for column, label, filename in [
        ("source_subreddit", "Source subreddit", "top_negative_sources.png"),
        ("target_subreddit", "Target subreddit", "top_negative_targets.png"),
    ]:
        counts = negative_frame[column].value_counts().head(top_n).sort_values()
        colors = [PALETTE["negative_dark"] if value == counts.max() else PALETTE["negative"] for value in counts.values]
        fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.35 * len(counts))))
        bars = ax.barh(counts.index, counts.values, color=colors, alpha=0.92)
        ax.set_title(f"Top {top_n} {label.lower()}s by negative hyperlinks")
        ax.set_xlabel("Negative hyperlink count")
        ax.set_ylabel("")
        ax.set_xlim(0, counts.max() * 1.18)
        ax.xaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
        for bar, value in zip(bars, counts.values):
            ax.text(
                value + counts.max() * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({value / total_negative:.1%})",
                va="center",
                fontsize=8.5,
                color=PALETTE["text"],
            )
        _despine(ax)
        paths.append(_finalize_figure(fig, _output_path(output_dir, filename), show=show))
    return paths


def _ccdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cleaned = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    cleaned = cleaned[cleaned > 0]
    if cleaned.size == 0:
        return np.array([]), np.array([])
    probabilities = (cleaned.size - np.arange(cleaned.size)) / cleaned.size
    return cleaned, probabilities


def plot_degree_distribution(frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot source and target degree CCDFs on a log-log scale."""
    out_degree = frame.groupby("source_subreddit")["target_subreddit"].nunique()
    in_degree = frame.groupby("target_subreddit")["source_subreddit"].nunique()
    out_x, out_y = _ccdf(out_degree)
    in_x, in_y = _ccdf(in_degree)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.step(out_x, out_y, where="post", color=PALETTE["positive"], linewidth=2.0, label="Out-degree")
    ax.step(in_x, in_y, where="post", color="#D28B26", linewidth=2.0, label="In-degree")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Heavy-tailed directed subreddit degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Share of subreddits with degree >= x")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.04,
        0.08,
        "A small number of subreddits\nreceive or create many links.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4A4A4A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F6F7", "edgecolor": "#D7DADE"},
    )
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "degree_distribution.png"), show=show)


def plot_model_comparison(metrics_frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot test PR-AUC for the strongest model and feature-set combinations."""
    plot_frame = metrics_frame.copy()
    plot_frame["model_label"] = [_model_label(row.feature_set, row.model) for row in plot_frame.itertuples(index=False)]
    plot_frame = plot_frame.sort_values("test_pr_auc", ascending=True).tail(20).reset_index(drop=True)
    best_index = int(plot_frame["test_pr_auc"].idxmax())
    colors = [
        PALETTE["negative_dark"] if index == best_index else MODEL_COLORS.get(model, PALETTE["muted"])
        for index, model in enumerate(plot_frame["model"])
    ]

    fig, ax = plt.subplots(figsize=(10.4, max(6.0, 0.36 * len(plot_frame))))
    bars = ax.barh(plot_frame["model_label"], plot_frame["test_pr_auc"], color=colors, alpha=0.95)
    baseline = metrics_frame[metrics_frame["model"] == "dummy_prior"]["test_pr_auc"].max()
    if pd.notna(baseline):
        ax.axvline(baseline, color="#6B7280", linestyle="--", linewidth=1.2)
        ax.text(baseline, len(plot_frame) - 0.35, "dummy prior", rotation=90, va="top", ha="right", fontsize=8, color="#6B7280")
    ax.set_title("Model comparison by test PR-AUC")
    ax.set_xlabel("Test PR-AUC")
    ax.set_ylabel("")
    ax.set_xlim(0, plot_frame["test_pr_auc"].max() * 1.16)
    for bar, value, index in zip(bars, plot_frame["test_pr_auc"], range(len(plot_frame))):
        ax.text(
            value + plot_frame["test_pr_auc"].max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
            fontweight="bold" if index == best_index else "normal",
        )
    handles = [
        Patch(facecolor=PALETTE["negative_dark"], label="Best model"),
        Patch(facecolor=MODEL_COLORS["logistic_regression"], label="Logistic regression"),
        Patch(facecolor=MODEL_COLORS["lightgbm"], label="LightGBM"),
        Patch(facecolor=MODEL_COLORS["xgboost"], label="XGBoost"),
        Patch(facecolor=MODEL_COLORS["random_forest"], label="Random forest"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "model_comparison_pr_auc.png"), show=show)


def plot_best_confusion_matrix(metrics_frame: pd.DataFrame, output_dir: str | Path | None = None, *, show: bool = False) -> Path | None:
    """Plot the row-normalized test confusion matrix for the best PR-AUC model."""
    best = metrics_frame.sort_values(["test_pr_auc", "test_f1"], ascending=False).iloc[0]
    matrix = np.array(
        [
            [int(best["test_tn"]), int(best["test_fp"])],
            [int(best["test_fn"]), int(best["test_tp"])],
        ],
        dtype=float,
    )
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    normalized = matrix / row_sums
    annotations = np.empty_like(matrix, dtype=object)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            annotations[row, col] = f"{int(matrix[row, col]):,}\n{normalized[row, col]:.1%}"

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["Predicted\nnon-negative", "Predicted\nnegative"],
        yticklabels=["Actual\nnon-negative", "Actual\nnegative"],
        cbar_kws={"label": "Row share"},
        linewidths=1.2,
        linecolor="white",
        ax=ax,
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Best model confusion matrix\n{_feature_set_display_name(best['feature_set'])} | {_model_display_name(best['model'])}")
    ax.set_xlabel("")
    ax.set_ylabel("")

    return _finalize_figure(fig, _output_path(output_dir, "best_confusion_matrix.png"), show=show)


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 20,
    show: bool = False,
) -> Path | None:
    """Plot feature importances for the best non-baseline model."""
    ranked_models = metrics_frame[
        ~metrics_frame["model"].str.startswith("dummy")
        & (metrics_frame["model"] != "historical_negative_ratio")
    ].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    if ranked_models.empty:
        raise ValueError("No fitted model with feature importances is available.")
    best = ranked_models.iloc[0]
    filtered = (
        importance_frame[
            (importance_frame["feature_set"] == best["feature_set"])
            & (importance_frame["model"] == best["model"])
        ]
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance")
        .copy()
    )
    filtered["category"] = filtered["feature"].map(_feature_category)
    filtered["feature_label"] = filtered["feature"].map(_feature_display_name)
    colors = [FEATURE_COLORS.get(category, FEATURE_COLORS["Other"]) for category in filtered["category"]]

    fig, ax = plt.subplots(figsize=(10.0, max(6.2, 0.36 * len(filtered))))
    bars = ax.barh(filtered["feature_label"], filtered["importance"], color=colors, alpha=0.95)
    ax.set_title(f"Top {top_n} explanatory features\n{_feature_set_display_name(best['feature_set'])} | {_model_display_name(best['model'])}")
    ax.set_xlabel("Absolute importance")
    ax.set_ylabel("")
    ax.set_xlim(0, filtered["importance"].max() * 1.14)
    for bar, value in zip(bars, filtered["importance"]):
        ax.text(value + filtered["importance"].max() * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
    handles = [Patch(facecolor=color, label=category) for category, color in FEATURE_COLORS.items() if category in set(filtered["category"])]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Feature group")
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "feature_importance_top20.png"), show=show)


def _select_curve_models(metrics_frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Select a compact, informative set of models for curve plots."""
    ranked = metrics_frame[~metrics_frame["model"].str.startswith("dummy")].sort_values(["test_pr_auc", "test_f1"], ascending=False)
    selected = ranked.head(top_n).copy()
    historical = ranked[ranked["model"] == "historical_negative_ratio"].head(1)
    if not historical.empty:
        selected = pd.concat([selected, historical], ignore_index=True).drop_duplicates(["feature_set", "model"])
    return selected


def plot_precision_recall_curve(
    score_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 5,
    show: bool = False,
) -> Path | None:
    """Plot test-set precision-recall curves for top models and a heuristic baseline."""
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)
    best_key = tuple(selected.iloc[0][["feature_set", "model"]]) if not selected.empty else None

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(subset["y_true"], subset["score"])
        is_best = (row.feature_set, row.model) == best_key
        label = f"{_feature_set_display_name(row.feature_set)} | {_model_display_name(row.model)} (AP={row.test_pr_auc:.3f})"
        ax.plot(
            recall,
            precision,
            linewidth=2.8 if is_best else 1.7,
            alpha=1.0 if is_best else 0.72,
            color=PALETTE["negative_dark"] if is_best else MODEL_COLORS.get(row.model, PALETTE["muted"]),
            label=label,
        )

    prevalence = test_scores["y_true"].mean() if not test_scores.empty else 0.0
    ax.axhline(prevalence, color="#6B7280", linestyle="--", linewidth=1.2, label=f"prevalence={prevalence:.3f}")
    ax.set_title("Precision-recall curves on strict temporal test set")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "precision_recall_curve.png"), show=show)


def plot_roc_curve(
    score_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 5,
    show: bool = False,
) -> Path | None:
    """Plot test-set ROC curves for top models and a heuristic baseline."""
    test_scores = score_frame[score_frame["split"] == "test"].copy()
    selected = _select_curve_models(metrics_frame, top_n=top_n)
    best_key = tuple(selected.iloc[0][["feature_set", "model"]]) if not selected.empty else None

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for row in selected.itertuples(index=False):
        subset = test_scores[(test_scores["feature_set"] == row.feature_set) & (test_scores["model"] == row.model)]
        if subset.empty or subset["y_true"].nunique() < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(subset["y_true"], subset["score"])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        is_best = (row.feature_set, row.model) == best_key
        label = f"{_feature_set_display_name(row.feature_set)} | {_model_display_name(row.model)} (AUC={roc_auc:.3f})"
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.8 if is_best else 1.7,
            alpha=1.0 if is_best else 0.72,
            color=PALETTE["negative_dark"] if is_best else MODEL_COLORS.get(row.model, PALETTE["muted"]),
            label=label,
        )

    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1.2, label="random")
    ax.set_title("ROC curves on strict temporal test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8, loc="lower right")
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "roc_curve.png"), show=show)


def plot_community_negative_ratio(
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    min_size: int = 20,
    show: bool = False,
) -> Path | None:
    """Plot communities with the highest average negative-link ratio."""
    required = {"community_id", "community_negative_ratio", "node"}
    if not required.issubset(node_features.columns):
        raise ValueError(f"node_features must contain columns: {sorted(required)}")

    summary = (
        node_features.groupby("community_id", dropna=False)
        .agg(
            community_size=("node", "size"),
            negative_ratio=("community_negative_ratio", "mean"),
            avg_pagerank=("pagerank", "mean") if "pagerank" in node_features.columns else ("node", "size"),
        )
        .reset_index()
    )
    summary = summary[summary["community_size"] >= min_size]
    summary = summary.sort_values(["negative_ratio", "community_size"], ascending=False).head(top_n)
    summary = summary.sort_values("negative_ratio")
    summary["label"] = summary.apply(lambda row: f"C{int(row.community_id)} (n={int(row.community_size):,})", axis=1)

    values = summary["negative_ratio"].to_numpy()
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    colors = [plt.get_cmap("Reds")(0.35 + 0.55 * value) for value in norm]

    fig, ax = plt.subplots(figsize=(8.8, max(5.3, 0.36 * len(summary))))
    bars = ax.barh(summary["label"], summary["negative_ratio"], color=colors)
    ax.set_title(f"Communities with highest negative-link ratios")
    ax.set_xlabel("Average negative hyperlink ratio")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, summary["negative_ratio"].max() * 1.18)
    for bar, value in zip(bars, summary["negative_ratio"]):
        ax.text(value + summary["negative_ratio"].max() * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=8.5)
    ax.text(
        0.01,
        -0.13,
        "Labels show detected community id and number of subreddits.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5A5F66",
    )
    _despine(ax)

    return _finalize_figure(fig, _output_path(output_dir, "community_negative_ratio.png"), show=show)


def plot_community_network_sample(
    interactions: pd.DataFrame,
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    max_nodes: int = 92,
    max_edges: int = 180,
    label_count: int = 10,
    show: bool = False,
) -> Path | None:
    """Plot a readable negative-link backbone among major subreddit communities."""
    required = {"node", "community_id", "community_size", "pagerank", "total_degree"}
    if not required.issubset(node_features.columns):
        raise ValueError(f"node_features must contain columns: {sorted(required)}")

    community_map = node_features.set_index("node")["community_id"].astype(int).to_dict()
    pagerank_map = node_features.set_index("node")["pagerank"].to_dict()
    degree_map = node_features.set_index("node")["total_degree"].to_dict()
    community_sizes = _community_size_map(node_features)
    top_communities = set(node_features["community_id"].value_counts().head(8).index.astype(int).tolist())

    edge_frame = interactions[["source_subreddit", "target_subreddit", "link_sentiment"]].copy()
    edge_frame["source_community"] = edge_frame["source_subreddit"].map(community_map)
    edge_frame["target_community"] = edge_frame["target_subreddit"].map(community_map)
    edge_frame = edge_frame[
        edge_frame["source_community"].isin(top_communities)
        & edge_frame["target_community"].isin(top_communities)
    ]
    edge_summary = (
        edge_frame.groupby(["source_subreddit", "target_subreddit"], dropna=False)["link_sentiment"]
        .agg(
            interaction_count="size",
            negative_count=lambda values: int((values == -1).sum()),
        )
        .reset_index()
    )
    edge_summary = edge_summary[edge_summary["negative_count"] > 0].copy()
    edge_summary["negative_ratio"] = edge_summary["negative_count"] / edge_summary["interaction_count"].clip(lower=1)
    edge_summary["source_pagerank"] = edge_summary["source_subreddit"].map(pagerank_map).fillna(0.0)
    edge_summary["target_pagerank"] = edge_summary["target_subreddit"].map(pagerank_map).fillna(0.0)
    edge_summary["score"] = (
        np.log1p(edge_summary["interaction_count"])
        * (edge_summary["source_pagerank"] + edge_summary["target_pagerank"] + 1e-8)
        * (1 + edge_summary["negative_ratio"])
        + 0.15 * np.log1p(edge_summary["negative_count"])
    )
    edge_summary = edge_summary.sort_values("score", ascending=False).head(max_edges)

    graph = nx.DiGraph()
    for row in edge_summary.itertuples(index=False):
        for node in [row.source_subreddit, row.target_subreddit]:
            graph.add_node(
                node,
                community_id=int(community_map.get(node, -1)),
                total_degree=float(degree_map.get(node, 1.0)),
                pagerank=float(pagerank_map.get(node, 0.0)),
            )
        graph.add_edge(
            row.source_subreddit,
            row.target_subreddit,
            weight=float(row.interaction_count),
            negative_ratio=float(row.negative_ratio),
        )

    if graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Negative-link subreddit backbone")
        ax.text(0.5, 0.5, "No sampled negative-link edges available", ha="center", va="center")
        ax.axis("off")
        return _finalize_figure(fig, _output_path(output_dir, "community_network_sample.png"), show=show)

    largest_component = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(largest_component).copy()
    if graph.number_of_nodes() > max_nodes:
        node_scores = {
            node: graph.degree(node, weight="weight") + 5000 * graph.nodes[node].get("pagerank", 0.0)
            for node in graph.nodes()
        }
        keep_nodes = sorted(node_scores, key=node_scores.get, reverse=True)[:max_nodes]
        graph = graph.subgraph(keep_nodes).copy()
        largest_component = max(nx.weakly_connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()

    nodes = list(graph.nodes())
    layout_graph = graph.to_undirected()
    pos = nx.spring_layout(layout_graph, seed=42, weight="weight", iterations=240, k=0.78 / np.sqrt(max(graph.number_of_nodes(), 1)))
    degree_values = np.array([graph.nodes[node].get("total_degree", 1.0) for node in nodes], dtype=float)
    log_degree = np.log1p(degree_values)
    node_sizes = 120 + 880 * (log_degree - log_degree.min()) / (log_degree.max() - log_degree.min() + 1e-9)

    visible_communities = sorted({graph.nodes[node].get("community_id", -1) for node in nodes})
    community_color_map = {community: plt.get_cmap("tab10")(index % 10) for index, community in enumerate(visible_communities)}
    node_colors = [community_color_map[graph.nodes[node].get("community_id", -1)] for node in nodes]
    edge_ratios = np.array([data.get("negative_ratio", 0.0) for _, _, data in graph.edges(data=True)], dtype=float)
    edge_widths = [0.5 + min(3.0, np.log1p(data.get("weight", 1.0)) / 2.0) for _, _, data in graph.edges(data=True)]

    fig, ax = plt.subplots(figsize=(11.7, 8.2))
    if graph.number_of_edges():
        edge_artist = nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color=edge_ratios,
            edge_cmap=plt.get_cmap("Reds"),
            edge_vmin=0,
            edge_vmax=max(0.25, float(np.nanquantile(edge_ratios, 0.9))),
            width=edge_widths,
            alpha=0.30,
            arrows=False,
        )
        colorbar = fig.colorbar(edge_artist, ax=ax, fraction=0.035, pad=0.01)
        colorbar.set_label("Edge negative-link ratio")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.93,
        linewidths=1.0,
        edgecolors="white",
    )
    label_nodes = sorted(
        nodes,
        key=lambda node: (graph.nodes[node].get("pagerank", 0.0), graph.nodes[node].get("total_degree", 0.0)),
        reverse=True,
    )[:label_count]
    labels = {node: node for node in label_nodes}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=7.7,
        font_color=PALETTE["text"],
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    legend_communities = visible_communities[:8]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"C{community} (n={community_sizes.get(community, 0):,})",
            markerfacecolor=community_color_map[community],
            markersize=8,
        )
        for community in legend_communities
    ]
    ax.legend(handles=handles, title="Detected community", loc="lower left", fontsize=8, title_fontsize=9)
    ax.set_title(f"Negative-link backbone among major Reddit communities\n{graph.number_of_nodes()} subreddits, {graph.number_of_edges()} directed edges")
    ax.axis("off")

    return _finalize_figure(fig, _output_path(output_dir, "community_network_sample.png"), show=show)


def plot_community_pair_negative_heatmap(
    interactions: pd.DataFrame,
    node_features: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    top_n: int = 15,
    show: bool = False,
) -> Path | None:
    """Plot negative-link ratios between the largest detected communities."""
    if not {"node", "community_id"}.issubset(node_features.columns):
        raise ValueError("node_features must contain 'node' and 'community_id'.")

    community_map = node_features.set_index("node")["community_id"].to_dict()
    working = interactions[["source_subreddit", "target_subreddit", "link_sentiment"]].copy()
    working["source_community"] = working["source_subreddit"].map(community_map)
    working["target_community"] = working["target_subreddit"].map(community_map)
    working = working.dropna(subset=["source_community", "target_community"])
    working["source_community"] = working["source_community"].astype(int)
    working["target_community"] = working["target_community"].astype(int)

    top_communities = node_features["community_id"].value_counts().head(top_n).index.astype(int).tolist()
    working = working[
        working["source_community"].isin(top_communities)
        & working["target_community"].isin(top_communities)
    ]
    pair_summary = (
        working.groupby(["source_community", "target_community"], dropna=False)["link_sentiment"]
        .agg(total="size", negative=lambda values: int((values == -1).sum()))
        .reset_index()
    )
    pair_summary["negative_ratio"] = pair_summary["negative"] / pair_summary["total"].clip(lower=1)
    pivot = pair_summary.pivot(index="source_community", columns="target_community", values="negative_ratio")
    pivot = pivot.reindex(index=top_communities, columns=top_communities)
    pivot = pivot.apply(pd.to_numeric, errors="coerce").astype(float)
    mask = pivot.isna()
    plot_data = pivot.fillna(0.0)
    labels = _community_labels(node_features, top_communities)
    vmax = max(0.12, float(pair_summary["negative_ratio"].quantile(0.90)) if not pair_summary.empty else 0.12)

    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    sns.heatmap(
        plot_data,
        mask=mask,
        cmap="Reds",
        vmin=0,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Negative hyperlink ratio"},
        linewidths=0.45,
        linecolor="white",
        ax=ax,
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    top_cells = pair_summary.sort_values("negative_ratio", ascending=False).head(7)
    index_positions = {community: index for index, community in enumerate(top_communities)}
    for row in top_cells.itertuples(index=False):
        if row.source_community in index_positions and row.target_community in index_positions:
            y = index_positions[row.source_community]
            x = index_positions[row.target_community]
            ax.text(x + 0.5, y + 0.5, f"{row.negative_ratio:.0%}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.set_title(f"Negative-link ratio between top {len(top_communities)} communities", pad=16)
    ax.text(
        0.0,
        1.015,
        "Color scale is capped at the 90th percentile to keep lower-intensity structure visible.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#5A5F66",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Target community", labelpad=10)
    ax.set_ylabel("Source community", labelpad=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8.2)
    ax.tick_params(axis="y", rotation=0, labelsize=8.2)

    return _finalize_figure(fig, _output_path(output_dir, "community_pair_negative_heatmap.png"), show=show)


def export_report_figures(
    interactions: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    output_dir: str | Path,
    score_frame: pd.DataFrame | None = None,
    node_features: pd.DataFrame | None = None,
    *,
    show: bool = False,
) -> dict[str, Path | list[Path | None] | None]:
    """Export the main figures expected in the final report."""
    figures: dict[str, Path | list[Path | None] | None] = {
        "label_distribution": plot_label_distribution(interactions, output_dir, show=show),
        "monthly_negative_ratio": plot_monthly_negative_ratio(interactions, output_dir, show=show),
        "top_negative_subreddits": plot_top_negative_subreddits(interactions, output_dir, show=show),
        "degree_distribution": plot_degree_distribution(interactions, output_dir, show=show),
        "model_comparison": plot_model_comparison(metrics_frame, output_dir, show=show),
        "best_confusion_matrix": plot_best_confusion_matrix(metrics_frame, output_dir, show=show),
        "feature_importance": plot_feature_importance(importance_frame, metrics_frame, output_dir, show=show),
    }
    if score_frame is not None and not score_frame.empty:
        figures["precision_recall_curve"] = plot_precision_recall_curve(score_frame, metrics_frame, output_dir, show=show)
        figures["roc_curve"] = plot_roc_curve(score_frame, metrics_frame, output_dir, show=show)
    if node_features is not None and not node_features.empty:
        figures["community_negative_ratio"] = plot_community_negative_ratio(node_features, output_dir, show=show)
        figures["community_network_sample"] = plot_community_network_sample(interactions, node_features, output_dir, show=show)
        figures["community_pair_negative_heatmap"] = plot_community_pair_negative_heatmap(interactions, node_features, output_dir, show=show)
    return figures
