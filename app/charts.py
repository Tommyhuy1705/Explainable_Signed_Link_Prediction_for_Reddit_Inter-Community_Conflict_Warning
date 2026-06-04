"""Plotly chart builders for the Streamlit research dashboard."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRIC_LABELS = {
    "test_pr_auc": "Test PR-AUC",
    "test_roc_auc": "Test ROC-AUC",
    "test_f1": "Negative-class F1",
    "test_precision": "Precision",
    "test_recall": "Recall",
}

MODEL_COLORS = {
    "logistic_regression": "#8F1D2C",
    "random_forest": "#6AA6A6",
    "xgboost": "#D99027",
    "lightgbm": "#3D8B5B",
    "historical_negative_ratio": "#2F5D8C",
    "dummy_prior": "#B7BDC5",
    "dummy_most_frequent": "#D0D4DA",
}

NEGATIVE_EDGE = "#A11D2D"
POSITIVE_EDGE = "#2F5D8C"
FOCUS_NODE = "#111827"
NEIGHBOR_NODE = "#6AA6A6"
RISK_NODE = "#8F1D2C"
CHART_TEXT = "#111827"
CHART_MUTED = "#5F6876"
CHART_GRID = "#E6E8EB"
CHART_BORDER = "#D8DEE7"


def pretty_name(value: str) -> str:
    """Convert artifact ids into readable labels."""
    return value.replace("_", " ").title()


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Keep Plotly text readable even when the surrounding Streamlit theme changes."""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "size": 13, "color": CHART_TEXT},
        title_font={"color": CHART_TEXT},
        legend_font={"color": CHART_TEXT},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": CHART_BORDER, "font_color": CHART_TEXT},
    )
    fig.update_xaxes(
        color=CHART_MUTED,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        title_font={"color": CHART_MUTED},
        tickfont={"color": CHART_MUTED},
    )
    fig.update_yaxes(
        color=CHART_MUTED,
        gridcolor=CHART_GRID,
        zerolinecolor=CHART_GRID,
        title_font={"color": CHART_MUTED},
        tickfont={"color": CHART_MUTED},
    )
    return fig


def model_metric_bar(
    metrics: pd.DataFrame,
    *,
    metric: str = "test_pr_auc",
    feature_sets: list[str] | None = None,
    top_n: int = 20,
) -> go.Figure:
    """Build a ranked model-comparison bar chart."""
    if metrics.empty or metric not in metrics.columns:
        return go.Figure()

    frame = metrics.copy()
    if feature_sets:
        frame = frame[frame["feature_set"].isin(feature_sets)]
    frame = frame.sort_values([metric, "test_f1"], ascending=False).head(top_n)
    frame["label"] = frame["feature_set"].map(pretty_name) + " | " + frame["model"].map(pretty_name)
    frame["model_color"] = frame["model"].map(MODEL_COLORS).fillna("#6B7280")

    fig = go.Figure()
    fig.add_bar(
        x=frame[metric],
        y=frame["label"],
        orientation="h",
        marker_color=frame["model_color"],
        text=frame[metric].map(lambda value: f"{value:.3f}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{METRIC_LABELS.get(metric, metric)}: " + "%{x:.4f}<br>"
            "<extra></extra>"
        ),
    )

    baseline = metrics.loc[metrics["model"] == "dummy_prior", "test_pr_auc"]
    if metric == "test_pr_auc" and not baseline.empty:
        baseline_value = float(baseline.max())
        fig.add_vline(
            x=baseline_value,
            line_dash="dash",
            line_color="#6B7280",
            annotation_text="dummy prior",
            annotation_position="top left",
        )

    fig.update_layout(
        height=max(460, 34 * len(frame) + 140),
        margin={"l": 8, "r": 56, "t": 52, "b": 36},
        title=f"Model comparison by {METRIC_LABELS.get(metric, metric)}",
        xaxis_title=METRIC_LABELS.get(metric, metric),
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(gridcolor="#E6E8EB")
    return apply_chart_theme(fig)


def signed_ego_network_figure(
    edges: pd.DataFrame,
    *,
    focus: str,
    direction: str = "Outgoing",
    min_negative_ratio: float = 0.3,
    max_edges: int = 45,
    sort_by: str = "negative_count",
) -> tuple[go.Figure, pd.DataFrame]:
    """Build an interactive signed ego-network around one subreddit."""
    required = {
        "source_subreddit",
        "target_subreddit",
        "interaction_count",
        "positive_count",
        "negative_count",
        "negative_ratio",
    }
    if edges.empty or not required.issubset(edges.columns):
        return go.Figure(), pd.DataFrame()

    direction_key = direction.casefold()
    if direction_key == "incoming":
        mask = edges["target_subreddit"] == focus
    elif direction_key == "both":
        mask = (edges["source_subreddit"] == focus) | (edges["target_subreddit"] == focus)
    else:
        mask = edges["source_subreddit"] == focus

    frame = edges.loc[mask].copy()
    frame = frame[frame["negative_ratio"] >= min_negative_ratio]
    if frame.empty:
        return go.Figure(), frame

    sort_column = sort_by if sort_by in frame.columns else "negative_count"
    frame = frame.sort_values([sort_column, "interaction_count"], ascending=False).head(max_edges).copy()

    graph = nx.DiGraph()
    for row in frame.itertuples(index=False):
        graph.add_edge(
            row.source_subreddit,
            row.target_subreddit,
            weight=max(float(row.interaction_count), 1.0),
            negative_ratio=float(row.negative_ratio),
        )

    if graph.number_of_nodes() <= 1:
        return go.Figure(), frame

    layout_graph = graph.to_undirected()
    k_value = 1.25 / np.sqrt(max(graph.number_of_nodes(), 2))
    positions = nx.spring_layout(layout_graph, seed=7, k=k_value, iterations=90, weight="weight")
    weighted_degree = dict(graph.degree(weight="weight"))
    incident_risk: dict[str, float] = {node: 0.0 for node in graph.nodes}
    for row in frame.itertuples(index=False):
        ratio = float(row.negative_ratio)
        incident_risk[row.source_subreddit] = max(incident_risk[row.source_subreddit], ratio)
        incident_risk[row.target_subreddit] = max(incident_risk[row.target_subreddit], ratio)

    annotations = []
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_text: list[str] = []
    mid_color: list[str] = []
    for row in frame.itertuples(index=False):
        source_x, source_y = positions[row.source_subreddit]
        target_x, target_y = positions[row.target_subreddit]
        ratio = float(row.negative_ratio)
        color = NEGATIVE_EDGE if ratio >= 0.5 else POSITIVE_EDGE
        width = float(np.clip(np.log1p(float(row.interaction_count)), 1.2, 4.8))
        annotations.append(
            {
                "x": target_x,
                "y": target_y,
                "ax": source_x,
                "ay": source_y,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1,
                "arrowwidth": width,
                "arrowcolor": color,
                "opacity": 0.52,
            }
        )
        mid_x.append((source_x + target_x) / 2)
        mid_y.append((source_y + target_y) / 2)
        mid_color.append(color)
        mid_text.append(
            "<br>".join(
                [
                    f"<b>{row.source_subreddit} -> {row.target_subreddit}</b>",
                    f"links: {int(row.interaction_count):,}",
                    f"negative: {int(row.negative_count):,}",
                    f"positive: {int(row.positive_count):,}",
                    f"negative ratio: {ratio:.3f}",
                ]
            )
        )

    top_label_nodes = {focus}
    top_label_nodes.update(
        node for node, _ in sorted(weighted_degree.items(), key=lambda item: item[1], reverse=True)[:12]
    )
    node_x: list[float] = []
    node_y: list[float] = []
    node_size: list[float] = []
    node_color: list[str] = []
    node_text: list[str] = []
    node_label: list[str] = []
    for node in graph.nodes:
        x_value, y_value = positions[node]
        degree_value = float(weighted_degree.get(node, 1.0))
        node_x.append(float(x_value))
        node_y.append(float(y_value))
        node_size.append(float(np.clip(12 + np.log1p(degree_value) * 4.3, 14, 42)))
        if node == focus:
            node_color.append(FOCUS_NODE)
        elif incident_risk.get(node, 0.0) >= 0.5:
            node_color.append(RISK_NODE)
        else:
            node_color.append(NEIGHBOR_NODE)
        node_label.append(node if node in top_label_nodes else "")
        node_text.append(
            "<br>".join(
                [
                    f"<b>{node}</b>",
                    f"weighted degree: {degree_value:,.0f}",
                    f"max incident negative ratio: {incident_risk.get(node, 0.0):.3f}",
                ]
            )
        )

    fig = go.Figure()
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={"color": NEGATIVE_EDGE, "width": 4},
        name="negative-dominant edge",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        line={"color": POSITIVE_EDGE, "width": 4},
        name="mixed/positive edge",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker={"size": 8, "color": mid_color, "line": {"width": 1, "color": "#FFFFFF"}},
        text=mid_text,
        hovertemplate="%{text}<extra></extra>",
        name="edge evidence",
        showlegend=False,
    )
    fig.add_scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker={"size": node_size, "color": node_color, "line": {"width": 1.5, "color": "#FFFFFF"}},
        text=node_label,
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        name="subreddit",
        showlegend=False,
    )
    fig.update_layout(
        annotations=annotations,
        height=650,
        margin={"l": 0, "r": 0, "t": 46, "b": 0},
        title=f"Signed ego-network around {focus}",
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
    )
    return apply_chart_theme(fig), frame


def threshold_scan(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute precision, recall, F1, and confusion counts across thresholds."""
    if scores.empty:
        return pd.DataFrame()

    y_true = scores["y_true"].astype(int).to_numpy()
    score_values = scores["score"].astype(float).to_numpy()
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = score_values >= threshold
        actual = y_true == 1
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "predicted_positive_rate": float(predicted.mean()),
            }
        )
    return pd.DataFrame(rows)


def metrics_at_threshold(scores: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    """Compute binary classification metrics for the selected threshold."""
    if scores.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    y_true = scores["y_true"].astype(int).to_numpy()
    score_values = scores["score"].astype(float).to_numpy()
    predicted = score_values >= threshold
    actual = y_true == 1
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_curve(tradeoff: pd.DataFrame, selected_threshold: float) -> go.Figure:
    """Build the threshold trade-off chart."""
    if tradeoff.empty:
        return go.Figure()

    fig = go.Figure()
    for column, color in [
        ("precision", "#2F5D8C"),
        ("recall", "#3D8B5B"),
        ("f1", "#8F1D2C"),
    ]:
        fig.add_scatter(
            x=tradeoff["threshold"],
            y=tradeoff[column],
            mode="lines",
            name=pretty_name(column),
            line={"color": color, "width": 3 if column == "f1" else 2.4},
        )
    fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color="#6B7280",
        annotation_text=f"threshold {selected_threshold:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        height=430,
        margin={"l": 12, "r": 24, "t": 44, "b": 36},
        title="Precision, recall, and F1 across decision thresholds",
        xaxis_title="Decision threshold",
        yaxis_title="Metric value",
        yaxis_range=[0, 1.02],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": -0.18},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
    )
    fig.update_xaxes(gridcolor="#E6E8EB")
    fig.update_yaxes(gridcolor="#E6E8EB")
    return apply_chart_theme(fig)


def confusion_matrix_figure(values: dict[str, float | int]) -> go.Figure:
    """Build a compact confusion-matrix heatmap."""
    matrix = [[values["tn"], values["fp"]], [values["fn"], values["tp"]]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[row][col]}<br>{matrix[row][col]:,}" for col in range(2)] for row in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#F6E6E9"], [1, "#8F1D2C"]],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        height=330,
        margin={"l": 12, "r": 12, "t": 38, "b": 18},
        title="Confusion matrix at selected threshold",
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
    )
    return apply_chart_theme(fig)


def robustness_line(robustness: pd.DataFrame) -> go.Figure:
    """Build the k-core robustness comparison chart."""
    if robustness.empty:
        return go.Figure()

    frame = robustness.copy()
    frame["filter_mode_label"] = frame["filter_mode"].str.replace("_", " ").str.title()
    fig = px.line(
        frame,
        x="k_core",
        y="test_pr_auc",
        color="filter_mode_label",
        markers=True,
        text=frame["test_pr_auc"].map(lambda value: f"{value:.3f}"),
        color_discrete_sequence=["#2F5D8C", "#8F1D2C"],
    )
    fig.update_traces(textposition="top center", line={"width": 3})
    fig.update_layout(
        height=390,
        margin={"l": 12, "r": 24, "t": 46, "b": 42},
        title="K-core robustness probe using history-only logistic models",
        xaxis_title="k-core threshold",
        yaxis_title="Test PR-AUC",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text=None,
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
    )
    fig.update_xaxes(dtick=1, gridcolor="#E6E8EB")
    fig.update_yaxes(gridcolor="#E6E8EB")
    return apply_chart_theme(fig)
