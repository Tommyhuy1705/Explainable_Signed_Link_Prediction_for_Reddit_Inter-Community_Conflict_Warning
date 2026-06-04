"""Streamlit dashboard for the Reddit signed-network final project."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import charts
import data_access


ACCENT = "#8F1D2C"
BLUE = "#2F5D8C"
GREEN = "#3D8B5B"
MUTED = "#6B7280"
PRESENTATION_BLOCK_EXACT = {
    "ass",
    "askgaybros",
    "polyamory",
}
PRESENTATION_BLOCK_SUBSTRINGS = (
    "bluepill",
    "butt",
    "darknet",
    "drug",
    "fasc",
    "fuck",
    "gonewild",
    "hotchick",
    "jihad",
    "nazi",
    "nsfw",
    "opiates",
    "porn",
    "purplepill",
    "puss",
    "redpill",
    "shit",
    "slut",
    "terror",
)


def fmt_int(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def fmt_float(value: int | float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_presentation_safe_name(name: object) -> bool:
    text = str(name).casefold()
    if text in PRESENTATION_BLOCK_EXACT:
        return False
    return not any(term in text for term in PRESENTATION_BLOCK_SUBSTRINGS)


def presentation_safe_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return cases
    safe_mask = cases["source_subreddit"].map(is_presentation_safe_name) & cases["target_subreddit"].map(
        is_presentation_safe_name
    )
    return cases[safe_mask].copy()


def presentation_safe_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges
    safe_mask = edges["source_subreddit"].map(is_presentation_safe_name) & edges["target_subreddit"].map(
        is_presentation_safe_name
    )
    return edges[safe_mask].copy()


def parse_feature_summary(value: object) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for part in str(value).split(";"):
        if "=" not in part:
            continue
        label, raw_value = part.split("=", 1)
        rows.append({"signal": charts.pretty_name(label.strip()), "value": raw_value.strip()})
    return pd.DataFrame(rows)


def show_image(filename: str, caption: str | None = None) -> None:
    path = data_access.figure_path(filename)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing figure: {path}")


def sorted_metrics(metrics: pd.DataFrame, metric: str, feature_sets: list[str] | None = None) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frame = metrics.copy()
    if feature_sets:
        frame = frame[frame["feature_set"].isin(feature_sets)]
    return frame.sort_values([metric, "test_f1"], ascending=False)


def network_focus_options(edges: pd.DataFrame, limit: int = 250) -> list[str]:
    if edges.empty:
        return []
    source_strength = (
        edges.groupby("source_subreddit", as_index=False)
        .agg({"negative_count": "sum", "interaction_count": "sum"})
        .sort_values(["negative_count", "interaction_count"], ascending=False)
        .head(limit)
    )
    return source_strength["source_subreddit"].tolist()


def apply_style() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: light;
            --app-bg: #FFFFFF;
            --panel-bg: #FFFFFF;
            --soft-bg: #F3F6FA;
            --note-bg: #FAF7F8;
            --text-main: #111827;
            --text-muted: #5F6876;
            --border-soft: #D8DEE7;
            --accent: {ACCENT};
        }}
        html,
        body,
        .stApp {{
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }}
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stBottomBlockContainer"] {{
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }}
        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
            max-width: 1360px;
            color: var(--text-main) !important;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
            color: var(--text-main) !important;
        }}
        p,
        li,
        label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: var(--text-main) !important;
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
            color: var(--text-muted) !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--accent) !important;
        }}
        .metric-card {{
            border: 1px solid var(--border-soft);
            border-radius: 8px;
            padding: 15px 16px 13px 16px;
            background: var(--panel-bg);
            min-height: 104px;
        }}
        .metric-label {{
            color: var(--text-muted) !important;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 7px;
        }}
        .metric-value {{
            color: var(--text-main) !important;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.05;
        }}
        .metric-detail {{
            color: var(--text-muted) !important;
            font-size: 0.86rem;
            margin-top: 8px;
            line-height: 1.3;
        }}
        .evidence-note {{
            border-left: 4px solid var(--accent);
            background: var(--note-bg);
            padding: 0.85rem 1rem;
            border-radius: 6px;
            color: var(--text-main) !important;
            margin: 0.4rem 0 1rem 0;
        }}
        .evidence-note * {{
            color: var(--text-main) !important;
        }}
        .small-muted {{
            color: var(--text-muted) !important;
            font-size: 0.92rem;
        }}
        section[data-testid="stSidebar"] {{
            background: #F1F5F9 !important;
            color: var(--text-main) !important;
        }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--text-main) !important;
        }}
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {{
            color: var(--text-main) !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p,
        button[data-baseweb="tab"][aria-selected="true"] span {{
            color: #D43545 !important;
        }}
        div[data-baseweb="select"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: var(--soft-bg) !important;
            border-color: var(--border-soft) !important;
            color: var(--text-main) !important;
        }}
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-testid="stSelectbox"] span,
        div[data-testid="stSelectbox"] input {{
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
        }}
        div[role="listbox"],
        ul[role="listbox"],
        [data-baseweb="popover"] {{
            background: var(--panel-bg) !important;
            color: var(--text-main) !important;
        }}
        div[role="option"],
        div[role="option"] * {{
            color: var(--text-main) !important;
        }}
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label *,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label * {{
            color: var(--text-main) !important;
        }}
        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] label *,
        div[data-testid="stSlider"] p,
        div[data-testid="stSlider"] span,
        div[data-testid="stSlider"] [data-baseweb="slider"] div {{
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
        }}
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        div[data-testid="stExpander"] div {{
            color: var(--text-main) !important;
        }}
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {{
            color: var(--text-main);
        }}
        input,
        textarea {{
            background: var(--soft-bg) !important;
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
        }}
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {{
            display: none;
        }}
        header {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status() -> None:
    st.sidebar.title("Reddit Conflict Radar")
    st.sidebar.caption("Offline research demo using exported project artifacts.")
    st.sidebar.divider()
    st.sidebar.subheader("Artifact status")
    for label, exists in data_access.artifact_status():
        st.sidebar.write(("OK " if exists else "Missing ") + label)
    st.sidebar.divider()
    st.sidebar.caption("The dashboard reads saved CSV/PNG artifacts and does not retrain models.")


def project_radar(metrics: pd.DataFrame, audit: dict[str, object]) -> None:
    best = metrics.sort_values(["test_pr_auc", "test_f1"], ascending=False).iloc[0] if not metrics.empty else None
    combined = audit.get("combined", {}) if isinstance(audit.get("combined"), dict) else {}
    raw_rows = combined.get("rows")
    raw_negative_ratio = combined.get("negative_ratio")
    kcore_rows = data_access.count_csv_rows("data/processed/phase1/phase1_kcore_filtered.csv")
    node_rows = data_access.count_csv_rows("data/processed/phase2/phase2_node_features.csv")
    pair_rows = data_access.count_csv_rows("data/processed/phase2/phase2_modeling_table.csv")

    st.markdown(
        '<div class="evidence-note">Predict whether a source-target subreddit pair becomes negative-dominant in a future time window using historical signed-network, community, balance, and text-property features.</div>',
        unsafe_allow_html=True,
    )

    cards = st.columns(6)
    with cards[0]:
        metric_card("Raw hyperlinks", fmt_int(raw_rows), "Kaggle/SNAP files")
    with cards[1]:
        ratio_text = f"{float(raw_negative_ratio) * 100:.2f}%" if raw_negative_ratio is not None else "n/a"
        metric_card("Raw negative share", ratio_text, "LINK_SENTIMENT = -1")
    with cards[2]:
        metric_card("K-core interactions", fmt_int(kcore_rows), "main modeling graph")
    with cards[3]:
        metric_card("Subreddit nodes", fmt_int(node_rows), "phase 2 node table")
    with cards[4]:
        metric_card("Pair rows", fmt_int(pair_rows), "source-target pairs")
    with cards[5]:
        value = fmt_float(best["test_pr_auc"], 3) if best is not None else "n/a"
        detail = "best test PR-AUC"
        metric_card("Best model", value, detail)

    left, right = st.columns(2)
    with left:
        show_image("label_distribution.png", "Negative links are the minority class, so PR-AUC is the headline metric.")
    with right:
        show_image("monthly_negative_ratio.png", "Negative-link share varies over time, motivating temporal evaluation.")

    if best is not None:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "feature_set": best["feature_set"],
                        "model": best["model"],
                        "test_pr_auc": best["test_pr_auc"],
                        "test_roc_auc": best["test_roc_auc"],
                        "test_f1": best["test_f1"],
                        "test_precision": best["test_precision"],
                        "test_recall": best["test_recall"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def network_explorer(edge_features: pd.DataFrame) -> None:
    st.markdown(
        '<div class="evidence-note">Explore a focused signed-network neighborhood: arrows are directed hyperlinks, red edges are negative-dominant, and node size reflects local interaction volume.</div>',
        unsafe_allow_html=True,
    )
    if edge_features.empty:
        st.warning("Edge-feature artifact is missing.")
        return

    safe_labels = st.checkbox(
        "Presentation-safe labels",
        value=True,
        help="Hide raw subreddit names that are NSFW, offensive, or awkward for a classroom demo.",
    )
    edges = presentation_safe_edges(edge_features) if safe_labels else edge_features
    focus_options = network_focus_options(edges)
    if not focus_options:
        st.warning("No edge data is available after the current label filter.")
        return

    controls = st.columns([1.2, 0.9, 0.9, 0.9])
    with controls[0]:
        default_focus = "subredditdrama" if "subredditdrama" in focus_options else focus_options[0]
        focus = st.selectbox("Focus subreddit", focus_options, index=focus_options.index(default_focus))
    with controls[1]:
        direction = st.radio("Neighborhood", ["Outgoing", "Incoming", "Both"], horizontal=True)
    with controls[2]:
        min_negative_ratio = st.slider("Min negative ratio", 0.0, 1.0, 0.50, 0.05)
    with controls[3]:
        max_edges = st.slider("Max edges", 10, 80, 45, 5)

    sort_by = st.radio(
        "Rank edges by",
        ["negative_count", "interaction_count", "negative_ratio"],
        index=0,
        horizontal=True,
        format_func=lambda value: charts.pretty_name(value),
    )

    fig, ego_edges = charts.signed_ego_network_figure(
        edges,
        focus=focus,
        direction=direction,
        min_negative_ratio=min_negative_ratio,
        max_edges=max_edges,
        sort_by=sort_by,
    )
    if ego_edges.empty:
        st.warning("No edges match the current focus and threshold. Lower the negative-ratio filter or switch focus.")
        return

    node_count = len(set(ego_edges["source_subreddit"]).union(set(ego_edges["target_subreddit"])))
    total_links = int(ego_edges["interaction_count"].sum())
    total_negatives = int(ego_edges["negative_count"].sum())
    avg_negative_ratio = float(ego_edges["negative_count"].sum() / max(ego_edges["interaction_count"].sum(), 1))
    cards = st.columns(4)
    with cards[0]:
        metric_card("Visible nodes", fmt_int(node_count), "focused subgraph")
    with cards[1]:
        metric_card("Visible edges", fmt_int(len(ego_edges)), direction.lower())
    with cards[2]:
        metric_card("Negative links", fmt_int(total_negatives), f"{fmt_int(total_links)} total links")
    with cards[3]:
        metric_card("Avg negative ratio", fmt_float(avg_negative_ratio, 3), "visible subgraph")

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

    edge_table = ego_edges[
        [
            "source_subreddit",
            "target_subreddit",
            "interaction_count",
            "positive_count",
            "negative_count",
            "negative_ratio",
            "reciprocal_edge",
        ]
    ].rename(
        columns={
            "source_subreddit": "source",
            "target_subreddit": "target",
            "interaction_count": "links",
            "positive_count": "positive",
            "negative_count": "negative",
            "negative_ratio": "negative ratio",
            "reciprocal_edge": "reciprocal",
        }
    )
    st.dataframe(edge_table, width="stretch", hide_index=True, height=260)

    with st.expander("Report figures"):
        selected = st.selectbox(
            "Static network figure",
            [
                "Negative-link backbone",
                "Community-pair negative heatmap",
                "Community negative ratio",
                "Degree distribution",
            ],
        )
        if selected == "Negative-link backbone":
            show_image("community_network_sample.png")
        elif selected == "Community-pair negative heatmap":
            show_image("community_pair_negative_heatmap.png")
        elif selected == "Community negative ratio":
            show_image("community_negative_ratio.png")
        else:
            show_image("degree_distribution.png")


def model_arena(metrics: pd.DataFrame, robustness: pd.DataFrame) -> None:
    if metrics.empty:
        st.warning("Model metrics artifact is missing.")
        return

    all_feature_sets = sorted(metrics["feature_set"].dropna().unique())
    controls = st.columns([1.0, 1.5])
    with controls[0]:
        metric = st.selectbox(
            "Ranking metric",
            list(charts.METRIC_LABELS.keys()),
            format_func=lambda value: charts.METRIC_LABELS[value],
        )
    with controls[1]:
        selected_feature_sets = st.multiselect(
            "Feature sets",
            all_feature_sets,
            default=all_feature_sets,
        )

    fig = charts.model_metric_bar(metrics, metric=metric, feature_sets=selected_feature_sets)
    st.plotly_chart(fig, width="stretch")

    table_cols = [
        "feature_set",
        "model",
        "test_pr_auc",
        "test_roc_auc",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_threshold",
        "n_features",
    ]
    st.dataframe(
        sorted_metrics(metrics, metric, selected_feature_sets)[table_cols].head(18),
        width="stretch",
        hide_index=True,
    )

    st.subheader("K-core robustness")
    st.markdown(
        '<div class="small-muted">The sampled robustness probe compares k=3, k=5, and k=10 under global and history-safe filtering.</div>',
        unsafe_allow_html=True,
    )
    if robustness.empty:
        st.warning("Robustness artifact is missing.")
    else:
        left, right = st.columns([1.25, 1.0])
        with left:
            st.plotly_chart(charts.robustness_line(robustness), width="stretch")
        with right:
            st.dataframe(
                robustness[
                    [
                        "filter_mode",
                        "k_core",
                        "test_pr_auc",
                        "test_f1",
                        "test_recall",
                        "test_pairs",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )


def threshold_simulator(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.warning("Model metrics artifact is missing.")
        return

    best = metrics.sort_values(["test_pr_auc", "test_f1"], ascending=False).iloc[0]
    default_threshold = float(best.get("test_threshold", 0.72))
    st.markdown(
        '<div class="evidence-note">Threshold tuning shows the operational trade-off: lower thresholds catch more risky pairs, while higher thresholds reduce false alarms.</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading threshold metrics..."):
        scores = data_access.load_prediction_scores(
            feature_set=str(best["feature_set"]),
            model=str(best["model"]),
            split="test",
        )
    tradeoff = charts.threshold_scan(scores) if not scores.empty else data_access.load_threshold_scan()
    if tradeoff.empty:
        st.warning("Threshold artifact is missing.")
        return

    threshold = st.slider("Decision threshold", 0.01, 0.99, default_threshold, 0.01)
    selected_metrics = (
        charts.metrics_at_threshold(scores, threshold)
        if not scores.empty
        else charts.metrics_from_threshold_scan(tradeoff, threshold)
    )

    cards = st.columns(7)
    with cards[0]:
        metric_card("Precision", fmt_float(selected_metrics["precision"], 3), "predicted positives")
    with cards[1]:
        metric_card("Recall", fmt_float(selected_metrics["recall"], 3), "actual positives found")
    with cards[2]:
        metric_card("F1", fmt_float(selected_metrics["f1"], 3), "operational balance")
    with cards[3]:
        metric_card("TP", fmt_int(selected_metrics["tp"]), "true positives")
    with cards[4]:
        metric_card("FP", fmt_int(selected_metrics["fp"]), "false positives")
    with cards[5]:
        metric_card("FN", fmt_int(selected_metrics["fn"]), "false negatives")
    with cards[6]:
        metric_card("TN", fmt_int(selected_metrics["tn"]), "true negatives")

    left, right = st.columns([1.35, 1.0])
    with left:
        st.plotly_chart(charts.threshold_curve(tradeoff, threshold), width="stretch")
    with right:
        st.plotly_chart(charts.confusion_matrix_figure(selected_metrics), width="stretch")


def case_inspector(cases: pd.DataFrame) -> None:
    if cases.empty:
        st.warning("Error-analysis artifact is missing.")
        return

    st.markdown(
        '<div class="evidence-note">Case-level inspection connects aggregate metrics to concrete source-target subreddit relationships.</div>',
        unsafe_allow_html=True,
    )
    case_types = [value for value in ["true_positive", "false_positive", "false_negative"] if value in set(cases["case_type"])]
    selected_type = st.selectbox("Case type", case_types, format_func=charts.pretty_name)
    subset = cases[cases["case_type"] == selected_type].copy()
    safe_only = st.checkbox(
        "Presentation-safe cases only",
        value=True,
        help="Hide raw subreddit names that are NSFW, offensive, or awkward for a classroom demo.",
    )
    if safe_only:
        safe_subset = presentation_safe_cases(subset)
        if safe_subset.empty:
            st.warning("No presentation-safe case is available for this case type, so raw cases are shown.")
        else:
            subset = safe_subset
    subset = subset.sort_values("score", ascending=selected_type == "false_negative")

    table_cols = [
        "source_subreddit",
        "target_subreddit",
        "interaction_count",
        "score",
        "prediction",
        "y_true",
        "negative_ratio",
        "future_negative_count",
    ]
    table = subset[table_cols].rename(
        columns={
            "source_subreddit": "source",
            "target_subreddit": "target",
            "interaction_count": "history links",
            "score": "risk score",
            "prediction": "alert",
            "y_true": "future negative",
            "negative_ratio": "prior negative ratio",
            "future_negative_count": "future negatives",
        }
    )
    table_height = min(300, 38 * (len(table) + 1))
    st.dataframe(table, width="stretch", hide_index=True, height=table_height)

    selected_index = st.selectbox(
        "Inspect one pair",
        subset.index,
        format_func=lambda idx: f"{subset.loc[idx, 'source_subreddit']} -> {subset.loc[idx, 'target_subreddit']}",
    )
    row = subset.loc[selected_index]
    alert_text = "Alert" if int(row["prediction"]) == 1 else "No alert"
    actual_text = "Future negative" if int(row["y_true"]) == 1 else "Future non-negative"
    st.markdown(
        f'<div class="evidence-note"><b>{row["source_subreddit"]} -> {row["target_subreddit"]}</b>: '
        f'model decision is <b>{alert_text}</b>; observed label is <b>{actual_text}</b>.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    with cols[0]:
        metric_card("Score", fmt_float(row["score"], 3), f"threshold {fmt_float(row['threshold'], 2)}")
    with cols[1]:
        metric_card("Decision", alert_text, charts.pretty_name(row["case_type"]))
    with cols[2]:
        metric_card("Prior negative ratio", fmt_float(row["negative_ratio"], 3), "history window")
    with cols[3]:
        metric_card("History negatives", fmt_int(row["negative_count"]), "before test window")
    with cols[4]:
        metric_card("Future negatives", fmt_int(row["future_negative_count"]), "label window")
    with cols[5]:
        metric_card("Future positives", fmt_int(row["future_positive_count"]), "label window")
    feature_summary = parse_feature_summary(row["top_contributing_features"])
    if not feature_summary.empty:
        st.dataframe(feature_summary, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Reddit Conflict Radar",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_style()
    sidebar_status()

    metrics = data_access.load_metrics()
    audit = data_access.load_dataset_audit()
    robustness = data_access.load_robustness()
    cases = data_access.load_error_cases()
    edge_features = data_access.load_edge_features()

    st.title("Reddit Conflict Radar")
    st.caption("Temporal signed-network evidence dashboard for the Social Media Data Analysis final project.")

    tabs = st.tabs(
        [
            "Project Radar",
            "Network Explorer",
            "Model Arena",
            "Threshold Simulator",
            "Case Inspector",
        ]
    )
    with tabs[0]:
        project_radar(metrics, audit)
    with tabs[1]:
        network_explorer(edge_features)
    with tabs[2]:
        model_arena(metrics, robustness)
    with tabs[3]:
        threshold_simulator(metrics)
    with tabs[4]:
        case_inspector(cases)


if __name__ == "__main__":
    main()
