"""Phase 3 modeling and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .phase2 import build_feature_components, assemble_feature_dataset, load_phase1_filtered

try:  # pragma: no cover - optional dependency
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


TARGET_COLUMN = "negative_label"
TEXT_FEATURE_PREFIX = "text_property_"
IGNORED_COLUMNS = {
    "source_subreddit",
    "target_subreddit",
    "first_timestamp",
    "last_timestamp",
    "negative_label",
    "future_positive_count",
    "future_negative_count",
    "future_interaction_count",
    "future_start",
    "future_end",
    "source_community_id",
    "target_community_id",
}

PAIR_HISTORY_FEATURES = {
    "interaction_count",
    "positive_count",
    "negative_count",
    "sentiment_balance",
    "negative_ratio",
    "reciprocal_edge",
}

BALANCE_FEATURES = {
    "common_neighbors",
    "balance_+++",
    "balance_++-",
    "balance_+--",
    "balance_---",
}


@dataclass
class SplitData:
    """Container for split modeling tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_phase1_interactions(path: str | Path) -> pd.DataFrame:
    """Load cleaned interactions from phase 1 for strict temporal modeling."""
    return load_phase1_filtered(path)


def _aggregate_future_labels(frame: pd.DataFrame, future_start: pd.Timestamp, future_end: pd.Timestamp) -> pd.DataFrame:
    """Build labels from a disjoint future window."""
    future_frame = frame[(frame["timestamp"] > future_start) & (frame["timestamp"] <= future_end)].copy()
    if future_frame.empty:
        return pd.DataFrame(
            columns=[
                "source_subreddit",
                "target_subreddit",
                "future_positive_count",
                "future_negative_count",
                "future_interaction_count",
                TARGET_COLUMN,
                "future_start",
                "future_end",
            ]
        )

    grouped = future_frame.groupby(["source_subreddit", "target_subreddit"], dropna=False)["link_sentiment"]
    labels = grouped.agg(
        future_interaction_count="size",
        future_positive_count=lambda values: int((values == 1).sum()),
        future_negative_count=lambda values: int((values == -1).sum()),
    ).reset_index()
    labels[TARGET_COLUMN] = (labels["future_negative_count"] > labels["future_positive_count"]).astype(int)
    labels["future_start"] = future_start
    labels["future_end"] = future_end
    return labels


def _build_history_features(frame: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    """Compute feature table from interactions available up to history_end."""
    history_frame = frame[frame["timestamp"] <= history_end].copy()
    if history_frame.empty:
        return pd.DataFrame(columns=["source_subreddit", "target_subreddit"])

    _, node_features, edge_features, triadic_features, text_features = build_feature_components(history_frame)
    feature_table = assemble_feature_dataset(node_features, edge_features, triadic_features, text_features)
    if TARGET_COLUMN in feature_table.columns:
        feature_table = feature_table.drop(columns=[TARGET_COLUMN])
    return feature_table


def build_strict_temporal_splits(frame: pd.DataFrame) -> SplitData:
    """Create train/validation/test splits with strict history-label separation."""
    windows = {
        "train": (pd.Timestamp("2015-12-31 23:59:59"), pd.Timestamp("2016-06-30 23:59:59")),
        "validation": (pd.Timestamp("2016-06-30 23:59:59"), pd.Timestamp("2016-12-31 23:59:59")),
        "test": (pd.Timestamp("2016-12-31 23:59:59"), pd.Timestamp("2017-04-30 23:59:59")),
    }

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, (history_end, label_end) in windows.items():
        features = _build_history_features(frame, history_end)
        labels = _aggregate_future_labels(frame, history_end, label_end)
        merged = features.merge(labels, on=["source_subreddit", "target_subreddit"], how="inner")
        split_frames[split_name] = merged.reset_index(drop=True)

    return SplitData(
        train=split_frames["train"],
        validation=split_frames["validation"],
        test=split_frames["test"],
    )


def build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare the numeric feature matrix and target vector."""
    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)
    feature_frame = cleaned.drop(columns=[column for column in IGNORED_COLUMNS if column in cleaned.columns])
    feature_frame = feature_frame.select_dtypes(include=["number", "bool"]).copy()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(0).astype("float64")
    feature_columns = list(feature_frame.columns)
    return feature_frame, y, feature_columns


def _apply_smote_if_available(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally apply SMOTE when the dependency is available and the class distribution allows it."""
    if SMOTE is None:
        return x_train, y_train
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        return x_train, y_train
    sampler = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    return pd.DataFrame(x_resampled, columns=x_train.columns), pd.Series(y_resampled)


def _safe_probability_scores(model, x_frame: pd.DataFrame) -> np.ndarray:
    """Get probability-like scores for evaluation."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_frame)
        if probabilities.shape[1] == 1:
            classes = getattr(model, "classes_", np.array([0]))
            return np.ones(len(x_frame)) if int(classes[0]) == 1 else np.zeros(len(x_frame))
        return probabilities[:, 1]
    decision = model.decision_function(x_frame)
    return 1 / (1 + np.exp(-decision))


def _evaluate_scores(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute threshold-free and threshold-dependent binary metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, probabilities) if y_true.nunique() > 1 else float("nan"),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions) if y_true.nunique() > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, predictions),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def _tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Select a decision threshold on validation data by maximizing F1."""
    if y_true.nunique() < 2:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold


def _evaluate_model(model, x_frame: pd.DataFrame, y_true: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate a fitted model with a supplied decision threshold."""
    probabilities = _safe_probability_scores(model, x_frame)
    return _evaluate_scores(y_true, probabilities, threshold)


def _build_score_rows(
    split_name: str,
    feature_set: str,
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    """Create row-wise prediction scores for PR/ROC curve plotting."""
    predictions = (probabilities >= threshold).astype(int)
    return [
        {
            "split": split_name,
            "feature_set": feature_set,
            "model": model_name,
            "y_true": int(label),
            "score": float(score),
            "prediction": int(prediction),
            "threshold": float(threshold),
        }
        for label, score, prediction in zip(y_true, probabilities, predictions)
    ]


def _is_text_feature(column: str) -> bool:
    """Identify features derived from the 86 SNAP text-property vector."""
    return (
        column.startswith(TEXT_FEATURE_PREFIX)
        or column == "text_feature_count"
        or column.startswith("link_location_")
    )


def _is_balance_feature(column: str) -> bool:
    """Identify local structural-balance features."""
    return column in BALANCE_FEATURES or column.startswith("balance_")


def _build_feature_sets(feature_columns: list[str]) -> dict[str, list[str]]:
    """Create ablation feature sets for paper-style model comparison."""
    text_columns = [column for column in feature_columns if _is_text_feature(column)]
    graph_columns = [column for column in feature_columns if column not in text_columns]
    feature_sets = {"hybrid": feature_columns}
    if graph_columns:
        feature_sets["graph_only"] = graph_columns
    if text_columns:
        feature_sets["text_only"] = text_columns
    no_balance_columns = [column for column in feature_columns if not _is_balance_feature(column)]
    if len(no_balance_columns) < len(feature_columns):
        feature_sets["hybrid_no_balance"] = no_balance_columns
    graph_no_balance_columns = [column for column in graph_columns if not _is_balance_feature(column)]
    if graph_columns and len(graph_no_balance_columns) < len(graph_columns):
        feature_sets["graph_no_balance"] = graph_no_balance_columns
    history_columns = [column for column in feature_columns if column in PAIR_HISTORY_FEATURES]
    if history_columns:
        feature_sets["history_only"] = history_columns
    return feature_sets


def _fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    """Train baseline and advanced models for phase 3."""
    x_resampled, y_resampled = _apply_smote_if_available(x_train, y_train)
    models: dict[str, object] = {}

    models["dummy_most_frequent"] = DummyClassifier(strategy="most_frequent")
    models["dummy_prior"] = DummyClassifier(strategy="prior")
    models["logistic_regression"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            ),
        ]
    )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )

    fitted_models: dict[str, object] = {}
    for name, model in models.items():
        if name.startswith("dummy_"):
            model.fit(x_train, y_train)
        else:
            model.fit(x_resampled, y_resampled)
        fitted_models[name] = model
    return fitted_models


def run_phase3_pipeline(modeling_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitData]:
    """Run strict temporal phase 3 and return metrics, importances, scores, and splits."""
    splits = build_strict_temporal_splits(modeling_table)

    x_train, y_train, feature_columns = build_feature_matrix(splits.train)
    x_validation, y_validation, _ = build_feature_matrix(splits.validation)
    x_test, y_test, _ = build_feature_matrix(splits.test)

    metric_rows = []
    importance_rows = []
    score_rows = []

    for feature_set, selected_columns in _build_feature_sets(feature_columns).items():
        selected_train = x_train[selected_columns]
        selected_validation = x_validation.reindex(columns=selected_columns, fill_value=0)
        selected_test = x_test.reindex(columns=selected_columns, fill_value=0)

        fitted_models = _fit_models(selected_train, y_train)

        if "negative_ratio" in selected_columns:
            validation_scores = selected_validation["negative_ratio"].to_numpy()
            test_scores = selected_test["negative_ratio"].to_numpy()
            threshold = _tune_threshold(y_validation, validation_scores)
            validation_metrics = _evaluate_scores(y_validation, validation_scores, threshold)
            test_metrics = _evaluate_scores(y_test, test_scores, threshold)
            score_rows.extend(
                _build_score_rows("validation", feature_set, "historical_negative_ratio", y_validation, validation_scores, threshold)
            )
            score_rows.extend(
                _build_score_rows("test", feature_set, "historical_negative_ratio", y_test, test_scores, threshold)
            )
            metric_rows.append(
                {
                    "feature_set": feature_set,
                    "model": "historical_negative_ratio",
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    "n_features": 1,
                }
            )

        for name, model in fitted_models.items():
            validation_probabilities = _safe_probability_scores(model, selected_validation)
            threshold = _tune_threshold(y_validation, validation_probabilities)
            validation_metrics = _evaluate_scores(y_validation, validation_probabilities, threshold)
            test_probabilities = _safe_probability_scores(model, selected_test)
            test_metrics = _evaluate_scores(y_test, test_probabilities, threshold)
            score_rows.extend(
                _build_score_rows("validation", feature_set, name, y_validation, validation_probabilities, threshold)
            )
            score_rows.extend(
                _build_score_rows("test", feature_set, name, y_test, test_probabilities, threshold)
            )
            metric_rows.append(
                {
                    "feature_set": feature_set,
                    "model": name,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    "n_features": len(selected_columns),
                }
            )

            if hasattr(model, "named_steps"):
                model_object = model.named_steps["model"]
                importance_values = np.abs(model_object.coef_).ravel()
            elif hasattr(model, "feature_importances_"):
                importance_values = np.asarray(model.feature_importances_)
            else:
                importance_values = np.zeros(len(selected_columns))

            for feature_name, importance in zip(selected_columns, importance_values):
                importance_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": name,
                        "feature": feature_name,
                        "importance": float(importance),
                    }
                )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["test_pr_auc", "test_f1"], ascending=False).reset_index(drop=True)
    importance_frame = pd.DataFrame(importance_rows).sort_values(["feature_set", "model", "importance"], ascending=[True, True, False])
    score_frame = pd.DataFrame(score_rows)
    return metrics_frame, importance_frame, score_frame, splits


def export_phase3_outputs(
    metrics_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    output_dir: str | Path,
    score_frame: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist the phase 3 outputs to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "phase3_model_metrics.csv"
    importance_path = output_path / "phase3_feature_importance.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    importance_frame.to_csv(importance_path, index=False)
    paths = {"metrics": metrics_path, "importance": importance_path}
    if score_frame is not None:
        scores_path = output_path / "phase3_prediction_scores.csv"
        score_frame.to_csv(scores_path, index=False)
        paths["scores"] = scores_path
    return paths
