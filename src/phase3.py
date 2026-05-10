"""Phase 3 modeling and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
TIME_COLUMN = "last_timestamp"
IGNORED_COLUMNS = {
    "source_subreddit",
    "target_subreddit",
    "first_timestamp",
    "last_timestamp",
    "negative_label",
    "interaction_count",
    "positive_count",
    "negative_count",
    "sentiment_balance",
    "negative_ratio",
}


@dataclass
class SplitData:
    """Container for split modeling tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_phase2_modeling_table(path: str | Path) -> pd.DataFrame:
    """Load the phase 2 modeling table and parse timestamps."""
    frame = pd.read_csv(path)
    for column in ["first_timestamp", "last_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def temporal_split_modeling_table(
    frame: pd.DataFrame,
    train_end: str = "2016-12-31 23:59:59",
    validation_end: str = "2017-02-28 23:59:59",
) -> SplitData:
    """Split the modeling table by the last interaction timestamp."""
    ordered = frame.sort_values(TIME_COLUMN).copy()
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    train_frame = ordered[ordered[TIME_COLUMN] <= train_cutoff].copy()
    validation_frame = ordered[(ordered[TIME_COLUMN] > train_cutoff) & (ordered[TIME_COLUMN] <= validation_cutoff)].copy()
    test_frame = ordered[ordered[TIME_COLUMN] > validation_cutoff].copy()
    return SplitData(
        train=train_frame.reset_index(drop=True),
        validation=validation_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )


def build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare the numeric feature matrix and target vector."""
    cleaned = frame.copy()
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)
    feature_frame = cleaned.drop(columns=[column for column in IGNORED_COLUMNS if column in cleaned.columns])
    feature_frame = feature_frame.select_dtypes(include=["number"]).copy()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(0)
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
        return model.predict_proba(x_frame)[:, 1]
    decision = model.decision_function(x_frame)
    return 1 / (1 + np.exp(-decision))


def _evaluate_model(model, x_frame: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """Compute ROC-AUC, F1, and PR-AUC for a fitted model."""
    probabilities = _safe_probability_scores(model, x_frame)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else float("nan"),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities) if y_true.nunique() > 1 else float("nan"),
    }
    return metrics


def _fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    """Train baseline and advanced models for phase 3."""
    x_resampled, y_resampled = _apply_smote_if_available(x_train, y_train)
    models: dict[str, object] = {}

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
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )

    fitted_models: dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_resampled, y_resampled)
        fitted_models[name] = model
    return fitted_models


def run_phase3_pipeline(modeling_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, SplitData]:
    """Run the full phase 3 pipeline and return metrics plus feature importances."""
    splits = temporal_split_modeling_table(modeling_table)

    x_train, y_train, feature_columns = build_feature_matrix(splits.train)
    x_validation, y_validation, _ = build_feature_matrix(splits.validation)
    x_test, y_test, _ = build_feature_matrix(splits.test)

    fitted_models = _fit_models(x_train, y_train)

    metric_rows = []
    importance_rows = []
    for name, model in fitted_models.items():
        validation_metrics = _evaluate_model(model, x_validation, y_validation)
        test_metrics = _evaluate_model(model, x_test, y_test)
        metric_rows.append(
            {
                "model": name,
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_f1": validation_metrics["f1"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
            }
        )

        if hasattr(model, "named_steps"):
            model_object = model.named_steps["model"]
            importance_values = np.abs(model_object.coef_).ravel()
        elif hasattr(model, "feature_importances_"):
            importance_values = np.asarray(model.feature_importances_)
        else:
            importance_values = np.zeros(len(feature_columns))

        for feature_name, importance in zip(feature_columns, importance_values):
            importance_rows.append(
                {
                    "model": name,
                    "feature": feature_name,
                    "importance": float(importance),
                }
            )

    metrics_frame = pd.DataFrame(metric_rows).sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
    importance_frame = pd.DataFrame(importance_rows).sort_values(["model", "importance"], ascending=[True, False])
    return metrics_frame, importance_frame, splits


def export_phase3_outputs(metrics_frame: pd.DataFrame, importance_frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Persist the phase 3 outputs to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "phase3_model_metrics.csv"
    importance_path = output_path / "phase3_feature_importance.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    importance_frame.to_csv(importance_path, index=False)
    return {"metrics": metrics_path, "importance": importance_path}