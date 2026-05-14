# Phase 3: Modeling and Evaluation

## Checklist

- [x] Use strict history-label temporal separation.
- [x] Train dummy and historical-ratio baselines.
- [x] Train Logistic Regression, Random Forest, XGBoost, and LightGBM when dependencies are available.
- [x] Compare graph-only, text-only, and hybrid feature sets.
- [x] Compare additional ablations: history-only, graph without balance, and hybrid without balance.
- [x] Handle class imbalance with class weights and optional SMOTE.
- [x] Tune decision thresholds on validation data.
- [x] Report PR-AUC, ROC-AUC, F1, Macro-F1, precision, recall, balanced accuracy, and confusion counts.
- [x] Export feature-importance tables.

## Strict Anti-Leakage Protocol

Features are computed only from interactions before a history cutoff. Labels are computed from a disjoint future window.

| Split | History Window | Label Window |
|---|---|---|
| Train | <= 2015-12-31 | 2016-01-01 to 2016-06-30 |
| Validation | <= 2016-06-30 | 2016-07-01 to 2016-12-31 |
| Test | <= 2016-12-31 | 2017-01-01 to 2017-04-30 |

## Deliverables

- `notebooks/04_modeling_and_evaluation.ipynb`
- `data/processed/phase3/phase3_model_metrics.csv`
- `data/processed/phase3/phase3_feature_importance.csv`
- `data/processed/phase3/phase3_prediction_scores.csv`
- Report figures in `reports/figures/`

## Reporting Guidance

Accuracy should not be the headline metric because negative interactions are rare. The main comparison should use PR-AUC and F1 for the negative class, with ROC-AUC as a supporting metric.
