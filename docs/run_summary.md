# Latest Pipeline Run Summary

Last verified run: 2026-05-14 after adding clustering/community features, additional ablations, and PR/ROC curves.

## Data Preparation

| Artifact | Rows |
|---|---:|
| Combined clean interactions | 858,488 |
| K-core filtered interactions | 708,425 |
| Phase 1 train interactions | 619,294 |
| Phase 1 validation interactions | 43,048 |
| Phase 1 test interactions | 46,083 |

## Strict Temporal Modeling Splits

| Split | Rows |
|---|---:|
| Train source-target pairs | 25,045 |
| Validation source-target pairs | 26,450 |
| Test source-target pairs | 24,185 |

## Best Test Results

The best model by test PR-AUC in the latest run is:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Test Macro-F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| graph_no_balance | Random Forest | 0.1853 | 0.7595 | 0.2602 | 0.5865 | 0.1944 | 0.3930 |

Important baseline:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 |
|---|---|---:|---:|---:|
| graph_only | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| history_only | Logistic Regression | 0.1424 | 0.6911 | 0.2343 |
| text_only | Logistic Regression | 0.1382 | 0.7111 | 0.2143 |
| hybrid | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

## Interpretation for the Report

- The best graph model without structural-balance features slightly outperforms the full hybrid model by PR-AUC in this run.
- Graph-only and graph-no-balance models perform close to or above hybrid, meaning historical signed-network structure is the strongest signal.
- Text-only is better than dummy baselines but weaker than graph/history features.
- The result supports the project thesis: signed temporal network features are useful for predicting future negative cross-community interactions.
- Structural-balance features are useful for interpretability, but the ablation indicates that they do not always improve predictive PR-AUC in the strict temporal setting.

## Generated Artifacts

- `data/processed/phase2/phase2_node_features.csv`
- `data/processed/phase2/phase2_edge_features.csv`
- `data/processed/phase2/phase2_triadic_features.csv`
- `data/processed/phase2/phase2_text_features.csv`
- `data/processed/phase2/phase2_modeling_table.csv`
- `data/processed/phase3/phase3_model_metrics.csv`
- `data/processed/phase3/phase3_feature_importance.csv`
- `data/processed/phase3/phase3_prediction_scores.csv`
- `reports/figures/*.png`
