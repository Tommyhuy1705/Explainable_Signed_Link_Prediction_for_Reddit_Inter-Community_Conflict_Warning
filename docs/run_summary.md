# Latest Pipeline Run Summary

Last verified run: 2026-05-14.

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
| hybrid | Logistic Regression | 0.1813 | 0.7573 | 0.2685 | 0.5942 | 0.2076 | 0.3800 |

Important baseline:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 |
|---|---|---:|---:|---:|
| graph_only | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| hybrid | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

## Interpretation for the Report

- The hybrid model improves over text-only, graph-only, historical-ratio, and dummy baselines by PR-AUC.
- Graph-only performs close to hybrid, meaning historical signed-network structure is the strongest signal.
- Text-only is better than dummy baselines but weaker than graph/history features.
- The result supports the project thesis: signed temporal network features are useful for predicting future negative cross-community interactions.

## Generated Artifacts

- `data/processed/phase2/phase2_node_features.csv`
- `data/processed/phase2/phase2_edge_features.csv`
- `data/processed/phase2/phase2_triadic_features.csv`
- `data/processed/phase2/phase2_text_features.csv`
- `data/processed/phase2/phase2_modeling_table.csv`
- `data/processed/phase3/phase3_model_metrics.csv`
- `data/processed/phase3/phase3_feature_importance.csv`
- `reports/figures/*.png`
