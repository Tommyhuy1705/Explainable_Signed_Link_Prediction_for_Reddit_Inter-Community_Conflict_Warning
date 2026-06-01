# Latest Pipeline Run Summary

Last verified run: 2026-05-31 using the bundled Python runtime after adding Kaggle/SNAP provenance, reproducibility scripts, robustness probes, threshold analysis, error analysis, and a real presentation deck.

## Data Preparation

| Artifact | Rows |
| --- | ---: |
| Combined clean interactions | 858,488 |
| K-core filtered interactions | 708,425 |
| Phase 1 train interactions | 619,294 |
| Phase 1 validation interactions | 43,048 |
| Phase 1 test interactions | 46,083 |

## Strict Temporal Modeling Splits

The modeling target is a future negative-dominant source-target relationship: `negative_label = 1` when future negative hyperlinks outnumber future positive/neutral hyperlinks in the label window.

| Split | Rows |
| --- | ---: |
| Train source-target pairs | 25,045 |
| Validation source-target pairs | 26,450 |
| Test source-target pairs | 24,185 |

## Best Test Results

The best model by test PR-AUC in the latest run is:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Test Macro-F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid | Logistic Regression | 0.1840 | 0.7569 | 0.2700 | 0.5935 | 0.2050 | 0.3954 |

Important baseline:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 |
| --- | --- | ---: | ---: | ---: |
| graph_only | Historical negative ratio | 0.1237 | 0.6328 | 0.2327 |
| history_only | Logistic Regression | 0.1424 | 0.6905 | 0.2341 |
| text_only | Logistic Regression | 0.1398 | 0.7118 | 0.2202 |
| hybrid | Dummy prior | 0.0698 | 0.5000 | 0.1304 |

Model families included in the verified run:

- Dummy most frequent.
- Dummy prior.
- Historical negative-ratio heuristic.
- Logistic Regression.
- Random Forest.
- XGBoost.
- LightGBM.

The exported `phase3_model_metrics.csv` contains 41 rows across six feature sets.

## Dataset Audit and Provenance

The project uses the Kaggle mirror `Signed Graphs`, file family `soc-RedditHyperlinks`, with the original academic source documented as Stanford SNAP / Kumar et al. The reproducible audit script validates the raw files before modeling:

```powershell
python scripts/audit_dataset.py --raw-dir data/raw --json-out data/processed/dataset_audit.json
```

Latest audit results:

| Raw File | Rows | Negative Links | Negative Ratio |
| --- | ---: | ---: | ---: |
| `soc-redditHyperlinks-body.tsv` | 286,561 | 21,070 | 0.0735 |
| `soc-redditHyperlinks-title.tsv` | 571,927 | 61,140 | 0.1069 |
| Combined | 858,488 | 82,210 | 0.0958 |

The audit confirms required columns, valid labels in `{-1, 1}`, parseable timestamps, and exactly 86 `PROPERTIES` values per row.

## Interpretation for the Report

- The best model is a hybrid Logistic Regression model, indicating that text-property features add some useful signal when combined with temporal graph/history features.
- Graph-only and graph-no-balance models remain close to the hybrid results, meaning historical signed-network structure is still the strongest and most stable signal family.
- Text-only is better than dummy baselines but weaker than graph/history features.
- The result supports the project thesis: signed temporal network features are useful for predicting future negative-dominant cross-community relationships.
- Structural-balance features are useful for interpretability, but the ablation indicates that their predictive contribution is small in the strict temporal setting.

## Robustness and Explainability Additions

The grade-maximization pass adds three reproducible analysis artifacts:

- `robustness_metrics.csv`: sampled k-core sensitivity probe for `k=3`, `k=5`, and `k=10`, including a history-safe cohort setting.
- `error_analysis_cases.csv`: local contribution examples for true positives, false positives, and false negatives.
- `threshold_tradeoff.csv` and `threshold_tradeoff.png`: precision/recall/F1 tradeoff for the best hybrid logistic model.

The sampled robustness probe shows that PR-AUC stays above the dummy prevalence baseline under all tested k-core settings, supporting the conclusion that the result is not a one-off artifact of `k=5`.

## Generated Artifacts

- `data/processed/phase1/phase1_kcore_filtered.csv`
- `data/processed/phase1/phase1_train.csv`
- `data/processed/phase1/phase1_validation.csv`
- `data/processed/phase1/phase1_test.csv`
- `data/processed/phase2/phase2_node_features.csv`
- `data/processed/phase2/phase2_edge_features.csv`
- `data/processed/phase2/phase2_triadic_features.csv`
- `data/processed/phase2/phase2_text_features.csv`
- `data/processed/phase2/phase2_modeling_table.csv`
- `data/processed/phase3/phase3_model_metrics.csv`
- `data/processed/phase3/phase3_feature_importance.csv`
- `data/processed/phase3/phase3_prediction_scores.csv`
- `data/processed/phase3/error_analysis_cases.csv`
- `data/processed/phase3/robustness_metrics.csv`
- `reports/figures/label_distribution.png`
- `reports/figures/monthly_negative_ratio.png`
- `reports/figures/top_negative_sources.png`
- `reports/figures/top_negative_targets.png`
- `reports/figures/degree_distribution.png`
- `reports/figures/community_negative_ratio.png`
- `reports/figures/community_network_sample.png`
- `reports/figures/community_pair_negative_heatmap.png`
- `reports/figures/model_comparison_pr_auc.png`
- `reports/figures/precision_recall_curve.png`
- `reports/figures/roc_curve.png`
- `reports/figures/best_confusion_matrix.png`
- `reports/figures/feature_importance_top20.png`
- `reports/figures/threshold_tradeoff.png`
- `reports/figures/robustness_kcore_pr_auc.png`
- `docs/final_presentation.pptx`
