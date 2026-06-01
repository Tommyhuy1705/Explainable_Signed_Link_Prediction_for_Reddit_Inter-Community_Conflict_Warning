# Phase 4: Report and Presentation

## Checklist

- [x] Complete a paper-style report draft in `docs/final_report.md`.
- [x] Export and polish all report figures in `reports/figures/`.
- [x] Include model-comparison and ablation tables.
- [x] Document Kaggle mirror provenance and original SNAP/Kumar et al. source in `docs/data_provenance.md`.
- [x] Add dataset audit script and reproducible run wrapper.
- [x] Add robustness results for k-core sensitivity.
- [x] Add threshold tradeoff and local error-analysis artifacts.
- [x] Add limitations about proxy labels, historical data, and k-core filtering.
- [x] Prepare a presentation outline in `docs/presentation_outline.md`.
- [x] Prepare Q&A answers for leakage, class imbalance, baselines, and real-world validity in `docs/presentation_qna.md`.
- [x] Create a real 12-slide deck in `docs/final_presentation.pptx`.

## Required Figures

- Label distribution.
- Monthly negative-link ratio.
- Top negative source subreddits.
- Top negative target subreddits.
- Directed degree distribution.
- Community-level negative ratio.
- Readable subreddit network sample colored by detected community.
- Community-pair negative-ratio heatmap.
- Model comparison by test PR-AUC.
- Precision-recall curve.
- ROC curve.
- Best-model confusion matrix.
- Top feature-importance plot.
- Threshold tradeoff plot.
- K-core robustness PR-AUC plot.

## Suggested Report Structure

1. Abstract
2. Introduction
3. Related Work
4. Dataset and Provenance
5. Problem Formulation
6. Methodology
7. Experiments
8. Results
9. Robustness and Error Analysis
10. Discussion
11. Threats to Validity
12. Rubric Mapping
13. Conclusion and Future Work
14. References

## Key Message

The final report should frame the project as negative-dominant cross-community relationship prediction. The stronger conflict-warning framing belongs in the future-work section.

The submission package should emphasize that the dataset requirement is satisfied through the Kaggle `Signed Graphs` mirror, while the scholarly provenance remains Stanford SNAP and Kumar et al.
