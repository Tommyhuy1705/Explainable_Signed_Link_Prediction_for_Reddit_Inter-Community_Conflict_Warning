# Phase 1: Data Preparation

## Checklist

- [x] Load and verify the two SNAP Reddit hyperlink files.
- [x] Standardize schema and data types.
- [x] Preserve `dataset_source` so title/body links can be compared.
- [x] Remove missing core fields, self-loops, and exact duplicate interactions.
- [x] Sort interactions by timestamp.
- [x] Apply optional k-core filtering for a denser modeling graph.
- [x] Create chronological train/validation/test data extracts for EDA.

## Deliverables

- `notebooks/01_data_exploration.ipynb`
- `data/interim/phase1_combined_clean.csv`
- `data/processed/phase1/phase1_kcore_filtered.csv`
- `data/processed/phase1/phase1_train.csv`
- `data/processed/phase1/phase1_validation.csv`
- `data/processed/phase1/phase1_test.csv`

## Notes for the Report

K-core filtering should be described as a practical course-project restriction to reduce sparsity. The stricter modeling protocol in Phase 3 still computes features only from historical windows.
