# Models Directory

This folder is reserved for optional trained model artifacts.

The current course-project workflow retrains all models from `notebooks/04_modeling_and_evaluation.ipynb` and stores evaluation outputs in:

- `data/processed/phase3/phase3_model_metrics.csv`
- `data/processed/phase3/phase3_feature_importance.csv`
- `data/processed/phase3/phase3_prediction_scores.csv`

No binary model artifact is required for the current report because the project focuses on reproducible evaluation, model comparison, and interpretation. If an inference demo is added later, this folder can store files such as:

- `best_model.joblib`
- `best_model_features.json`
- `model_card.md`
