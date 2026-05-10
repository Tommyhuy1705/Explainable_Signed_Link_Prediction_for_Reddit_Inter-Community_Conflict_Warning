# Phase 3: Modeling & Evaluation

## Checklist
- [x] Huấn luyện baseline (Logistic Regression, Random Forest).
- [x] Xử lý mất cân bằng lớp (SMOTE hoặc class weight).
- [x] Huấn luyện mô hình nâng cao (XGBoost/LightGBM).
- [x] Đánh giá bằng ROC-AUC, F1, PR-AUC.
- [x] Phân tích feature importance / SHAP.

## Deliverables
- [x] Notebook `04_modeling_and_evaluation.ipynb`.
- [x] `data/processed/phase3/phase3_model_metrics.csv`.
- [x] `data/processed/phase3/phase3_feature_importance.csv`.

## Strict Anti-Leakage Protocol
- Features chỉ được xây từ lịch sử trước mốc cutoff (`history_end`).
- Nhãn được tạo từ cửa sổ tương lai rời nhau (`future_start` -> `future_end`).
- Không dùng cột tạo nhãn tương lai trong feature matrix.
- Split strict hiện tại:
  - Train: history <= 2015-12-31, label window = 2016-01-01 đến 2016-06-30.
  - Validation: history <= 2016-06-30, label window = 2016-07-01 đến 2016-12-31.
  - Test: history <= 2016-12-31, label window = 2017-01-01 đến 2017-04-30.
