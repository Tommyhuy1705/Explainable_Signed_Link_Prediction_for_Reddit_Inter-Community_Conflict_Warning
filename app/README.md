# Reddit Conflict Radar Dashboard

This Streamlit app is an offline research demo for the final project. It does
not retrain models. It reads exported artifacts from `data/processed/` and
`reports/figures/` so the classroom demo stays fast and reproducible.

## Setup

Use the Python 3.13 runtime available on this machine:

```powershell
& "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pip.exe install -r requirements-app.txt
```

If optional ML wheels such as XGBoost or LightGBM are unavailable for the local
Python version, the dashboard can still run after installing the app
requirements because it only reads existing CSV/PNG artifacts.

## Run

```powershell
.\.venv\Scripts\streamlit.exe run app\app.py
```

The app opens at:

```text
http://localhost:8501
```

## Demo Flow

1. `Project Radar`: dataset scale, negative-class rarity, and best result.
2. `Network Explorer`: interactive signed ego-network with edge filters.
3. `Model Arena`: model comparison, ablations, and k-core robustness.
4. `Threshold Simulator`: precision/recall/F1 trade-off with a live threshold.
5. `Case Inspector`: true-positive, false-positive, and false-negative examples.
