"""Feature engineering utilities for signed network data."""

from __future__ import annotations

import pandas as pd

from .phase2 import build_feature_dataset


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full phase 2 tabular feature dataset."""
    return build_feature_dataset(frame)
