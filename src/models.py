"""Model configuration and training helpers."""

from sklearn.linear_model import LogisticRegression


def build_baseline_model() -> LogisticRegression:
    """Return a simple baseline classifier."""
    return LogisticRegression(max_iter=1000)
