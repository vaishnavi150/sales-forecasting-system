"""
src/evaluation.py - measures how good a forecast was.

All metrics compare a FORECAST with the ACTUAL result, after the
forecast period has finished.

    error = forecast - actual
    MAE   = average size of the error (ignoring + or -)
    RMSE  = like MAE but punishes big misses more (square, average, root)
    WAPE  = total absolute error / total actual  (a percentage)
            -> "on average we missed by X% of what was actually booked"
    Bias  = total error / total actual
            -> positive = we forecast TOO HIGH, negative = TOO LOW

ZERO ACTUALS: dividing by zero is impossible. WAPE and bias add up all
periods first, so one zero period does not break them. If the total
actual is zero, we return NaN ("not available") instead of a fake number.
Per-period percentage error is also NaN when that period's actual is 0.

For individual deals we also use:
    Brier score  = average of (predicted probability - 0/1 outcome)^2 (lower is better)
    ROC AUC      = how well the model ranks winners above losers (0.5 = random, 1 = perfect)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _arrays(forecast, actual):
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    if f.shape != a.shape:
        raise ValueError("forecast and actual must have the same length")
    if len(f) == 0:
        raise ValueError("cannot evaluate an empty forecast")
    return f, a


def mae(forecast, actual) -> float:
    f, a = _arrays(forecast, actual)
    return float(np.mean(np.abs(f - a)))


def rmse(forecast, actual) -> float:
    f, a = _arrays(forecast, actual)
    return float(np.sqrt(np.mean((f - a) ** 2)))


def wape(forecast, actual) -> float:
    """Weighted Absolute Percentage Error. NaN if total actual is zero."""
    f, a = _arrays(forecast, actual)
    total = np.sum(np.abs(a))
    return float("nan") if total == 0 else float(np.sum(np.abs(f - a)) / total)


def bias(forecast, actual) -> float:
    """Relative bias. Positive = over-forecast. NaN if total actual is zero."""
    f, a = _arrays(forecast, actual)
    total = np.sum(a)
    return float("nan") if total == 0 else float(np.sum(f - a) / total)


def percentage_errors(forecast, actual) -> np.ndarray:
    """Per-period (forecast - actual) / actual; NaN where actual is zero."""
    f, a = _arrays(forecast, actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a == 0, np.nan, (f - a) / a)


def aggregate_metrics(forecast, actual) -> dict:
    return {"MAE": mae(forecast, actual), "RMSE": rmse(forecast, actual),
            "WAPE": wape(forecast, actual), "Bias": bias(forecast, actual),
            "periods": int(len(np.asarray(forecast)))}


def deal_level_metrics(probabilities, outcomes) -> dict:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, dtype=int)
    both_classes = len(np.unique(y)) == 2
    return {"Brier": float(brier_score_loss(y, p)),
            "LogLoss": float(log_loss(y, p, labels=[0, 1])),
            "ROC_AUC": float(roc_auc_score(y, p)) if both_classes else float("nan"),
            "deals": int(len(y))}


def compare_methods(period_results: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    period_results: one row per period with columns 'actual' and one column
    per method holding that method's forecast. Only periods where EVERY
    method has a forecast are used, so the comparison is fair.
    """
    common = period_results.dropna(subset=methods)
    rows = []
    for m in methods:
        row = {"method": m, **aggregate_metrics(common[m], common["actual"])}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
