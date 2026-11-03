"""
src/forecasting.py - baseline forecasts, machine-learning models and backtesting.

THE QUESTION WE ANSWER
----------------------
Standing on a snapshot date (say 31 March), for each OPEN deal:
"What is the chance this deal is WON during the NEXT quarter (Apr-Jun)?"

    expected booking = deal amount x that probability
    forecast         = sum of expected bookings over all open deals

FOUR METHODS
------------
1. crm_stage_probability  (simple baseline)
   The standard probability of the deal's stage (Proposal = 50%, ...),
   counted only if the expected close date falls on or before the end
   of the forecast quarter. This is what many CRMs show by default.
2. historical_stage_rate  (better baseline)
   Same idea, but uses the REAL historical win rate of the stage,
   calculated from deals that had closed before the snapshot date.
3. logistic_regression    (machine learning)
4. random_forest          (machine learning)
   Both learn from past snapshots: features -> did the deal win in
   the next quarter (1) or not (0)?

BACKTESTING (time-based evaluation)
-----------------------------------
For each snapshot date D, the ML models are trained ONLY on earlier
snapshots whose forecast quarter had already ENDED by D (so their
outcomes were known on D). They then forecast D's next quarter and
we compare with what actually happened. No random shuffling of time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src import evaluation
from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from src.snapshots import build_snapshot, horizon_end, label_snapshot

METHODS = ["crm_stage_probability", "historical_stage_rate",
           "logistic_regression", "random_forest"]
ML_METHODS = ["logistic_regression", "random_forest"]
METHOD_LABELS = {
    "crm_stage_probability": "Baseline: CRM stage probability",
    "historical_stage_rate": "Baseline: historical stage conversion",
    "logistic_regression": "ML: Logistic Regression",
    "random_forest": "ML: Random Forest",
}


# ======================================================================
# Core formula
# ======================================================================

def expected_booking(deal_amount, probability):
    """Expected Booking = Deal Amount x Probability of Winning."""
    amount = np.asarray(deal_amount, dtype=float)
    p = np.asarray(probability, dtype=float)
    if np.any(amount < 0):
        raise ValueError("Deal amounts must not be negative.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")
    return amount * p


def quarter_of(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


# ======================================================================
# Baselines
# ======================================================================

def baseline_probabilities(features: pd.DataFrame, snapshot_date, column: str) -> np.ndarray:
    """
    Probability from a column (e.g. stage_conversion_rate), but only for
    deals expected to close by the end of the forecast quarter
    (overdue deals are included, because they are still expected soon).
    """
    h_end = horizon_end(snapshot_date)
    in_horizon = features["expected_close_date"] <= h_end
    return np.where(in_horizon, features[column].to_numpy(dtype=float), 0.0)


def add_baseline_columns(features: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    out = features.copy()
    out["stage_probability"] = out["stage"].map(
        {s.name: s.default_probability for s in config.STAGES})
    out["p_crm_stage_probability"] = baseline_probabilities(out, snapshot_date, "stage_probability")
    out["p_historical_stage_rate"] = baseline_probabilities(out, snapshot_date, "stage_conversion_rate")
    return out


# ======================================================================
# Machine-learning models
# ======================================================================

def make_model(kind: str, seed: int = config.RANDOM_SEED) -> Pipeline:
    """
    A scikit-learn Pipeline = preprocessing + model in one object.
      * numbers are scaled (mean 0, spread 1) so no feature dominates;
      * categories (stage, region, ...) become 0/1 columns ("one-hot").
    """
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    if kind == "logistic_regression":
        model = LogisticRegression(max_iter=2000, C=1.0)
    elif kind == "random_forest":
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                       max_features="sqrt", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([("prep", pre), ("model", model)])


def train_model(kind: str, train: pd.DataFrame, seed: int = config.RANDOM_SEED,
                weight_by_amount: bool = True) -> Pipeline:
    """
    Fit a model on labelled snapshot rows.

    weight_by_amount: we forecast booking VALUE, so a mistake on a
    1-crore deal matters more than one on a 1-lakh deal. Giving each
    training row a weight proportional to its deal amount tells the
    model to pay more attention to big deals. (In our backtest this
    lowered the ML models' WAPE; see README.)
    """
    if train.empty:
        raise ValueError("No training data available.")
    if train["won_in_horizon"].nunique() < 2:
        raise ValueError("Training data must contain both won and not-won deals.")
    model = make_model(kind, seed)
    weights = None
    if weight_by_amount:
        weights = (train["deal_amount"] / train["deal_amount"].mean()).to_numpy()
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["won_in_horizon"],
              model__sample_weight=weights)
    return model


def predict_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.array([])
    return model.predict_proba(features[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]


def training_rows(frames: dict, as_of) -> pd.DataFrame:
    """
    Rows whose labels were KNOWN on `as_of`: only snapshots whose
    forecast quarter ended on or before `as_of`.
    """
    usable = [f for d, f in frames.items()
              if horizon_end(d) <= pd.Timestamp(as_of) and "won_in_horizon" in f]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


# ======================================================================
# Snapshot preparation, backtest and live forecast
# ======================================================================

def prepare_frames(tables: dict, dates, data_available_until=config.DATA_END_DATE) -> dict:
    """Features (+ labels when the forecast quarter is complete) for each date."""
    frames = {}
    for d in dates:
        d = pd.Timestamp(d)
        snap = build_snapshot(tables, d)
        feats = add_baseline_columns(build_features(tables, snap, d), d)
        if horizon_end(d) <= pd.Timestamp(data_available_until):
            labels = label_snapshot(snap, tables["opportunity_outcomes"], data_available_until)
            feats = feats.merge(labels[["opportunity_id", "won_in_horizon", "actual_booking"]],
                                on="opportunity_id", how="left")
        frames[d] = feats
    return frames


def backtest(frames: dict, seed: int = config.RANDOM_SEED):
    """
    Returns (deal_predictions, period_results).

    deal_predictions: one row per (snapshot, open deal) with each method's probability.
    period_results:   one row per snapshot with each method's total forecast and the actual.
    """
    deal_rows, period_rows = [], []
    for d, feats in frames.items():
        if "won_in_horizon" not in feats:
            continue  # outcomes not known yet -> cannot evaluate
        preds = feats.copy()
        train = training_rows(frames, d)
        for kind in ML_METHODS:
            if not train.empty and train["won_in_horizon"].nunique() == 2:
                model = train_model(kind, train, seed)
                preds[f"p_{kind}"] = predict_probability(model, preds)
            else:
                preds[f"p_{kind}"] = np.nan
        preds["training_rows"] = len(train)
        deal_rows.append(preds)

        row = {"snapshot_date": d, "forecast_quarter": quarter_of(horizon_end(d)),
               "open_deals": len(preds), "training_rows": len(train),
               "actual": float(preds["actual_booking"].sum())}
        for m in METHODS:
            p = preds[f"p_{m}"]
            row[m] = float((preds["deal_amount"] * p).sum()) if p.notna().all() else np.nan
        period_rows.append(row)
    return pd.concat(deal_rows, ignore_index=True), pd.DataFrame(period_rows)


def choose_best_method(period_results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Pick the method with the LOWEST WAPE over the backtest periods where
    every method had a forecast. The data decides - not our expectations.
    """
    comparison = evaluation.compare_methods(period_results, METHODS)
    return str(comparison.iloc[0]["method"]), comparison


def choose_risk_model(deal_predictions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    For DEAL-LEVEL risk flags we need good individual probabilities, so we
    pick the method with the lowest Brier score (rows where all methods exist).
    """
    common = deal_predictions.dropna(subset=[f"p_{m}" for m in METHODS])
    rows = [{"method": m, **evaluation.deal_level_metrics(common[f"p_{m}"], common["won_in_horizon"])}
            for m in METHODS]
    table = pd.DataFrame(rows).sort_values("Brier").reset_index(drop=True)
    return str(table.iloc[0]["method"]), table


def segment_results(deal_predictions: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Forecast vs actual per snapshot and segment (e.g. region), for every method."""
    dp = deal_predictions.copy()
    agg = {"actual": ("actual_booking", "sum")}
    for m in METHODS:
        dp[f"eb_{m}"] = dp["deal_amount"] * dp[f"p_{m}"]
        agg[m] = (f"eb_{m}", lambda s: s.sum(min_count=len(s)))
    return dp.groupby(["snapshot_date", segment]).agg(**agg).reset_index()


def live_forecast(tables: dict, frames: dict, live_date=config.LIVE_SNAPSHOT_DATE,
                  seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Forecast the quarter AFTER the last data date (outcomes unknown).
    ML models are trained on every snapshot whose outcomes are known.
    """
    d = pd.Timestamp(live_date)
    snap = build_snapshot(tables, d)
    feats = add_baseline_columns(build_features(tables, snap, d), d)
    train = training_rows(frames, d)
    for kind in ML_METHODS:
        model = train_model(kind, train, seed)
        feats[f"p_{kind}"] = predict_probability(model, feats)
    feats["training_rows"] = len(train)
    return feats
