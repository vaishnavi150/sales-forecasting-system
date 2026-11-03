"""
src/feature_engineering.py - turns a pipeline snapshot into model inputs.

A "feature" is one piece of information the model can learn from, such
as "how many days since the last call". Every feature here is calculated
using ONLY records dated on or before the snapshot date:

    * stage history     -> only events with change_date  <= snapshot date
    * activities        -> only activities with activity_date <= snapshot date
    * win rates         -> only deals that had CLOSED by the snapshot date

The ground-truth outcome file is never read in this module.

SMOOTHING (why we don't use raw win rates)
------------------------------------------
A new rep with 1 win out of 1 deal has a "100% win rate", which is
misleading. We blend each rate with an overall average:

    smoothed = (wins + prior_rate * strength) / (closed_deals + strength)

With few deals the result stays near the average; with many deals it
moves toward the rep's own rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

STAGE_ORDER = {s.name: s.order for s in config.STAGES}
DEFAULT_PROB = {s.name: s.default_probability for s in config.STAGES}

NUMERIC_FEATURES = [
    "stage_order", "opportunity_age_days", "days_until_expected_close",
    "days_in_current_stage", "log_deal_amount", "stage_conversion_rate",
    "rep_win_rate", "rep_closed_deals", "account_win_rate", "num_calls",
    "num_emails", "num_meetings", "num_followups", "activities_last_30d",
    "days_since_last_activity", "close_date_pushes", "overall_win_rate",
]
CATEGORICAL_FEATURES = ["stage", "region", "industry", "account_size", "product"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _known_history(stage_history: pd.DataFrame, as_of) -> pd.DataFrame:
    return stage_history[stage_history["change_date"] <= pd.Timestamp(as_of)]


def closed_deals_as_of(stage_history: pd.DataFrame, as_of) -> pd.DataFrame:
    """Deals that were closed on or before `as_of`: opportunity_id, won (0/1), close_date."""
    h = _known_history(stage_history, as_of)
    closed = h[h["stage"].isin([config.WON_STAGE, config.LOST_STAGE])]
    closed = closed.sort_values("change_date").drop_duplicates("opportunity_id", keep="first")
    return pd.DataFrame({"opportunity_id": closed["opportunity_id"],
                         "won": (closed["stage"] == config.WON_STAGE).astype(int),
                         "close_date": closed["change_date"]})


def stage_conversion_rates(stage_history: pd.DataFrame, as_of, strength: float = 20.0) -> dict:
    """
    For each open stage: of the deals that REACHED this stage and have
    since closed (by `as_of`), what share were won?  Smoothed toward the
    stage's default probability.
    """
    h = _known_history(stage_history, as_of)
    closed = closed_deals_as_of(stage_history, as_of)
    reached = (h[h["stage"].isin(config.OPEN_STAGE_NAMES)][["opportunity_id", "stage"]]
               .drop_duplicates().merge(closed, on="opportunity_id"))
    rates = {}
    for stage in config.OPEN_STAGE_NAMES:
        grp = reached[reached["stage"] == stage]
        prior = DEFAULT_PROB[stage]
        rates[stage] = float((grp["won"].sum() + prior * strength) / (len(grp) + strength))
    return rates


def _smoothed_rate(closed: pd.DataFrame, key: str, prior: float, strength: float) -> pd.DataFrame:
    g = closed.groupby(key)["won"].agg(["sum", "count"])
    g["rate"] = (g["sum"] + prior * strength) / (g["count"] + strength)
    return g


def build_features(tables: dict[str, pd.DataFrame], snapshot: pd.DataFrame,
                   snapshot_date) -> pd.DataFrame:
    """
    Return one row per snapshot opportunity with all FEATURE_COLUMNS
    (plus identifier columns). Uses only information known on snapshot_date.
    """
    d = pd.Timestamp(snapshot_date)
    feats = snapshot[["snapshot_id", "snapshot_date", "opportunity_id", "rep_id", "account_id",
                      "stage", "deal_amount", "expected_close_date",
                      "opportunity_created_date", "product"]].copy()
    if feats.empty:
        for col in FEATURE_COLUMNS:
            if col not in feats:
                feats[col] = pd.Series(dtype=float)
        return feats
    history = tables["stage_history"]

    # --- basic timing / amount features ----------------------------------
    feats["stage_order"] = feats["stage"].map(STAGE_ORDER)
    feats["opportunity_age_days"] = (d - feats["opportunity_created_date"]).dt.days
    feats["days_until_expected_close"] = (feats["expected_close_date"] - d).dt.days
    feats["log_deal_amount"] = np.log1p(feats["deal_amount"])

    known = _known_history(history, d)
    last_stage_change = (known[known["event_type"] == "stage_change"]
                         .groupby("opportunity_id")["change_date"].max())
    feats["days_in_current_stage"] = (d - feats["opportunity_id"].map(last_stage_change)).dt.days
    pushes = known[known["event_type"] == "close_date_change"].groupby("opportunity_id").size()
    feats["close_date_pushes"] = feats["opportunity_id"].map(pushes).fillna(0).astype(int)

    # --- historical conversion and win rates (closed by snapshot date) ----
    rates = stage_conversion_rates(history, d)
    feats["stage_conversion_rate"] = feats["stage"].map(rates)

    closed = closed_deals_as_of(history, d)
    overall = float(closed["won"].mean()) if len(closed) else 0.3
    feats["overall_win_rate"] = overall
    opp_info = tables["opportunities"][["opportunity_id", "rep_id", "account_id"]]
    closed = closed.merge(opp_info, on="opportunity_id", how="left")
    rep_rates = _smoothed_rate(closed, "rep_id", overall, strength=10)
    feats["rep_win_rate"] = feats["rep_id"].map(rep_rates["rate"]).fillna(overall)
    feats["rep_closed_deals"] = feats["rep_id"].map(rep_rates["count"]).fillna(0).astype(int)
    acc_rates = _smoothed_rate(closed, "account_id", overall, strength=5)
    feats["account_win_rate"] = feats["account_id"].map(acc_rates["rate"]).fillna(overall)

    # --- activity features (activities on or before snapshot date) --------
    acts = tables["activities"]
    acts = acts[(acts["activity_date"] <= d) & acts["opportunity_id"].isin(feats["opportunity_id"])]
    by_type = acts.pivot_table(index="opportunity_id", columns="activity_type",
                               values="activity_count", aggfunc="sum", fill_value=0)
    for col, act_type in [("num_calls", "Call"), ("num_emails", "Email"),
                          ("num_meetings", "Meeting"), ("num_followups", "Follow-up")]:
        series = by_type[act_type] if act_type in by_type else pd.Series(dtype=float)
        feats[col] = feats["opportunity_id"].map(series).fillna(0).astype(int)
    recent = acts[acts["activity_date"] > d - pd.Timedelta(30, unit="D")]
    feats["activities_last_30d"] = (feats["opportunity_id"]
                                    .map(recent.groupby("opportunity_id")["activity_count"].sum())
                                    .fillna(0).astype(int))
    last_act = acts.groupby("opportunity_id")["activity_date"].max()
    feats["last_activity_date"] = feats["opportunity_id"].map(last_act)
    # No activity yet -> count days since the deal was created.
    feats["days_since_last_activity"] = (
        d - feats["last_activity_date"].fillna(feats["opportunity_created_date"])).dt.days

    # --- account attributes (static) ----------------------------------------
    acc = tables["accounts"][["account_id", "region", "industry", "account_size"]]
    feats = feats.merge(acc, on="account_id", how="left")
    return feats.reset_index(drop=True)
