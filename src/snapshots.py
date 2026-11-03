"""
src/snapshots.py - "photos" of the pipeline on a past date.

WHY THIS FILE IS IMPORTANT (the leakage guard)
----------------------------------------------
To test a forecast honestly, we must pretend we are standing on the
snapshot date and know NOTHING that happened afterwards.

build_snapshot() rebuilds the pipeline using only stage-history events
dated ON OR BEFORE the snapshot date. It does not read the current
opportunity stage (which may already show "Closed Won") and it does
not read the ground-truth outcome file.

label_snapshot() attaches the real outcomes, but only if the whole
forecast period has already finished. Otherwise it raises an error,
because those outcomes would not have been known yet.
"""

from __future__ import annotations

import pandas as pd

import config

SNAPSHOT_COLUMNS = [
    "snapshot_id", "snapshot_date", "opportunity_id", "account_id", "rep_id",
    "product", "opportunity_created_date", "stage", "stage_probability",
    "stage_as_of_date", "expected_close_date", "deal_amount", "currency",
    "weighted_amount",
]


def horizon_end(snapshot_date) -> pd.Timestamp:
    """The last day of the quarter AFTER the snapshot date's quarter.
    Example: 2025-03-31 -> 2025-06-30 (we forecast the next quarter)."""
    d = pd.Timestamp(snapshot_date)
    current_q_end = d + pd.offsets.QuarterEnd(0)
    return (current_q_end + pd.offsets.QuarterEnd(1)).normalize()


def snapshot_id(snapshot_date) -> str:
    return config.FORECAST_ID_FORMAT.format(
        snapshot_date=pd.Timestamp(snapshot_date).strftime("%Y-%m-%d"), n=1)


def build_snapshot(tables: dict[str, pd.DataFrame], snapshot_date) -> pd.DataFrame:
    """
    Open opportunities as they looked on `snapshot_date`.

    Uses: opportunities (only fields that never change: IDs, product,
    creation date, amount, currency) and stage_history up to the date.
    """
    d = pd.Timestamp(snapshot_date)
    opps = tables["opportunities"]
    history = tables["stage_history"]

    # 1. Only events that had happened by the snapshot date.
    known = history[history["change_date"] <= d]
    # 2. The latest known event per opportunity = its state on that day.
    known = known.sort_values(["opportunity_id", "change_date", "stage_history_id"])
    latest = known.groupby("opportunity_id").tail(1)
    # 3. Keep deals that were still open on that day.
    latest = latest[latest["stage"].isin(config.OPEN_STAGE_NAMES)]

    static_cols = ["opportunity_id", "account_id", "rep_id", "product",
                   "opportunity_created_date", "deal_amount", "currency"]
    snap = latest.merge(opps[static_cols], on="opportunity_id", how="inner")
    snap = snap[snap["opportunity_created_date"] <= d]

    snap = snap.rename(columns={"change_date": "stage_as_of_date"})
    snap["snapshot_id"] = snapshot_id(d)
    snap["snapshot_date"] = d
    snap["weighted_amount"] = snap["deal_amount"] * snap["stage_probability"]
    return snap[SNAPSHOT_COLUMNS].sort_values("opportunity_id").reset_index(drop=True)


def label_snapshot(snapshot: pd.DataFrame, outcomes: pd.DataFrame,
                   data_available_until=config.DATA_END_DATE) -> pd.DataFrame:
    """
    Add the real result for each snapshot row:
        won_in_horizon = 1 if the deal was WON inside the forecast quarter
        actual_booking = deal_amount if won_in_horizon else 0

    Raises ValueError if the forecast quarter is not finished yet,
    because using unfinished outcomes would be "looking into the future".
    """
    if snapshot.empty:
        out = snapshot.copy()
        out["won_in_horizon"] = pd.Series(dtype=int)
        out["actual_booking"] = pd.Series(dtype=float)
        return out
    d = pd.Timestamp(snapshot["snapshot_date"].iloc[0])
    h_end = horizon_end(d)
    if h_end > pd.Timestamp(data_available_until):
        raise ValueError(
            f"Outcomes for the period ending {h_end.date()} are not available yet "
            f"(data available until {pd.Timestamp(data_available_until).date()}).")
    o = outcomes[["opportunity_id", "final_status", "close_date"]]
    out = snapshot.merge(o, on="opportunity_id", how="left")
    won = ((out["final_status"] == "Won") & (out["close_date"] > d)
           & (out["close_date"] <= h_end))
    out["won_in_horizon"] = won.astype(int)
    out["actual_booking"] = out["deal_amount"] * out["won_in_horizon"]
    return out.drop(columns=["final_status", "close_date"])


def build_all_snapshots(tables, dates=None) -> pd.DataFrame:
    """Snapshots for every configured date, stacked into one table."""
    dates = dates or config.SNAPSHOT_DATES
    return pd.concat([build_snapshot(tables, d) for d in dates], ignore_index=True)
