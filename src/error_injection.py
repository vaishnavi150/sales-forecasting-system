"""
src/error_injection.py - builds a SMALL test dataset with KNOWN errors.

WHY
---
To prove that data validation works, we need data where we already know
exactly which rows are wrong. We take a small, clean subset of the
generated data, copy it, and deliberately break specific rows.

Every broken row is recorded in anomaly_labels.csv (the "answer key")
with its table, record ID and issue type. The validation tests then
check that every labelled problem is found.

The main clean dataset in data/raw/ is NEVER modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.snapshots import build_snapshot

SAMPLE_SNAPSHOT_DATE = pd.Timestamp(2025, 6, 30)


def _record(labels, table, record_id, issue_type, description):
    labels.append({"label_id": f"ANOM-{len(labels) + 1:03d}", "table_name": table,
                   "record_id": record_id, "issue_type": issue_type,
                   "description": description})


def build_error_sample(tables: dict[str, pd.DataFrame], seed: int = config.RANDOM_SEED,
                       size: int = 50):
    """Return (sample_tables, anomaly_labels)."""
    rng = np.random.default_rng(seed + 1)  # separate stream from the main data
    opps = tables["opportunities"]
    outcomes = tables["opportunity_outcomes"]

    # --- choose which opportunities go into the sample ----------------------
    clean_snap = build_snapshot(tables, SAMPLE_SNAPSHOT_DATE)
    closed_later = outcomes[(outcomes["final_status"] != "Open")
                            & (outcomes["close_date"] > SAMPLE_SNAPSHOT_DATE)]["opportunity_id"]
    snap_pool = clean_snap[clean_snap["opportunity_id"].isin(closed_later)]["opportunity_id"]
    # Error targets: lost or still-open deals (no bookings), not in the snapshot.
    no_booking = outcomes[outcomes["final_status"] != "Won"]["opportunity_id"]
    with_activity = set(tables["activities"]["opportunity_id"])
    inj_pool = opps[opps["opportunity_id"].isin(no_booking)
                    & ~opps["opportunity_id"].isin(clean_snap["opportunity_id"])
                    & opps["opportunity_id"].isin(with_activity)]["opportunity_id"]

    n_inj = 15
    n_snap = max(10, size - n_inj)
    inj_ids = list(rng.choice(inj_pool.to_numpy(), size=n_inj, replace=False))
    snap_ids = list(rng.choice(snap_pool.to_numpy(), size=min(n_snap, len(snap_pool)),
                               replace=False))
    chosen = sorted(inj_ids + snap_ids)

    # --- copy the clean subset (parents first, then children) ----------------
    s_opps = opps[opps["opportunity_id"].isin(chosen)].copy()
    s_accounts = tables["accounts"][tables["accounts"]["account_id"].isin(s_opps["account_id"])].copy()
    s_reps = tables["sales_reps"][tables["sales_reps"]["rep_id"].isin(s_opps["rep_id"])].copy()
    terr_ids = set(s_accounts["territory_id"]) | set(s_reps["territory_id"])
    s_terr = tables["territories"][tables["territories"]["territory_id"].isin(terr_ids)].copy()
    s_hist = tables["stage_history"][tables["stage_history"]["opportunity_id"].isin(chosen)].copy()
    s_act = tables["activities"][tables["activities"]["opportunity_id"].isin(chosen)].copy()
    s_book = tables["historical_bookings"][
        tables["historical_bookings"]["opportunity_id"].isin(chosen)].copy()
    s_snap = clean_snap[clean_snap["opportunity_id"].isin(snap_ids)].copy()

    labels: list[dict] = []
    opp_ix = {oid: s_opps.index[s_opps["opportunity_id"] == oid][0] for oid in inj_ids}
    t = inj_ids  # injection targets, in a fixed order

    # 1. Duplicate opportunity ID
    s_opps = pd.concat([s_opps, s_opps.loc[[opp_ix[t[0]]]]])
    _record(labels, "opportunities", t[0], "duplicate_opportunity_id",
            "The same opportunity row appears twice.")
    # 2. Negative deal amount
    s_opps.loc[opp_ix[t[1]], "deal_amount"] = -abs(s_opps.loc[opp_ix[t[1]], "deal_amount"])
    _record(labels, "opportunities", t[1], "negative_deal_amount", "Deal amount is negative.")
    # 3. Expected close date before creation date
    s_opps.loc[opp_ix[t[2]], "expected_close_date"] = (
        s_opps.loc[opp_ix[t[2]], "opportunity_created_date"] - pd.Timedelta(30, unit="D"))
    _record(labels, "opportunities", t[2], "close_before_creation",
            "Expected close date is 30 days before the creation date.")
    # 4. Invalid stage name
    s_opps.loc[opp_ix[t[3]], "opportunity_stage"] = "Closed Maybe"
    _record(labels, "opportunities", t[3], "invalid_stage", "Stage 'Closed Maybe' does not exist.")
    # 5. Probability outside 0..1
    s_opps.loc[opp_ix[t[4]], "current_probability"] = 1.4
    _record(labels, "opportunities", t[4], "invalid_probability", "Probability is 1.4 (140%).")
    # 6. Mixed currency
    s_opps.loc[opp_ix[t[5]], "currency"] = "USD"
    _record(labels, "opportunities", t[5], "mixed_currency",
            "Amount stored in USD while the dataset uses INR.")
    # 7. Missing rep reference
    s_opps.loc[opp_ix[t[6]], "rep_id"] = "REP-999999"
    _record(labels, "opportunities", t[6], "missing_rep_reference", "Rep REP-999999 does not exist.")
    # 8. Missing account reference
    s_opps.loc[opp_ix[t[7]], "account_id"] = "ACC-999999"
    _record(labels, "opportunities", t[7], "missing_account_reference",
            "Account ACC-999999 does not exist.")
    # 9. Missing required value
    s_opps.loc[opp_ix[t[8]], "product"] = np.nan
    _record(labels, "opportunities", t[8], "missing_required_value", "Product is empty.")
    # 10. Invalid (impossible) date. Dates are stored as text so we can
    #     write an impossible value such as month 13.
    s_opps["opportunity_created_date"] = s_opps["opportunity_created_date"].dt.strftime("%Y-%m-%d")
    s_opps.loc[opp_ix[t[9]], "opportunity_created_date"] = "2025-13-45"
    _record(labels, "opportunities", t[9], "invalid_date", "Creation date '2025-13-45' is impossible.")
    # 11. Activity logged before the opportunity existed
    act_rows = s_act.index[s_act["opportunity_id"] == t[10]]
    a_ix = act_rows[0]
    created = opps.loc[opps["opportunity_id"] == t[10], "opportunity_created_date"].iloc[0]
    s_act.loc[a_ix, "activity_date"] = created - pd.Timedelta(10, unit="D")
    _record(labels, "activities", s_act.loc[a_ix, "activity_id"], "activity_before_creation",
            "Activity dated 10 days before the opportunity was created.")

    # --- forecast snapshot problems -----------------------------------------
    snap_ix = list(s_snap.index[:5])
    key = lambda ix: f"{s_snap.loc[ix, 'snapshot_id']}|{s_snap.loc[ix, 'opportunity_id']}"
    # 12. Forecast uses final outcome data (future information leakage)
    ix = snap_ix[0]
    oc = outcomes.set_index("opportunity_id").loc[s_snap.loc[ix, "opportunity_id"]]
    final_stage = config.WON_STAGE if oc["final_status"] == "Won" else config.LOST_STAGE
    s_snap.loc[ix, "stage"] = final_stage
    s_snap.loc[ix, "stage_as_of_date"] = oc["close_date"]
    s_snap.loc[ix, "stage_probability"] = 1.0 if final_stage == config.WON_STAGE else 0.0
    s_snap.loc[ix, "weighted_amount"] = s_snap.loc[ix, "deal_amount"] * s_snap.loc[ix, "stage_probability"]
    _record(labels, "forecast_snapshots", key(ix), "future_information_leakage",
            f"Snapshot uses the final outcome '{final_stage}' recorded after the snapshot date.")
    # 13. Stage probability inconsistent with the snapshot stage
    ix = snap_ix[1]
    s_snap.loc[ix, "stage_probability"] = 0.95
    s_snap.loc[ix, "weighted_amount"] = s_snap.loc[ix, "deal_amount"] * 0.95
    _record(labels, "forecast_snapshots", key(ix), "stage_probability_inconsistent",
            "Probability 0.95 does not match the stage's standard probability.")
    # 14. Weighted amount does not equal amount x probability
    ix = snap_ix[2]
    s_snap.loc[ix, "weighted_amount"] = s_snap.loc[ix, "weighted_amount"] * 2
    _record(labels, "forecast_snapshots", key(ix), "weighted_pipeline_mismatch",
            "Weighted amount is double the correct value.")
    # 15. Currency totals mixed inside the snapshot
    ix = snap_ix[3]
    s_snap.loc[ix, "currency"] = "USD"
    _record(labels, "forecast_snapshots", key(ix), "mixed_currency",
            "Snapshot row in USD would be added to INR totals.")
    # 16. Duplicate opportunity counted twice in the same snapshot
    ix = snap_ix[4]
    dup_key = key(ix)  # compute the ID before the row is copied
    s_snap = pd.concat([s_snap, s_snap.loc[[ix]]])
    _record(labels, "forecast_snapshots", dup_key, "duplicate_snapshot_row",
            "The same opportunity is counted twice in one snapshot.")

    sample = {
        "territories": s_terr, "sales_reps": s_reps, "accounts": s_accounts,
        "stages": tables["stages"].copy(), "opportunities": s_opps,
        "stage_history": s_hist, "activities": s_act,
        "historical_bookings": s_book, "forecast_snapshots": s_snap,
    }
    sample = {k: v.reset_index(drop=True) for k, v in sample.items()}
    return sample, pd.DataFrame(labels)
