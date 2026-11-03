"""
src/data_validation.py - checks the data for problems and writes a report.

HOW IT WORKS
------------
Each check looks at one kind of problem and returns a list of "issues".
Every issue says:
    table, record_id     -> exactly which row is affected (traceability)
    issue_type           -> short name, e.g. "negative_deal_amount"
    severity             -> "error" (data is wrong) or "warning" (suspicious)
    message              -> what is wrong, in plain words
    recommendation       -> how to fix it

validate_tables() runs every check on a dictionary of tables.
build_summary() turns the issue list into the data-quality report.

TIP: load CSVs WITHOUT converting dates first (parse_dates=False), so we can
tell the difference between an EMPTY date and an IMPOSSIBLE date like
"2025-13-45". The checks convert dates themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

ERROR, WARNING = "error", "warning"

RECOMMENDATIONS = {
    "duplicate_opportunity_id": "Keep one row per opportunity ID; merge or delete the extra copy.",
    "duplicate_primary_key": "Make the ID unique; investigate how the duplicate was created.",
    "missing_required_value": "Fill the empty field from the source system or exclude the row.",
    "invalid_date": "Correct the date to a real calendar date in YYYY-MM-DD format.",
    "negative_deal_amount": "Check the sign; deal amounts must be zero or positive.",
    "zero_deal_amount": "Confirm the deal value; a zero amount adds nothing to the forecast.",
    "close_before_creation": "Set the expected close date on or after the creation date.",
    "invalid_stage": "Map the value to one of the allowed stages.",
    "invalid_probability": "Probabilities must be between 0 and 1 (0% to 100%).",
    "stage_probability_inconsistent": "Use the standard probability for the stage, or document the override.",
    "mixed_currency": f"Convert the amount to {config.BASE_CURRENCY} before adding it to totals.",
    "missing_rep_reference": "Point the record to an existing sales rep ID.",
    "missing_account_reference": "Point the record to an existing account ID.",
    "missing_opportunity_reference": "Point the record to an existing opportunity ID.",
    "missing_territory_reference": "Point the record to an existing territory ID.",
    "invalid_category": "Replace the value with one from the allowed list.",
    "event_before_creation": "Stage events cannot happen before the opportunity was created.",
    "activity_before_creation": "Correct the activity date; it is earlier than the opportunity.",
    "activity_after_close": "Check whether the activity belongs to a different, still-open deal.",
    "invalid_activity_count": "Activity counts must be whole numbers of 1 or more.",
    "invalid_lifecycle_transition": "No stage changes are allowed after a deal is closed.",
    "stage_history_mismatch": "Current stage must equal the latest stage in the history.",
    "closed_flag_inconsistent": "is_closed must be 1 exactly for Closed Won / Closed Lost.",
    "booking_reconciliation_error": "Bookings must match won opportunities one-to-one, with equal amounts.",
    "invalid_quota": "Quota amounts must be positive.",
    "future_information_leakage": "Rebuild the snapshot using only events on or before the snapshot date.",
    "duplicate_snapshot_row": "Each opportunity may appear only once per snapshot.",
    "weighted_pipeline_mismatch": "Recalculate weighted amount = deal amount x probability.",
    "closed_deal_in_snapshot": "Snapshots should contain only deals open on the snapshot date.",
}

PRIMARY_KEYS = {
    "territories": "territory_id", "sales_reps": "rep_id", "accounts": "account_id",
    "stages": "stage_id", "opportunities": "opportunity_id",
    "stage_history": "stage_history_id", "activities": "activity_id",
    "quotas": "quota_id", "historical_bookings": "booking_id",
}

REQUIRED_COLUMNS = {
    "territories": ["territory_id", "territory_name", "region"],
    "sales_reps": ["rep_id", "rep_name", "region", "territory_id", "team", "joining_date"],
    "accounts": ["account_id", "account_name", "industry", "region", "territory_id", "account_size"],
    "opportunities": ["opportunity_id", "account_id", "rep_id", "product",
                      "opportunity_created_date", "expected_close_date", "opportunity_stage",
                      "deal_amount", "currency", "current_probability"],
    "stage_history": ["stage_history_id", "opportunity_id", "change_date", "event_type", "stage"],
    "activities": ["activity_id", "opportunity_id", "rep_id", "activity_date",
                   "activity_type", "activity_count"],
    "historical_bookings": ["booking_id", "opportunity_id", "booking_date", "booking_amount"],
    "quotas": ["quota_id", "rep_id", "quota_quarter", "quota_amount"],
    "forecast_snapshots": ["snapshot_id", "snapshot_date", "opportunity_id", "stage",
                           "stage_probability", "deal_amount", "currency"],
}

DATE_COLUMNS = {
    "sales_reps": ["joining_date"],
    "opportunities": ["opportunity_created_date", "expected_close_date"],
    "stage_history": ["change_date", "expected_close_date"],
    "activities": ["activity_date"],
    "historical_bookings": ["booking_date"],
    "forecast_snapshots": ["snapshot_date", "stage_as_of_date", "expected_close_date",
                           "opportunity_created_date"],
}

DEFAULT_PROB = {s.name: s.default_probability for s in config.STAGES}
CLOSED_STAGES = {config.WON_STAGE, config.LOST_STAGE}


# ======================================================================
# Helpers
# ======================================================================

def _issues(table, ids, issue_type, severity, message) -> list[dict]:
    """Turn a list of record IDs into issue dictionaries."""
    return [{"table_name": table, "record_id": str(r), "issue_type": issue_type,
             "severity": severity, "message": message,
             "recommendation": RECOMMENDATIONS.get(issue_type, "")}
            for r in ids]


def _to_dates(series: pd.Series) -> pd.Series:
    """Convert to dates; impossible values become NaT (Not a Time)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")


def _record_ids(df: pd.DataFrame, table: str) -> pd.Series:
    if table == "forecast_snapshots":
        return df["snapshot_id"].astype(str) + "|" + df["opportunity_id"].astype(str)
    key = PRIMARY_KEYS.get(table)
    return df[key].astype(str) if key in df else df.index.astype(str)


# ======================================================================
# Generic checks (any table)
# ======================================================================

def check_primary_keys(table, df) -> list[dict]:
    key = PRIMARY_KEYS.get(table)
    if key not in df:
        return []
    dup = df[key][df[key].duplicated(keep="first") & df[key].notna()]
    issue = "duplicate_opportunity_id" if table == "opportunities" else "duplicate_primary_key"
    return _issues(table, dup.unique(), issue, ERROR, f"Duplicate {key}.")


def check_required_values(table, df) -> list[dict]:
    out = []
    ids = _record_ids(df, table)
    for col in REQUIRED_COLUMNS.get(table, []):
        if col not in df:
            out += _issues(table, ["<whole table>"], "missing_required_value", ERROR,
                           f"Column '{col}' is missing.")
            continue
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        out += _issues(table, ids[empty], "missing_required_value", ERROR,
                       f"'{col}' is empty.")
    return out


def check_dates(table, df) -> list[dict]:
    """Values that are present but are not real dates."""
    out = []
    ids = _record_ids(df, table)
    for col in DATE_COLUMNS.get(table, []):
        if col not in df or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        present = df[col].notna() & (df[col].astype(str).str.strip() != "")
        bad = present & _to_dates(df[col]).isna()
        out += _issues(table, ids[bad], "invalid_date", ERROR,
                       f"'{col}' is not a valid date.")
    return out


# ======================================================================
# Table-specific checks
# ======================================================================

def check_master_tables(t) -> list[dict]:
    out = []
    terr_ids = set(t["territories"]["territory_id"]) if "territories" in t else None
    for table, cols in [("sales_reps", {"region": config.REGIONS, "team": config.TEAMS}),
                        ("accounts", {"region": config.REGIONS, "industry": config.INDUSTRIES,
                                      "account_size": config.ACCOUNT_SIZES}),
                        ("territories", {"region": config.REGIONS})]:
        if table not in t:
            continue
        df, ids = t[table], _record_ids(t[table], table)
        for col, allowed in cols.items():
            bad = df[col].notna() & ~df[col].isin(allowed)
            out += _issues(table, ids[bad], "invalid_category", ERROR,
                           f"'{col}' has a value outside the allowed list.")
        if terr_ids is not None and table != "territories":
            bad = df["territory_id"].notna() & ~df["territory_id"].isin(terr_ids)
            out += _issues(table, ids[bad], "missing_territory_reference", ERROR,
                           "Territory ID not found.")
    return out


def check_opportunities(t) -> list[dict]:
    if "opportunities" not in t:
        return []
    df = t["opportunities"]
    ids = df["opportunity_id"].astype(str)
    out = []
    created = _to_dates(df["opportunity_created_date"])
    close = _to_dates(df["expected_close_date"])
    amount = pd.to_numeric(df["deal_amount"], errors="coerce")
    prob = pd.to_numeric(df["current_probability"], errors="coerce")
    stage = df["opportunity_stage"]

    out += _issues("opportunities", ids[amount < 0], "negative_deal_amount", ERROR,
                   "Deal amount is negative.")
    out += _issues("opportunities", ids[amount == 0], "zero_deal_amount", WARNING,
                   "Deal amount is zero.")
    out += _issues("opportunities", ids[close < created], "close_before_creation", ERROR,
                   "Expected close date is before the creation date.")
    bad_stage = stage.notna() & ~stage.isin(config.STAGE_NAMES)
    out += _issues("opportunities", ids[bad_stage], "invalid_stage", ERROR,
                   "Stage is not in the allowed list.")
    bad_prob = prob.notna() & ((prob < 0) | (prob > 1))
    out += _issues("opportunities", ids[bad_prob], "invalid_probability", ERROR,
                   "Probability is outside 0-1.")
    expected = stage.map(DEFAULT_PROB)
    inconsistent = expected.notna() & ~bad_prob & prob.notna() & ((prob - expected).abs() > 1e-9)
    out += _issues("opportunities", ids[inconsistent], "stage_probability_inconsistent", WARNING,
                   "Probability does not match the stage's standard probability.")
    bad_cur = df["currency"].notna() & ~df["currency"].isin(config.ALLOWED_CURRENCIES)
    out += _issues("opportunities", ids[bad_cur], "mixed_currency", ERROR,
                   f"Currency is not {config.BASE_CURRENCY}.")
    if "sales_reps" in t:
        bad = df["rep_id"].notna() & ~df["rep_id"].isin(t["sales_reps"]["rep_id"])
        out += _issues("opportunities", ids[bad], "missing_rep_reference", ERROR,
                       "Sales rep ID not found.")
    if "accounts" in t:
        bad = df["account_id"].notna() & ~df["account_id"].isin(t["accounts"]["account_id"])
        out += _issues("opportunities", ids[bad], "missing_account_reference", ERROR,
                       "Account ID not found.")
    if "is_closed" in df:
        should = stage.isin(CLOSED_STAGES).astype(int)
        bad = stage.isin(config.STAGE_NAMES) & (pd.to_numeric(df["is_closed"], errors="coerce") != should)
        out += _issues("opportunities", ids[bad], "closed_flag_inconsistent", WARNING,
                       "is_closed does not match the stage.")
    # Current stage must equal the latest stage in the history (lifecycle check).
    if "stage_history" in t and len(t["stage_history"]):
        h = t["stage_history"].assign(_d=_to_dates(t["stage_history"]["change_date"]))
        last = (h.sort_values(["opportunity_id", "_d", "stage_history_id"])
                 .groupby("opportunity_id")["stage"].last())
        hist_stage = df["opportunity_id"].map(last)
        bad = hist_stage.notna() & (hist_stage != stage) & ~bad_stage
        out += _issues("opportunities", ids[bad], "stage_history_mismatch", WARNING,
                       "Current stage differs from the latest stage-history event.")
    return out


def check_stage_history(t) -> list[dict]:
    if "stage_history" not in t:
        return []
    h = t["stage_history"].copy()
    ids = h["stage_history_id"].astype(str)
    out = []
    bad_stage = h["stage"].notna() & ~h["stage"].isin(config.STAGE_NAMES)
    out += _issues("stage_history", ids[bad_stage], "invalid_stage", ERROR,
                   "Stage is not in the allowed list.")
    if "opportunities" in t:
        opps = t["opportunities"].drop_duplicates("opportunity_id")
        known = h["opportunity_id"].isin(opps["opportunity_id"])
        out += _issues("stage_history", ids[~known], "missing_opportunity_reference", ERROR,
                       "Opportunity ID not found.")
        created = h["opportunity_id"].map(
            opps.set_index("opportunity_id")["opportunity_created_date"].pipe(_to_dates))
        before = _to_dates(h["change_date"]) < created
        out += _issues("stage_history", ids[before], "event_before_creation", ERROR,
                       "Event is dated before the opportunity was created.")
    # No events after a deal is closed.
    h["_d"] = _to_dates(h["change_date"])
    h = h.sort_values(["opportunity_id", "_d", "stage_history_id"])
    # Count closed events seen so far per deal; if one happened BEFORE this
    # row, then this row is an event after closing.
    is_closed_event = h["stage"].isin(CLOSED_STAGES).astype(int)
    closed_so_far = is_closed_event.groupby(h["opportunity_id"]).cumsum()
    closed_before = (closed_so_far - is_closed_event) > 0
    out += _issues("stage_history", h.loc[closed_before, "stage_history_id"],
                   "invalid_lifecycle_transition", ERROR, "Event recorded after the deal was closed.")
    return out


def check_activities(t) -> list[dict]:
    if "activities" not in t:
        return []
    a = t["activities"]
    ids = a["activity_id"].astype(str)
    out = []
    bad_type = a["activity_type"].notna() & ~a["activity_type"].isin(config.ACTIVITY_TYPES)
    out += _issues("activities", ids[bad_type], "invalid_category", ERROR,
                   "Activity type is not in the allowed list.")
    count = pd.to_numeric(a["activity_count"], errors="coerce")
    bad_count = count.notna() & ((count < 1) | (count != np.floor(count)))
    out += _issues("activities", ids[bad_count], "invalid_activity_count", ERROR,
                   "Activity count must be a whole number of at least 1.")
    if "sales_reps" in t:
        bad = a["rep_id"].notna() & ~a["rep_id"].isin(t["sales_reps"]["rep_id"])
        out += _issues("activities", ids[bad], "missing_rep_reference", ERROR, "Sales rep ID not found.")
    if "opportunities" in t:
        opps = t["opportunities"].drop_duplicates("opportunity_id").set_index("opportunity_id")
        known = a["opportunity_id"].isin(opps.index)
        out += _issues("activities", ids[~known], "missing_opportunity_reference", ERROR,
                       "Opportunity ID not found.")
        act_date = _to_dates(a["activity_date"])
        created = a["opportunity_id"].map(_to_dates(opps["opportunity_created_date"]))
        out += _issues("activities", ids[act_date < created], "activity_before_creation", ERROR,
                       "Activity is dated before the opportunity was created.")
        if "stage_history" in t:
            h = t["stage_history"]
            closes = (h[h["stage"].isin(CLOSED_STAGES)]
                      .assign(_d=lambda x: _to_dates(x["change_date"]))
                      .groupby("opportunity_id")["_d"].min())
            after = act_date > a["opportunity_id"].map(closes)
            out += _issues("activities", ids[after], "activity_after_close", WARNING,
                           "Activity is dated after the deal was closed.")
    return out


def check_bookings(t) -> list[dict]:
    """Bookings must reconcile one-to-one with won opportunities."""
    if "historical_bookings" not in t or "opportunities" not in t:
        return []
    b = t["historical_bookings"]
    opps = t["opportunities"].drop_duplicates("opportunity_id").set_index("opportunity_id")
    out = []
    ids = b["booking_id"].astype(str)
    opp_stage = b["opportunity_id"].map(opps["opportunity_stage"])
    opp_amount = b["opportunity_id"].map(pd.to_numeric(opps["deal_amount"], errors="coerce"))
    bad = (opp_stage != config.WON_STAGE) | ((b["booking_amount"] - opp_amount).abs() > 0.5)
    out += _issues("historical_bookings", ids[bad], "booking_reconciliation_error", ERROR,
                   "Booking does not match a won opportunity with the same amount.")
    won = opps.index[opps["opportunity_stage"] == config.WON_STAGE]
    missing = sorted(set(won) - set(b["opportunity_id"]))
    out += _issues("opportunities", missing, "booking_reconciliation_error", ERROR,
                   "Won opportunity has no booking record.")
    return out


def check_quotas(t) -> list[dict]:
    if "quotas" not in t:
        return []
    q = t["quotas"]
    out = _issues("quotas", q.loc[q["quota_amount"] <= 0, "quota_id"], "invalid_quota", ERROR,
                  "Quota amount must be positive.")
    if "sales_reps" in t:
        bad = ~q["rep_id"].isin(t["sales_reps"]["rep_id"])
        out += _issues("quotas", q.loc[bad, "quota_id"], "missing_rep_reference", ERROR,
                       "Sales rep ID not found.")
    return out


def check_snapshots(t) -> list[dict]:
    """Checks that forecast snapshots are time-consistent and reconcile."""
    if "forecast_snapshots" not in t or t["forecast_snapshots"].empty:
        return []
    s = t["forecast_snapshots"]
    ids = _record_ids(s, "forecast_snapshots")
    out = []
    dup = s.duplicated(["snapshot_id", "opportunity_id"], keep="first")
    out += _issues("forecast_snapshots", ids[dup].unique(), "duplicate_snapshot_row", ERROR,
                   "Opportunity counted more than once in the same snapshot.")
    snap_date = _to_dates(s["snapshot_date"])
    as_of = _to_dates(s["stage_as_of_date"])
    created = _to_dates(s["opportunity_created_date"])
    leak = (as_of > snap_date) | (created > snap_date)
    out += _issues("forecast_snapshots", ids[leak], "future_information_leakage", ERROR,
                   "Snapshot contains information dated after the snapshot date.")
    closed = s["stage"].isin(CLOSED_STAGES) & ~leak
    out += _issues("forecast_snapshots", ids[closed], "closed_deal_in_snapshot", WARNING,
                   "Closed deal appears in an open-pipeline snapshot.")
    bad_stage = s["stage"].notna() & ~s["stage"].isin(config.STAGE_NAMES)
    out += _issues("forecast_snapshots", ids[bad_stage], "invalid_stage", ERROR,
                   "Stage is not in the allowed list.")
    prob = pd.to_numeric(s["stage_probability"], errors="coerce")
    expected = s["stage"].map(DEFAULT_PROB)
    bad = expected.notna() & ((prob - expected).abs() > 1e-9)
    out += _issues("forecast_snapshots", ids[bad], "stage_probability_inconsistent", ERROR,
                   "Snapshot probability does not match the stage's standard probability.")
    if "weighted_amount" in s:
        diff = (s["weighted_amount"] - s["deal_amount"] * prob).abs()
        out += _issues("forecast_snapshots", ids[diff > 0.5], "weighted_pipeline_mismatch", ERROR,
                       "Weighted amount is not deal amount x probability.")
    bad_cur = ~s["currency"].isin(config.ALLOWED_CURRENCIES)
    out += _issues("forecast_snapshots", ids[bad_cur], "mixed_currency", ERROR,
                   f"Currency is not {config.BASE_CURRENCY}; totals would mix currencies.")
    return out


# ======================================================================
# Run everything + build the report
# ======================================================================

ISSUE_COLUMNS = ["table_name", "record_id", "issue_type", "severity", "message", "recommendation"]


def validate_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every check. Returns one row per issue found."""
    issues = []
    for table, df in tables.items():
        issues += check_primary_keys(table, df)
        issues += check_required_values(table, df)
        issues += check_dates(table, df)
    for check in (check_master_tables, check_opportunities, check_stage_history,
                  check_activities, check_bookings, check_quotas, check_snapshots):
        issues += check(tables)
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def build_summary(issues: pd.DataFrame, tables: dict[str, pd.DataFrame],
                  dataset_name: str) -> dict:
    by_type = (issues.groupby(["issue_type", "severity"]).size().reset_index(name="count")
               .sort_values("count", ascending=False))
    return {
        "dataset": dataset_name,
        "rows_checked_by_table": {k: int(len(v)) for k, v in tables.items()},
        "total_rows_checked": int(sum(len(v) for v in tables.values())),
        "errors": int((issues["severity"] == ERROR).sum()),
        "warnings": int((issues["severity"] == WARNING).sum()),
        "categories": [
            {"issue_type": r.issue_type, "severity": r.severity, "count": int(r.count),
             "recommendation": RECOMMENDATIONS.get(r.issue_type, "")}
            for r in by_type.itertuples()],
    }


def summary_to_markdown(summary: dict) -> str:
    lines = [f"# Data-quality report: {summary['dataset']}", "",
             f"- Total rows checked: **{summary['total_rows_checked']:,}**",
             f"- Errors: **{summary['errors']}**",
             f"- Warnings: **{summary['warnings']}**", "",
             "## Rows checked by table", "", "| Table | Rows |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in summary["rows_checked_by_table"].items()]
    lines += ["", "## Issue categories", ""]
    if not summary["categories"]:
        lines.append("No issues found.")
    else:
        lines += ["| Issue type | Severity | Count | Recommended correction |", "|---|---|---|---|"]
        lines += [f"| {c['issue_type']} | {c['severity']} | {c['count']} | {c['recommendation']} |"
                  for c in summary["categories"]]
    return "\n".join(lines) + "\n"


def write_report(issues: pd.DataFrame, summary: dict, folder: Path, prefix: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    issues.to_csv(folder / f"{prefix}_issues.csv", index=False, lineterminator="\n")
    (folder / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    (folder / f"{prefix}_report.md").write_text(summary_to_markdown(summary))


def compare_with_labels(issues: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For each injected (labelled) problem: was it detected?"""
    found = set(zip(issues["table_name"], issues["record_id"], issues["issue_type"]))
    out = labels.copy()
    out["detected"] = [(r.table_name, str(r.record_id), r.issue_type) in found
                       for r in labels.itertuples()]
    return out


def load_for_validation(folder: Path) -> dict[str, pd.DataFrame]:
    """Read every CSV in a folder as-is (dates stay as text)."""
    return {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))
            if p.stem != "manifest"}
