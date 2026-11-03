"""
src/business_insights.py - finds risks in the current pipeline.

Every insight is TRACEABLE: each flag is attached to a specific
opportunity_id, and every summary (per rep, per region, per account)
lists the opportunity IDs it was calculated from.

RULES (each threshold lives in config.py)
-----------------------------------------
stale             : no activity for STALE_ACTIVITY_DAYS (30) days or more
overdue           : expected close date is already in the past
large_low_prob    : deal amount in the top 10% AND win probability < 30%
slip_risk         : expected to close this quarter, but the model's chance
                    of winning THIS quarter is less than half of the stage's
                    historical win rate (the deal is likely to move later)

About sales reps: we only report FACTS about their open deals (how many
are flagged). We do not rate or judge people; a high share of flagged
deals simply suggests a pipeline review conversation.
"""

from __future__ import annotations

import pandas as pd

import config
from src.snapshots import horizon_end

FLAG_LABELS = {
    "flag_stale": f"No activity for {config.STALE_ACTIVITY_DAYS}+ days",
    "flag_overdue": "Past expected close date",
    "flag_large_low_prob": "Large deal with low win probability",
    "flag_slip_risk": "Likely to slip to a later quarter",
}


def flag_opportunities(feats: pd.DataFrame, risk_prob_col: str, forecast_prob_col: str,
                       snapshot_date) -> pd.DataFrame:
    """Add flag columns, a readable risk_reasons text and a risk_level."""
    df = feats.copy()
    h_end = horizon_end(snapshot_date)
    p = df[risk_prob_col]
    df["win_probability"] = p
    df["forecast_probability"] = df[forecast_prob_col]
    df["expected_booking"] = df["deal_amount"] * df["forecast_probability"]

    large_cutoff = df["deal_amount"].quantile(config.LARGE_DEAL_PERCENTILE) if len(df) else 0
    in_quarter = df["expected_close_date"] <= h_end
    df["flag_stale"] = df["days_since_last_activity"] >= config.STALE_ACTIVITY_DAYS
    df["flag_overdue"] = df["days_until_expected_close"] < 0
    df["flag_large_low_prob"] = (df["deal_amount"] >= large_cutoff) & (p < config.LOW_WIN_PROBABILITY)
    df["flag_slip_risk"] = in_quarter & (p < 0.5 * df["stage_conversion_rate"])

    flags = list(FLAG_LABELS)
    df["risk_flag_count"] = df[flags].sum(axis=1)
    df["risk_reasons"] = df[flags].apply(
        lambda row: "; ".join(FLAG_LABELS[f] for f in flags if row[f]), axis=1)
    df["risk_level"] = "Low"
    df.loc[df["risk_flag_count"] == 1, "risk_level"] = "Medium"
    df.loc[(df["risk_flag_count"] >= 2) | df["flag_large_low_prob"], "risk_level"] = "High"
    return df


def _ids(series: pd.Series, limit: int = 50) -> str:
    ids = sorted(series.astype(str))
    text = ", ".join(ids[:limit])
    return text + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")


def rep_followup(flagged: pd.DataFrame, reps: pd.DataFrame,
                 min_open: int = 3, min_share: float = 0.5) -> pd.DataFrame:
    """Reps whose open pipeline has a high share of flagged deals."""
    df = flagged.assign(is_flagged=flagged["risk_flag_count"] > 0)
    g = df.groupby("rep_id").agg(open_deals=("opportunity_id", "count"),
                                 flagged_deals=("is_flagged", "sum"),
                                 open_pipeline=("deal_amount", "sum"))
    g["flagged_share"] = g["flagged_deals"] / g["open_deals"]
    g["flagged_opportunity_ids"] = df[df["is_flagged"]].groupby("rep_id")["opportunity_id"].apply(_ids)
    g = g.reset_index().merge(reps[["rep_id", "rep_name", "region", "team"]], on="rep_id", how="left")
    g = g[(g["open_deals"] >= min_open) & (g["flagged_share"] >= min_share)]
    g["note"] = "Pipeline review suggested (based only on deal-level flags)."
    return g.sort_values(["flagged_share", "open_pipeline"], ascending=False).reset_index(drop=True)


def region_gaps(flagged: pd.DataFrame, quotas: pd.DataFrame, reps: pd.DataFrame,
                quarter: str) -> pd.DataFrame:
    """Forecast vs quota per region for the forecast quarter."""
    q = quotas[quotas["quota_quarter"] == quarter].merge(reps[["rep_id", "region"]], on="rep_id")
    quota_by_region = q.groupby("region")["quota_amount"].sum()
    f = flagged.groupby("region").agg(expected_booking=("expected_booking", "sum"),
                                      open_pipeline=("deal_amount", "sum"),
                                      open_deals=("opportunity_id", "count"),
                                      opportunity_ids=("opportunity_id", _ids))
    out = f.join(quota_by_region.rename("quota"), how="outer").fillna({"expected_booking": 0, "quota": 0})
    out["gap_to_quota"] = out["quota"] - out["expected_booking"]
    out["forecast_coverage"] = out["expected_booking"] / out["quota"].where(out["quota"] > 0)
    out["quarter"] = quarter
    return out.reset_index().rename(columns={"index": "region"}).sort_values("gap_to_quota", ascending=False)


def concentration(flagged: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Accounts / reps holding a large share of the open pipeline."""
    total = flagged["deal_amount"].sum()
    rows = []
    for key in ["account_id", "rep_id"]:
        g = flagged.groupby(key).agg(open_pipeline=("deal_amount", "sum"),
                                     open_deals=("opportunity_id", "count"),
                                     opportunity_ids=("opportunity_id", _ids)).reset_index()
        g["share_of_pipeline"] = g["open_pipeline"] / total if total else 0.0
        g = g.sort_values("share_of_pipeline", ascending=False).head(5)
        for r in g.itertuples():
            rows.append({"dimension": key, "id": getattr(r, key), "open_pipeline": r.open_pipeline,
                         "open_deals": r.open_deals, "share_of_pipeline": r.share_of_pipeline,
                         "opportunity_ids": r.opportunity_ids,
                         "above_threshold": r.share_of_pipeline > config.CONCENTRATION_SHARE})
    top10 = flagged.nlargest(10, "deal_amount")
    rows.append({"dimension": "top_10_deals", "id": "Top 10 deals by amount",
                 "open_pipeline": top10["deal_amount"].sum(), "open_deals": len(top10),
                 "share_of_pipeline": top10["deal_amount"].sum() / total if total else 0.0,
                 "opportunity_ids": _ids(top10["opportunity_id"]),
                 "above_threshold": (top10["deal_amount"].sum() / total if total else 0) > 0.5})
    out = pd.DataFrame(rows)
    names = dict(zip(accounts["account_id"], accounts["account_name"]))
    out["name"] = out["id"].map(names).fillna(out["id"])
    return out


def insight_summary(flagged: pd.DataFrame) -> dict:
    out = {"open_opportunities": int(len(flagged))}
    for f, label in FLAG_LABELS.items():
        sel = flagged[flagged[f]]
        out[f.replace("flag_", "")] = {"label": label, "count": int(len(sel)),
                                       "pipeline_value": float(sel["deal_amount"].sum()),
                                       "opportunity_ids": _ids(sel["opportunity_id"], limit=20)}
    return out
