"""Tests for src/business_insights.py."""

import pandas as pd
import pytest

from src import business_insights as bi

SNAP = pd.Timestamp(2026, 9, 30)


def _deals():
    return pd.DataFrame({
        "opportunity_id": ["OPP-1", "OPP-2", "OPP-3", "OPP-4"],
        "rep_id": ["REP-1", "REP-1", "REP-1", "REP-1"],
        "account_id": ["ACC-1", "ACC-1", "ACC-2", "ACC-3"],
        "region": ["North"] * 4,
        "stage": ["Negotiation", "Prospecting", "Proposal", "Qualification"],
        "deal_amount": [100.0, 1000.0, 50.0, 60.0],
        "expected_close_date": pd.to_datetime(["2026-09-01", "2026-12-20", "2026-11-01", "2027-03-01"]),
        "days_until_expected_close": [-29, 81, 32, 152],
        "days_since_last_activity": [45, 2, 5, 3],
        "close_date_pushes": [0, 0, 1, 0],
        "stage_conversion_rate": [0.7, 0.4, 0.5, 0.45],
        "p_risk": [0.6, 0.1, 0.5, 0.3],
        "p_fc": [0.75, 0.1, 0.5, 0.2],
    })


def test_flags_follow_the_rules():
    f = bi.flag_opportunities(_deals(), "p_risk", "p_fc", SNAP).set_index("opportunity_id")
    assert f.loc["OPP-1", "flag_stale"] and f.loc["OPP-1", "flag_overdue"]
    assert f.loc["OPP-2", "flag_large_low_prob"]          # biggest deal, 10% chance
    assert f.loc["OPP-2", "flag_slip_risk"]               # early stage, close 81 days away
    assert not f.loc["OPP-3"][list(bi.FLAG_LABELS)].any()
    assert not f.loc["OPP-4", "flag_slip_risk"]           # expected next quarter, not this one
    assert f.loc["OPP-1", "risk_level"] == "High"
    assert "Past expected close date" in f.loc["OPP-1", "risk_reasons"]
    assert f.loc["OPP-1", "expected_booking"] == 75.0


def test_rep_followup_lists_the_flagged_opportunity_ids():
    f = bi.flag_opportunities(_deals(), "p_risk", "p_fc", SNAP)
    reps = pd.DataFrame({"rep_id": ["REP-1"], "rep_name": ["Test Rep"], "region": ["North"],
                         "team": ["Field Sales"]})
    out = bi.rep_followup(f, reps, min_open=3, min_share=0.5)
    assert len(out) == 1
    assert "OPP-1" in out.loc[0, "flagged_opportunity_ids"]
    assert "pipeline review suggested" in out.loc[0, "note"].lower()


def test_concentration_shares_add_up():
    f = bi.flag_opportunities(_deals(), "p_risk", "p_fc", SNAP)
    accounts = pd.DataFrame({"account_id": ["ACC-1", "ACC-2", "ACC-3"],
                             "account_name": ["A", "B", "C"]})
    c = bi.concentration(f, accounts)
    acc = c[c["dimension"] == "account_id"]
    assert acc["share_of_pipeline"].sum() == pytest.approx(1.0)
    top = acc.iloc[0]
    assert top["id"] == "ACC-1" and top["above_threshold"]


def test_region_gap_uses_quota_of_forecast_quarter():
    f = bi.flag_opportunities(_deals(), "p_risk", "p_fc", SNAP)
    reps = pd.DataFrame({"rep_id": ["REP-1"], "region": ["North"]})
    quotas = pd.DataFrame({"rep_id": ["REP-1", "REP-1"], "quota_quarter": ["2026Q4", "2026Q3"],
                           "quota_amount": [500.0, 999.0]})
    g = bi.region_gaps(f, quotas, reps, "2026Q4").set_index("region")
    expected = f["expected_booking"].sum()
    assert g.loc["North", "quota"] == 500.0
    assert g.loc["North", "gap_to_quota"] == 500.0 - expected
