"""Tests for snapshots and features - mainly that NO future information leaks in."""

import pandas as pd
import pytest

import config
from src.feature_engineering import FEATURE_COLUMNS, build_features
from src.snapshots import build_snapshot, horizon_end, label_snapshot

D = pd.Timestamp(2025, 6, 30)


def _truncate(tables, d):
    """Delete every record dated after d - what a CRM export on day d would contain."""
    t = dict(tables)
    t["stage_history"] = tables["stage_history"][tables["stage_history"]["change_date"] <= d]
    t["activities"] = tables["activities"][tables["activities"]["activity_date"] <= d]
    t["opportunities"] = tables["opportunities"][tables["opportunities"]["opportunity_created_date"] <= d]
    return t


@pytest.mark.parametrize("d", [pd.Timestamp(2024, 12, 31), D, pd.Timestamp(2026, 3, 31)])
def test_features_identical_when_future_data_is_removed(tiny_tables, d):
    full = build_features(tiny_tables, build_snapshot(tiny_tables, d), d)
    cut_tables = _truncate(tiny_tables, d)
    cut = build_features(cut_tables, build_snapshot(cut_tables, d), d)
    pd.testing.assert_frame_equal(full, cut)


def test_snapshot_contains_only_deals_open_on_that_date(tiny_tables):
    snap = build_snapshot(tiny_tables, D)
    assert len(snap) > 0
    assert snap["stage"].isin(config.OPEN_STAGE_NAMES).all()
    assert (snap["stage_as_of_date"] <= D).all()
    assert (snap["opportunity_created_date"] <= D).all()
    assert snap["opportunity_id"].is_unique


def test_snapshot_weighted_amount_reconciles(tiny_tables):
    snap = build_snapshot(tiny_tables, D)
    assert ((snap["deal_amount"] * snap["stage_probability"] - snap["weighted_amount"]).abs() < 1e-6).all()


def test_all_feature_columns_present_and_complete(tiny_tables):
    feats = build_features(tiny_tables, build_snapshot(tiny_tables, D), D)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert feats[FEATURE_COLUMNS].notna().all().all()
    assert (feats["opportunity_age_days"] >= 0).all()
    assert (feats["days_since_last_activity"] >= 0).all()
    assert feats["stage_conversion_rate"].between(0, 1).all()


def test_labels_refused_before_period_is_finished(tiny_tables):
    snap = build_snapshot(tiny_tables, D)
    with pytest.raises(ValueError, match="not available"):
        label_snapshot(snap, tiny_tables["opportunity_outcomes"], data_available_until="2025-08-01")


def test_labels_only_count_wins_inside_the_quarter(tiny_tables):
    snap = build_snapshot(tiny_tables, D)
    lab = label_snapshot(snap, tiny_tables["opportunity_outcomes"])
    oc = tiny_tables["opportunity_outcomes"].set_index("opportunity_id")
    won = lab[lab["won_in_horizon"] == 1]
    dates = oc.loc[won["opportunity_id"], "close_date"]
    assert ((dates > D) & (dates <= horizon_end(D))).all()
    assert (oc.loc[won["opportunity_id"], "final_status"] == "Won").all()


def test_horizon_end_is_next_quarter_end():
    assert horizon_end("2025-03-31") == pd.Timestamp(2025, 6, 30)
    assert horizon_end("2026-09-30") == pd.Timestamp(2026, 12, 31)
    assert horizon_end("2025-12-31") == pd.Timestamp(2026, 3, 31)
