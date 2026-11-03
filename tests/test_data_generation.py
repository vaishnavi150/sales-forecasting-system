"""Tests for src/data_generation.py: reproducibility, IDs, joins, derived values."""

import re
import tempfile
from pathlib import Path

import pandas as pd

import config
from src.data_generation import generate_all, run_generation, save_tables


def test_same_seed_gives_identical_files(tiny_scale):
    sums = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            sums.append(save_tables(generate_all(tiny_scale, seed=123), Path(tmp)))
    assert sums[0] == sums[1]


def test_different_seed_gives_different_data(tiny_scale):
    a = generate_all(tiny_scale, seed=1)["opportunities"]
    b = generate_all(tiny_scale, seed=2)["opportunities"]
    assert not a["deal_amount"].equals(b["deal_amount"])


def test_row_counts_follow_scale(tiny_tables, tiny_scale):
    assert len(tiny_tables["sales_reps"]) == tiny_scale.num_reps
    assert len(tiny_tables["accounts"]) == tiny_scale.num_accounts
    assert len(tiny_tables["opportunities"]) == tiny_scale.num_opportunities
    assert len(tiny_tables["territories"]) == tiny_scale.num_territories


def test_id_formats(tiny_tables):
    assert tiny_tables["sales_reps"]["rep_id"].str.fullmatch(r"REP-\d{6}").all()
    assert tiny_tables["opportunities"]["opportunity_id"].str.fullmatch(r"OPP-20\d{2}-\d{6}").all()
    assert tiny_tables["quotas"]["quota_id"].str.fullmatch(r"QUOTA-REP-\d{6}-20\d{2}Q[1-4]").all()
    ids = tiny_tables["stage_history"][["stage_history_id", "opportunity_id"]]
    assert all(re.fullmatch(rf"STG-{o}-\d{{3}}", s) for s, o in ids.itertuples(index=False))


def test_opportunity_year_in_id_matches_creation_year(tiny_tables):
    o = tiny_tables["opportunities"]
    assert (o["opportunity_id"].str[4:8].astype(int) == o["opportunity_created_date"].dt.year).all()


def test_primary_keys_unique_and_foreign_keys_valid(tiny_tables):
    t = tiny_tables
    for table, key in [("sales_reps", "rep_id"), ("accounts", "account_id"),
                       ("opportunities", "opportunity_id"), ("activities", "activity_id"),
                       ("stage_history", "stage_history_id"), ("quotas", "quota_id")]:
        assert t[table][key].is_unique, table
    assert t["opportunities"]["rep_id"].isin(t["sales_reps"]["rep_id"]).all()
    assert t["opportunities"]["account_id"].isin(t["accounts"]["account_id"]).all()
    assert t["stage_history"]["opportunity_id"].isin(t["opportunities"]["opportunity_id"]).all()
    assert t["activities"]["opportunity_id"].isin(t["opportunities"]["opportunity_id"]).all()


def test_nothing_after_data_end_date(tiny_tables):
    end = pd.Timestamp(config.DATA_END_DATE)
    assert (tiny_tables["stage_history"]["change_date"] <= end).all()
    assert (tiny_tables["activities"]["activity_date"] <= end).all()
    assert (tiny_tables["opportunities"]["opportunity_created_date"] <= end).all()


def test_bookings_reconcile_with_won_deals(tiny_tables):
    o, b = tiny_tables["opportunities"], tiny_tables["historical_bookings"]
    won = o[o["opportunity_stage"] == config.WON_STAGE]
    assert set(won["opportunity_id"]) == set(b["opportunity_id"])
    assert b["booking_amount"].sum() == won["deal_amount"].sum()


def test_quotas_are_positive_even_for_last_day_joiners():
    """Regression test: a rep joining on a quarter's last day once got a 0 quota."""
    import numpy as np
    from src.data_generation import generate_quotas
    reps = pd.DataFrame({"rep_id": ["REP-000001"], "team": ["Field Sales"],
                         "joining_date": [pd.Timestamp(2025, 9, 30)]})
    q = generate_quotas(reps, np.random.default_rng(0))
    assert (q["quota_amount"] > 0).all()
    assert q.iloc[0]["quota_quarter"] == "2025Q3"


def test_outcomes_agree_with_current_stage(tiny_tables):
    o = tiny_tables["opportunities"].merge(tiny_tables["opportunity_outcomes"], on="opportunity_id")
    expected = o["opportunity_stage"].map({config.WON_STAGE: "Won", config.LOST_STAGE: "Lost"}).fillna("Open")
    assert (expected == o["final_status"]).all()


def test_outcome_columns_not_in_opportunity_table(tiny_tables):
    """Ground truth stays separate from the main input table."""
    cols = set(tiny_tables["opportunities"].columns)
    assert not cols & {"final_status", "actual_outcome", "won_amount", "slipped"}


def test_realistic_mix_of_outcomes(tiny_tables):
    status = tiny_tables["opportunity_outcomes"]["final_status"].value_counts(normalize=True)
    assert 0.2 < status["Won"] < 0.6
    assert status["Open"] > 0  # some deals are still open at the export date


def test_run_generation_writes_ground_truth_separately(tiny_scale, tmp_path):
    raw, truth, sample = tmp_path / "raw", tmp_path / "truth", tmp_path / "sample"
    manifest = run_generation(tiny_scale, 7, raw, truth, sample)
    assert (truth / "opportunity_outcomes.csv").exists()
    assert not (raw / "opportunity_outcomes.csv").exists()
    assert (truth / "anomaly_labels.csv").exists()
    assert (raw / "manifest.json").exists()
    assert manifest["injected_anomalies"] == 16
