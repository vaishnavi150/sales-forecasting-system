"""Tests for src/forecasting.py."""

import numpy as np
import pandas as pd
import pytest

import config
from src import forecasting as fc


def test_expected_booking_formula():
    assert np.allclose(fc.expected_booking([100_000, 200_000], [0.5, 0.25]), [50_000, 50_000])


@pytest.mark.parametrize("amount, prob", [([-1], [0.5]), ([100], [1.5]), ([100], [-0.1])])
def test_expected_booking_rejects_bad_inputs(amount, prob):
    with pytest.raises(ValueError):
        fc.expected_booking(amount, prob)


def test_baseline_ignores_deals_closing_after_the_quarter():
    feats = pd.DataFrame({"expected_close_date": pd.to_datetime(["2025-05-01", "2025-08-15", "2025-01-01"]),
                          "rate": [0.4, 0.4, 0.4]})
    p = fc.baseline_probabilities(feats, "2025-03-31", "rate")
    assert list(p) == [0.4, 0.0, 0.4]  # overdue deal still counts


@pytest.fixture(scope="module")
def frames(tiny_tables):
    return fc.prepare_frames(tiny_tables, config.SNAPSHOT_DATES)


def test_training_rows_only_use_finished_quarters(frames):
    d = pd.Timestamp(2025, 6, 30)
    train = fc.training_rows(frames, d)
    used = set(train["snapshot_date"])
    assert used == {s for s in frames if fc.horizon_end(s) <= d}
    assert d not in used


def test_backtest_produces_all_methods_and_actuals(frames):
    deals, periods = fc.backtest(frames)
    assert len(periods) == len(config.SNAPSHOT_DATES)
    assert periods["actual"].ge(0).all()
    later = periods.iloc[1:]  # first quarter has no earlier training data
    assert later[fc.METHODS].notna().all().all()
    for m in fc.METHODS:
        p = deals[f"p_{m}"].dropna()
        assert p.between(0, 1).all()


def test_actual_in_backtest_matches_ground_truth(frames, tiny_tables):
    _, periods = fc.backtest(frames)
    row = periods.iloc[3]
    d, h = row["snapshot_date"], fc.horizon_end(row["snapshot_date"])
    oc = tiny_tables["opportunity_outcomes"]
    snap_ids = frames[d]["opportunity_id"]
    won = oc[oc["opportunity_id"].isin(snap_ids) & (oc["final_status"] == "Won")
             & (oc["close_date"] > d) & (oc["close_date"] <= h)]
    assert row["actual"] == pytest.approx(won["won_amount"].sum())


def test_best_method_has_lowest_wape():
    periods = pd.DataFrame({"actual": [100.0, 100.0],
                            "crm_stage_probability": [150.0, 150.0],
                            "historical_stage_rate": [105.0, 95.0],
                            "logistic_regression": [120.0, 80.0],
                            "random_forest": [200.0, 200.0]})
    best, table = fc.choose_best_method(periods)
    assert best == "historical_stage_rate"
    assert table["WAPE"].is_monotonic_increasing


def test_train_model_needs_both_classes():
    df = pd.DataFrame({"won_in_horizon": [0, 0], "deal_amount": [1.0, 1.0]})
    with pytest.raises(ValueError, match="both"):
        fc.train_model("logistic_regression", df)


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        fc.make_model("magic")
