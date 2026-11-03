"""Tests for src/scenario_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from src import scenario_analysis as sa


def test_certain_deals_give_fixed_total():
    totals = sa.simulate_totals([100, 200, 300], [1.0, 0.0, 1.0], runs=500)
    assert np.all(totals == 400)


def test_simulation_is_reproducible():
    a = sa.simulate_totals([100] * 20, [0.3] * 20, runs=1000, seed=5)
    b = sa.simulate_totals([100] * 20, [0.3] * 20, runs=1000, seed=5)
    assert np.array_equal(a, b)


def test_simulation_mean_close_to_expected_value():
    amounts, probs = np.full(200, 1000.0), np.full(200, 0.25)
    totals = sa.simulate_totals(amounts, probs, runs=5000, seed=1)
    assert totals.mean() == pytest.approx(50_000, rel=0.02)


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        sa.simulate_totals([100], [1.2], runs=10)


def _history(ratios):
    return pd.DataFrame({"actual_to_expected": ratios, "inside_P10_P90": [True] * len(ratios)})


def test_scenarios_are_ordered_and_calibrated_with_enough_history():
    amounts, probs = np.full(50, 1000.0), np.full(50, 0.4)
    s = sa.build_scenarios(amounts, probs, _history([0.8, 0.9, 1.0, 1.1, 1.2]), "m", seed=1)
    assert s["downside"] <= s["base"] <= s["upside"]
    assert s["calibrated"] is True
    assert s["base"] == pytest.approx(20_000)
    assert s["downside"] <= s["base"] * 0.85  # at least as wide as the history implies


def test_not_calibrated_with_little_history():
    s = sa.build_scenarios(np.full(10, 1000.0), np.full(10, 0.5), _history([1.0, 1.1]), "m")
    assert s["calibrated"] is False
    assert "NOT calibrated" in s["limitation"]
    assert s["downside"] == s["simulation"]["P10"]
