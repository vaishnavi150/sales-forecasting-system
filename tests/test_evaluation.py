"""Tests for src/evaluation.py, using small numbers we can check by hand."""

import math

import numpy as np
import pytest

from src import evaluation as ev

F = [110.0, 90.0, 100.0]
A = [100.0, 100.0, 100.0]


def test_mae():
    assert ev.mae(F, A) == pytest.approx(20 / 3)   # (10 + 10 + 0) / 3


def test_rmse():
    assert ev.rmse(F, A) == pytest.approx(math.sqrt(200 / 3))


def test_wape():
    assert ev.wape(F, A) == pytest.approx(20 / 300)


def test_bias_sign():
    assert ev.bias([120, 120], [100, 100]) == pytest.approx(0.2)    # over-forecast
    assert ev.bias([80, 80], [100, 100]) == pytest.approx(-0.2)     # under-forecast
    assert ev.bias(F, A) == pytest.approx(0.0)


def test_zero_actual_period_does_not_break_wape():
    assert ev.wape([10, 100], [0, 100]) == pytest.approx(10 / 100)


def test_all_zero_actuals_give_nan_not_infinity():
    assert math.isnan(ev.wape([5, 5], [0, 0]))
    assert math.isnan(ev.bias([5, 5], [0, 0]))


def test_percentage_errors_handle_zero():
    pe = ev.percentage_errors([110, 5], [100, 0])
    assert pe[0] == pytest.approx(0.1)
    assert np.isnan(pe[1])


def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        ev.mae([1, 2], [1])
    with pytest.raises(ValueError):
        ev.mae([], [])


def test_deal_level_metrics():
    m = ev.deal_level_metrics([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert m["ROC_AUC"] == pytest.approx(1.0)
    assert m["Brier"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.04) / 4)


def test_deal_level_metrics_single_class_gives_nan_auc():
    assert math.isnan(ev.deal_level_metrics([0.2, 0.3], [0, 0])["ROC_AUC"])
