"""Tests for src/data_validation.py."""

import pandas as pd
import pytest

import config
from src import data_validation as dv
from src.error_injection import build_error_sample


@pytest.fixture(scope="module")
def sample(tiny_tables):
    return build_error_sample(tiny_tables, seed=config.RANDOM_SEED, size=40)


def test_clean_data_has_no_errors_or_warnings(tiny_tables):
    clean = {k: v for k, v in tiny_tables.items() if k != "opportunity_outcomes"}
    issues = dv.validate_tables(clean)
    assert issues.empty, issues.head().to_string()


def test_every_injected_error_is_detected(sample):
    tables, labels = sample
    issues = dv.validate_tables(tables)
    result = dv.compare_with_labels(issues, labels)
    missed = result[~result["detected"]]
    assert missed.empty, missed.to_string()


def test_sample_has_no_unexpected_issues(sample):
    """Precision: every issue found should correspond to an injected label."""
    tables, labels = sample
    issues = dv.validate_tables(tables)
    labelled = set(zip(labels["record_id"].astype(str), labels["issue_type"]))
    found = set(zip(issues["record_id"], issues["issue_type"]))
    assert found <= labelled, found - labelled


def test_all_required_edge_cases_are_covered(sample):
    _, labels = sample
    needed = {"future_information_leakage", "stage_probability_inconsistent",
              "duplicate_snapshot_row", "close_before_creation", "mixed_currency",
              "duplicate_opportunity_id", "negative_deal_amount", "invalid_date",
              "invalid_stage", "invalid_probability", "missing_rep_reference",
              "missing_account_reference", "missing_required_value"}
    assert needed <= set(labels["issue_type"])


def _opps(**overrides):
    row = {"opportunity_id": "OPP-2025-000001", "account_id": "ACC-000001",
           "rep_id": "REP-000001", "product": "CRM Cloud",
           "opportunity_created_date": "2025-01-10", "expected_close_date": "2025-03-10",
           "opportunity_stage": "Proposal", "deal_amount": 100000.0, "currency": "INR",
           "current_probability": 0.5}
    row.update(overrides)
    return {"opportunities": pd.DataFrame([row])}


@pytest.mark.parametrize("overrides, issue_type", [
    ({"deal_amount": -5.0}, "negative_deal_amount"),
    ({"expected_close_date": "2024-12-01"}, "close_before_creation"),
    ({"opportunity_stage": "Won-ish"}, "invalid_stage"),
    ({"current_probability": -0.1}, "invalid_probability"),
    ({"currency": "EUR"}, "mixed_currency"),
    ({"opportunity_created_date": "2025-02-30"}, "invalid_date"),
    ({"product": None}, "missing_required_value"),
    ({"current_probability": 0.9}, "stage_probability_inconsistent"),
])
def test_single_check(overrides, issue_type):
    issues = dv.validate_tables(_opps(**overrides))
    assert issue_type in set(issues["issue_type"])


def test_valid_single_row_passes():
    assert dv.validate_tables(_opps()).empty


def test_summary_counts_errors_and_warnings():
    tables = _opps(deal_amount=-1.0, current_probability=0.9)
    issues = dv.validate_tables(tables)
    s = dv.build_summary(issues, tables, "test")
    assert s["total_rows_checked"] == 1
    assert s["errors"] == 1 and s["warnings"] == 1
    assert {c["issue_type"] for c in s["categories"]} == {"negative_deal_amount",
                                                          "stage_probability_inconsistent"}
    assert all(c["recommendation"] for c in s["categories"])
