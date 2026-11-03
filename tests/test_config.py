"""
tests/test_config.py - checks that config.py is set up correctly.

Each function starting with "test_" is one test. pytest finds and runs
them automatically. "assert" means: "this must be true, otherwise the
test fails".
"""

import pytest

import config


def test_default_scale_is_starter(monkeypatch):
    # Remove SFS_SCALE in case it is set on this computer.
    monkeypatch.delenv("SFS_SCALE", raising=False)
    cfg = config.get_config()
    assert cfg.scale.name == "starter"
    assert cfg.scale.num_reps == 100
    assert cfg.scale.num_accounts == 500
    assert cfg.scale.num_opportunities == 5_000


def test_full_scale_can_be_selected_by_environment(monkeypatch):
    monkeypatch.setenv("SFS_SCALE", "full")
    cfg = config.get_config()
    assert cfg.scale.name == "full"
    assert cfg.scale.num_opportunities == 20_000


def test_unknown_scale_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown scale"):
        config.get_config("huge")


def test_full_scale_is_at_least_as_large_as_starter():
    starter, full = config.SCALES["starter"], config.SCALES["full"]
    assert full.num_reps >= starter.num_reps
    assert full.num_accounts >= starter.num_accounts
    assert full.num_opportunities >= starter.num_opportunities


def test_stage_probabilities_are_valid():
    for stage in config.STAGES:
        assert 0.0 <= stage.default_probability <= 1.0
    # Stage IDs and names must be unique.
    assert len({s.stage_id for s in config.STAGES}) == len(config.STAGES)
    assert len(set(config.STAGE_NAMES)) == len(config.STAGE_NAMES)


def test_snapshot_dates_are_in_order_and_inside_data_period():
    dates = config.SNAPSHOT_DATES
    assert len(dates) == 8
    assert dates == sorted(dates)
    assert all(config.DATA_START_DATE <= d <= config.DATA_END_DATE for d in dates)


def test_id_formats_match_the_specification():
    assert config.REP_ID_FORMAT.format(n=1) == "REP-000001"
    assert config.OPPORTUNITY_ID_FORMAT.format(year=2026, n=1) == "OPP-2026-000001"
    assert config.QUOTA_ID_FORMAT.format(rep_id="REP-000001", year=2026, quarter=1) \
        == "QUOTA-REP-000001-2026Q1"
    assert config.FORECAST_ID_FORMAT.format(snapshot_date="2026-01-31", n=1) \
        == "FCST-2026-01-31-001"
    assert config.STAGE_HISTORY_ID_FORMAT.format(opportunity_id="OPP-2026-000001", n=1) \
        == "STG-OPP-2026-000001-001"
