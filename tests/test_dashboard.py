"""
Smoke tests for the Streamlit dashboard.

They run dashboard/app.py in Streamlit's test mode (no browser) and check
that it loads, shows the key numbers, and survives every filter. They need
the outputs of run_pipeline.py, so they are SKIPPED until you have run it.
"""

import json

import pytest

import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(config.PROJECT_ROOT / "dashboard" / "app.py")
needs_outputs = pytest.mark.skipif(
    not (config.REPORTS_DIR / "current_forecast.csv").exists(),
    reason="Run `python run_pipeline.py` first to create dashboard inputs.")


def _page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


@needs_outputs
def test_dashboard_loads_with_key_numbers():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 6
    text = _page_text(at)
    for label in ["Open pipeline", "Open opportunities", "Stale opportunities", "High-risk deals",
                  "Expected bookings", "Downside", "Base", "Upside", "Likely range"]:
        assert label in text, label


@needs_outputs
def test_overview_shows_the_saved_base_forecast():
    scen = json.loads((config.REPORTS_DIR / "scenarios.json").read_text(encoding="utf-8"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert f"₹{scen['base'] / 1e7:,.2f} cr" in _page_text(at)


@needs_outputs
def test_dashboard_filters_and_search_work():
    at = AppTest.from_file(APP, default_timeout=120).run()
    region = at.sidebar.multiselect[0].options[0]
    at.sidebar.multiselect[0].select(region).run()
    assert not at.exception
    at.sidebar.multiselect[2].select(at.sidebar.multiselect[2].options[0]).run()
    assert not at.exception
    assert "Quota comparison is hidden" in _page_text(at)
    at.text_input[0].input("OPP").run()
    assert not at.exception
