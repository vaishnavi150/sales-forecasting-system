"""
Smoke test for the Streamlit dashboard.

It runs dashboard/app.py in Streamlit's test mode (no browser) and checks
that it loads without errors. It needs the outputs of run_pipeline.py,
so it is SKIPPED if you have not run the pipeline yet.
"""

import pytest

import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

needs_outputs = pytest.mark.skipif(
    not (config.REPORTS_DIR / "current_forecast.csv").exists(),
    reason="Run `python run_pipeline.py` first to create dashboard inputs.")


@needs_outputs
def test_dashboard_loads_without_errors():
    at = AppTest.from_file(str(config.PROJECT_ROOT / "dashboard" / "app.py"),
                           default_timeout=120).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    for expected in ["Total open pipeline", "Downside", "Base", "Upside",
                     "Open opportunities", "Stale opportunities"]:
        assert expected in labels
    assert len(at.tabs) == 5


@needs_outputs
def test_dashboard_filters_work():
    at = AppTest.from_file(str(config.PROJECT_ROOT / "dashboard" / "app.py"),
                           default_timeout=120).run()
    region = at.sidebar.multiselect[0].options[0]
    at.sidebar.multiselect[0].select(region).run()
    assert not at.exception
    at.text_input[0].input("OPP").run()
    assert not at.exception
