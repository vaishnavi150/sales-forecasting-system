"""
dashboard/app.py - interactive Streamlit dashboard.

Run from the project folder (after `python run_pipeline.py`):

    streamlit run dashboard/app.py

The dashboard only READS files produced by run_pipeline.py (reports/,
data/) and the SQLite database. It never changes data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Let the dashboard import config.py and src/ from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import database  # noqa: E402
from src import scenario_analysis as sa  # noqa: E402
from src.business_insights import FLAG_LABELS  # noqa: E402
from src.forecasting import METHOD_LABELS, METHODS  # noqa: E402

REPORTS = config.REPORTS_DIR
CRORE = 1e7

st.set_page_config(page_title="Sales Forecasting System", layout="wide")


# ----------------------------------------------------------------------
# Loading (cached so the files are read only once)
# ----------------------------------------------------------------------

@st.cache_data
def load_outputs() -> dict:
    out = {
        "current": pd.read_csv(REPORTS / "current_forecast.csv",
                               parse_dates=["expected_close_date", "opportunity_created_date",
                                            "last_activity_date"]),
        "periods": pd.read_csv(REPORTS / "backtest_periods.csv", parse_dates=["snapshot_date"]),
        "comparison": pd.read_csv(REPORTS / "method_comparison.csv"),
        "deal_metrics": pd.read_csv(REPORTS / "deal_level_metrics.csv"),
        "intervals": pd.read_csv(REPORTS / "scenario_backtest.csv", parse_dates=["snapshot_date"]),
        "monthly": pd.read_csv(REPORTS / "monthly_forecast.csv"),
        "region_gaps": pd.read_csv(REPORTS / "region_gaps.csv"),
        "rep_followup": pd.read_csv(REPORTS / "rep_followup.csv"),
        "concentration": pd.read_csv(REPORTS / "concentration.csv"),
        "dq_issues_clean": pd.read_csv(REPORTS / "data_quality" / "clean_issues.csv"),
        "dq_issues_sample": pd.read_csv(REPORTS / "data_quality" / "sample_issues.csv"),
        "detection": pd.read_csv(REPORTS / "data_quality" / "sample_detection.csv"),
    }
    for name in ["scenarios", "insights", "run_info"]:
        out[name] = json.loads((REPORTS / f"{name}.json").read_text())
    for name in ["clean", "sample"]:
        out[f"dq_{name}"] = json.loads((REPORTS / "data_quality" / f"{name}_summary.json").read_text())
        out[f"dq_{name}_md"] = (REPORTS / "data_quality" / f"{name}_report.md").read_text()
    return out


@st.cache_data
def filtered_scenarios(amounts: tuple, probs: tuple, method: str) -> dict:
    hist = pd.read_csv(REPORTS / "scenario_backtest.csv")
    return sa.build_scenarios(np.array(amounts), np.array(probs), hist, method)


def cr(x: float) -> str:
    """Format rupees as crore, e.g. 54,38,00,000 -> '₹54.38 cr'."""
    return f"₹{x / CRORE:,.2f} cr"


required = ["current_forecast.csv", "scenarios.json", "backtest_periods.csv"]
if not all((REPORTS / f).exists() for f in required):
    st.error("Results not found. Run the pipeline first:  python run_pipeline.py")
    st.stop()

data = load_outputs()
current = data["current"]
scen = data["scenarios"]
run_info = data["run_info"]

st.title("Sales Forecasting System")
st.caption(f"Synthetic data (scale: {run_info['scale']}, seed {run_info['seed']}). "
           f"Forecast made on {scen['snapshot_date']} for {scen['forecast_quarter']}. "
           "Results do not represent any real company.")

# ----------------------------------------------------------------------
# Sidebar filters (apply to the current open pipeline)
# ----------------------------------------------------------------------
st.sidebar.header("Filters")
min_d = current["expected_close_date"].min().date()
max_d = current["expected_close_date"].max().date()
date_range = st.sidebar.date_input("Expected close date", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
regions = st.sidebar.multiselect("Region", sorted(current["region"].unique()))
rep_options = (current[["rep_id", "rep_name"]].drop_duplicates().sort_values("rep_id")
               .apply(lambda r: f"{r.rep_id} - {r.rep_name}", axis=1).tolist())
reps = st.sidebar.multiselect("Sales representative", rep_options)
stages = st.sidebar.multiselect("Stage", [s for s in config.OPEN_STAGE_NAMES
                                          if s in set(current["stage"])])

view = current.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    view = view[(view["expected_close_date"] >= start) & (view["expected_close_date"] <= end)]
if regions:
    view = view[view["region"].isin(regions)]
if reps:
    view = view[view["rep_id"].isin([r.split(" - ")[0] for r in reps])]
if stages:
    view = view[view["stage"].isin(stages)]
is_filtered = len(view) != len(current)
st.sidebar.caption(f"{len(view):,} of {len(current):,} open opportunities shown")

tabs = st.tabs(["Overview", "Forecast analysis", "Pipeline analysis",
                "Data quality", "Opportunity explorer"])

# ----------------------------------------------------------------------
# 1. Overview
# ----------------------------------------------------------------------
with tabs[0]:
    if view.empty:
        st.warning("No opportunities match the filters.")
    else:
        if is_filtered:
            s = filtered_scenarios(tuple(view["deal_amount"]), tuple(view["forecast_probability"]),
                                   scen["method"])
        else:
            s = scen
        c = st.columns(5)
        c[0].metric("Total open pipeline", cr(view["deal_amount"].sum()))
        c[1].metric("Expected bookings (Base)", cr(s["base"]))
        c[2].metric("Open opportunities", f"{len(view):,}")
        c[3].metric("Stale opportunities", f"{int(view['flag_stale'].sum()):,}")
        c[4].metric("High-risk opportunities", f"{int((view['risk_level'] == 'High').sum()):,}")
        c = st.columns(3)
        c[0].metric("Downside", cr(s["downside"]))
        c[1].metric("Base", cr(s["base"]))
        c[2].metric("Upside", cr(s["upside"]))
        st.caption(f"Forecast method: {METHOD_LABELS[scen['method']]} (lowest WAPE in the backtest). "
                   + ("Scenarios recalculated for the filtered deals. " if is_filtered else "")
                   + s["limitation"])
        with st.expander("Scenario assumptions"):
            for a in s["assumptions"]:
                st.write("- " + a)
            st.write(f"- Backtest: the simulation's P10-P90 range contained the actual result in "
                     f"{s['simulation_coverage']:.0%} of {s['backtest_periods']} past quarters.")
            if s.get("empirical"):
                st.write(f"- Past actual/forecast ratios: 10th percentile "
                         f"{s['empirical']['ratio_P10']:.2f}, 90th percentile {s['empirical']['ratio_P90']:.2f}.")

# ----------------------------------------------------------------------
# 2. Forecast analysis
# ----------------------------------------------------------------------
with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.subheader("Scenario comparison")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        vals = np.array([scen["downside"], scen["base"], scen["upside"]]) / CRORE
        ax.bar(["Downside", "Base", "Upside"], vals, color=["#c0504d", "#4f81bd", "#9bbb59"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("INR crore")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader(f"{scen['forecast_quarter']} forecast by month")
        st.bar_chart((data["monthly"].set_index("month")["expected_booking"] / CRORE)
                     .rename("Expected bookings (crore)"))
        st.caption("Overdue deals are counted in the first month of the quarter.")

    st.subheader("Forecast vs actual (backtest, quarterly)")
    periods = data["periods"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(periods))
    ax.plot(x, periods["actual"] / CRORE, "o-", color="black", linewidth=2, label="Actual")
    for m in METHODS:
        ax.plot(x, periods[m] / CRORE, "o--", alpha=0.8, label=METHOD_LABELS[m])
    ax.set_xticks(x, periods["forecast_quarter"])
    ax.set_ylabel("INR crore")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Historical forecast accuracy")
    comp = data["comparison"].copy()
    comp["method"] = comp["method"].map(METHOD_LABELS)
    comp["MAE (crore)"] = comp["MAE"] / CRORE
    comp["RMSE (crore)"] = comp["RMSE"] / CRORE
    st.dataframe(comp[["method", "MAE (crore)", "RMSE (crore)", "WAPE", "Bias", "periods"]]
                 .style.format({"MAE (crore)": "{:.2f}", "RMSE (crore)": "{:.2f}",
                                "WAPE": "{:.1%}", "Bias": "{:+.1%}"}), hide_index=True)
    st.caption("Quarterly totals. WAPE = total absolute error / total actual. "
               "Bias > 0 means the method forecast too high. Only quarters where every method "
               "had a forecast are compared.")
    dm = data["deal_metrics"].copy()
    dm["method"] = dm["method"].map(METHOD_LABELS)
    st.dataframe(dm.style.format({"Brier": "{:.3f}", "LogLoss": "{:.3f}", "ROC_AUC": "{:.3f}"}),
                 hide_index=True)
    st.caption("Individual deals: lower Brier/LogLoss and higher ROC AUC are better. "
               f"Risk flags use {METHOD_LABELS[run_info['risk_method']]} (lowest Brier).")

# ----------------------------------------------------------------------
# 3. Pipeline analysis
# ----------------------------------------------------------------------
with tabs[2]:
    if view.empty:
        st.warning("No opportunities match the filters.")
    else:
        a, b = st.columns(2)
        with a:
            st.subheader("Stage distribution")
            stage_tbl = (view.groupby("stage")["deal_amount"].sum()
                         .reindex([s for s in config.OPEN_STAGE_NAMES if s in set(view["stage"])]) / CRORE)
            st.bar_chart(stage_tbl.rename("Pipeline (crore)"))
        with b:
            st.subheader("Pipeline by region")
            ids = tuple(view["opportunity_id"])
            if config.DATABASE_PATH.exists():
                placeholders = ",".join("?" * len(ids))
                region_tbl = database.query(
                    "SELECT region, SUM(deal_amount) / 1e7 AS pipeline_crore, "
                    "SUM(expected_booking) / 1e7 AS expected_crore "
                    f"FROM current_forecast WHERE opportunity_id IN ({placeholders}) "
                    "GROUP BY region ORDER BY pipeline_crore DESC", ids).set_index("region")
            else:
                region_tbl = (view.groupby("region")[["deal_amount", "expected_booking"]].sum() / CRORE)
            st.bar_chart(region_tbl)
            st.caption("Queried from the SQLite database.")
        st.subheader("Pipeline by representative (top 15)")
        rep_tbl = (view.groupby(["rep_id", "rep_name"])
                   .agg(open_deals=("opportunity_id", "count"),
                        pipeline_crore=("deal_amount", lambda s: s.sum() / CRORE),
                        expected_crore=("expected_booking", lambda s: s.sum() / CRORE),
                        flagged_deals=("risk_flag_count", lambda s: int((s > 0).sum())))
                   .reset_index().sort_values("pipeline_crore", ascending=False).head(15))
        st.dataframe(rep_tbl.style.format({"pipeline_crore": "{:.2f}", "expected_crore": "{:.2f}"}),
                     hide_index=True)
        st.subheader("Opportunity age")
        bins = [-1, 30, 60, 90, 180, 10_000]
        labels = ["0-30 days", "31-60", "61-90", "91-180", "180+"]
        age = pd.cut(view["opportunity_age_days"], bins=bins, labels=labels)
        st.bar_chart(view.groupby(age, observed=False)["deal_amount"].sum().div(CRORE)
                     .rename("Pipeline (crore)"))

        st.subheader("Risks and insights")
        st.caption("Every row below is built from the opportunity IDs listed with it.")
        st.markdown("**Regions: forecast vs quota**")
        st.dataframe(data["region_gaps"][["region", "quarter", "quota", "expected_booking",
                                          "gap_to_quota", "forecast_coverage", "opportunity_ids"]]
                     .style.format({"quota": "{:,.0f}", "expected_booking": "{:,.0f}",
                                    "gap_to_quota": "{:,.0f}", "forecast_coverage": "{:.0%}"}),
                     hide_index=True)
        st.markdown("**Pipeline concentration**")
        st.dataframe(data["concentration"][["dimension", "name", "open_pipeline", "share_of_pipeline",
                                            "above_threshold", "opportunity_ids"]]
                     .style.format({"open_pipeline": "{:,.0f}", "share_of_pipeline": "{:.1%}"}),
                     hide_index=True)
        st.markdown("**Reps where a pipeline review is suggested**")
        st.caption("Based only on how many of their open deals carry risk flags. "
                   "This is not a judgment of the person's performance.")
        st.dataframe(data["rep_followup"][["rep_id", "rep_name", "region", "open_deals",
                                           "flagged_deals", "flagged_share", "flagged_opportunity_ids"]]
                     .style.format({"flagged_share": "{:.0%}"}), hide_index=True)

# ----------------------------------------------------------------------
# 4. Data quality
# ----------------------------------------------------------------------
with tabs[3]:
    for key, title in [("clean", "Main clean dataset"), ("sample", "Error-test sample (injected errors)")]:
        summ = data[f"dq_{key}"]
        st.subheader(title)
        c = st.columns(3)
        c[0].metric("Rows checked", f"{summ['total_rows_checked']:,}")
        c[1].metric("Errors", summ["errors"])
        c[2].metric("Warnings", summ["warnings"])
        if summ["categories"]:
            cats = pd.DataFrame(summ["categories"])
            st.bar_chart(cats.set_index("issue_type")["count"])
            st.dataframe(cats, hide_index=True)
        else:
            st.success("No issues found.")
        st.download_button(f"Download {key} data-quality report (Markdown)", data[f"dq_{key}_md"],
                           file_name=f"{key}_data_quality_report.md", key=f"md_{key}")
        st.download_button(f"Download {key} issue list (CSV)",
                           data[f"dq_issues_{key}"].to_csv(index=False),
                           file_name=f"{key}_issues.csv", key=f"csv_{key}")
    det = data["detection"]
    st.subheader("Were all injected errors detected?")
    st.write(f"{int(det['detected'].sum())} of {len(det)} labelled anomalies detected.")
    st.dataframe(det, hide_index=True)

# ----------------------------------------------------------------------
# 5. Opportunity explorer
# ----------------------------------------------------------------------
with tabs[4]:
    search = st.text_input("Search (opportunity ID, account or rep name)")
    risk_only = st.checkbox("Only show opportunities with risk flags")
    tbl = view
    if search:
        s = search.lower()
        tbl = tbl[tbl["opportunity_id"].str.lower().str.contains(s, regex=False)
                  | tbl["account_name"].str.lower().str.contains(s, regex=False)
                  | tbl["rep_name"].str.lower().str.contains(s, regex=False)]
    if risk_only:
        tbl = tbl[tbl["risk_flag_count"] > 0]
    cols = ["opportunity_id", "account_name", "rep_name", "region", "stage", "deal_amount",
            "win_probability", "forecast_probability", "expected_booking", "expected_close_date",
            "last_activity_date", "days_since_last_activity", "risk_level", "risk_reasons"]
    st.write(f"{len(tbl):,} opportunities")
    st.dataframe(tbl[cols].style.format({
        "deal_amount": "{:,.0f}", "expected_booking": "{:,.0f}",
        "win_probability": "{:.0%}", "forecast_probability": "{:.0%}",
        "expected_close_date": lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "",
        "last_activity_date": lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "none"}),
        hide_index=True, width="stretch")
    st.caption("win_probability = chance of winning in the forecast quarter (risk model). "
               "forecast_probability = probability used for the Base forecast. "
               "Risk flags: " + "; ".join(FLAG_LABELS.values()) + ".")
    st.download_button("Download table (CSV)", tbl[cols].to_csv(index=False),
                       file_name="opportunities.csv")
