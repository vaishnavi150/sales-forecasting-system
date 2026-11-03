"""
dashboard/app.py - interactive Streamlit dashboard.

Run from the project folder (after `python run_pipeline.py`):

    streamlit run dashboard/app.py

The dashboard only READS files produced by run_pipeline.py (reports/,
data/) and the SQLite database. It never changes data.

HOW THIS FILE IS ORGANISED
--------------------------
1. Design tokens (colours) and small formatting helpers
2. Loading the pipeline outputs (cached, so files are read once)
3. Chart helpers (Altair, which ships with Streamlit) - every chart has tooltips
4. Sidebar filters
5. One section per tab: Overview, Forecast, Pipeline, Risks, Data quality, Explorer
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

# Let the dashboard import config.py and src/ from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import database  # noqa: E402
from src import scenario_analysis as sa  # noqa: E402
from src.business_insights import FLAG_LABELS  # noqa: E402
from src.forecasting import METHOD_LABELS, METHODS  # noqa: E402

REPORTS = config.REPORTS_DIR
CRORE, LAKH = 1e7, 1e5

st.set_page_config(page_title="Sales forecast", page_icon=":material/insights:", layout="wide")

# ======================================================================
# 1. Design tokens and helpers
# ======================================================================
# Colours come from a colour-blind-checked palette. Each colour has ONE job.
INK = "#142033"          # main text, "actual" results
INK_2 = "#4e5a6b"        # secondary text
MUTED = "#8a94a3"        # axis labels
GRID = "#e4e8ee"         # gridlines
BASELINE = "#c3c9d2"     # axis lines
BLUE = "#2a78d6"         # the forecast (Base) and the main measure
BLUE_PALE = "#cde2fb"    # ranges and "total pipeline" behind the forecast
BLUE_MID = "#86b6ef"
# One fixed colour per forecasting method (never re-assigned by rank).
METHOD_COLORS = {"crm_stage_probability": "#2a78d6", "historical_stage_rate": "#eb6834",
                 "logistic_regression": "#1baf7a", "random_forest": "#6250d6"}
# Status colours are reserved for risk and always shown with a text label.
RISK_COLORS = {"High": "#d03b3b", "Medium": "#ec835a", "Low": "#0ca30c"}
STAGES = config.OPEN_STAGE_NAMES
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def inr(x: float) -> str:
    """Indian-style money: ₹54.38 cr, ₹12.4 L, ₹85,000."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if abs(x) >= CRORE:
        return f"₹{x / CRORE:,.2f} cr"
    if abs(x) >= LAKH:
        return f"₹{x / LAKH:,.1f} L"
    return f"₹{x:,.0f}"


def quarter_months(q: str) -> str:
    """'2026Q4' -> 'Oct–Dec 2026'."""
    names = ["Jan–Mar", "Apr–Jun", "Jul–Sep", "Oct–Dec"]
    return f"{names[int(q[-1]) - 1]} {q[:4]}"


CSS = f"""
<style>
.block-container {{ padding-top: 3.2rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.stMarkdown .page-title {{ font-size: 1.9rem; font-weight: 700; color: {INK}; margin: 0; line-height: 1.2; }}
.stMarkdown .page-sub {{ color: {INK_2}; margin: .35rem 0 1.4rem 0; font-size: .95rem; }}
.stMarkdown .section-title {{ font-size: 1.05rem; font-weight: 650; color: {INK}; margin: .2rem 0 .1rem 0; }}
.stMarkdown .section-note {{ color: {INK_2}; font-size: .86rem; margin: 0 0 .6rem 0; line-height: 1.45; }}
.hero {{ background: #ffffff; border: 1px solid #dce1e8; border-radius: 12px;
         padding: 1.4rem 1.6rem 1rem 1.6rem; margin-bottom: 1rem; }}
.hero-grid {{ display: grid; grid-template-columns: minmax(220px, 300px) 1fr; gap: 2rem; align-items: center; }}
.stMarkdown .hero-label {{ color: {INK_2}; font-size: .92rem; margin: 0; }}
.stMarkdown .hero-value {{ color: {INK}; font-size: 3rem; font-weight: 700; line-height: 1.05; margin: .25rem 0 .35rem 0;
               letter-spacing: -0.02em; }}
.stMarkdown .hero-sub {{ color: {INK_2}; font-size: .9rem; margin: 0; line-height: 1.5; }}
.hero svg text {{ font-family: {FONT}; }}
@media (max-width: 900px) {{ .hero-grid {{ grid-template-columns: 1fr; gap: .8rem; }} }}
.deal-list {{ list-style: none; margin: 0; padding: 0; }}
.deal-list li {{ display: grid; grid-template-columns: 1fr auto; gap: .2rem 1rem; padding: .62rem 0;
                 border-bottom: 1px solid #edf0f4; }}
.deal-list li:last-child {{ border-bottom: none; }}
.deal-name {{ color: {INK}; font-weight: 600; font-size: .92rem; }}
.deal-meta {{ color: {INK_2}; font-size: .8rem; }}
.deal-amt {{ color: {INK}; font-weight: 600; font-size: .92rem; text-align: right;
             font-variant-numeric: tabular-nums; }}
.deal-why {{ grid-column: 1 / -1; color: {INK_2}; font-size: .8rem; }}
.pill {{ display: inline-flex; align-items: center; gap: .3rem; font-size: .75rem; font-weight: 600;
         padding: .05rem .5rem; border-radius: 999px; }}
.pill .dot {{ width: .5rem; height: .5rem; border-radius: 50%; display: inline-block; }}
.method-tag {{ display: inline-block; background: #e8f1fc; color: #1c5cab; border-radius: 6px;
               padding: .1rem .5rem; font-size: .8rem; font-weight: 600; }}
[data-testid="stMetric"] {{ background: #ffffff; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem;
          margin-bottom: 1rem; }}
.stat {{ background: #ffffff; border: 1px solid #dce1e8; border-radius: 10px; padding: .85rem 1rem; }}
.stat-label {{ color: {INK_2}; font-size: .85rem; }}
.stat-value {{ color: {INK}; font-size: 1.65rem; font-weight: 600; line-height: 1.25; margin: .15rem 0; }}
.stat-note {{ color: {MUTED}; font-size: .78rem; line-height: 1.35; }}
div[class*="st-key-panel_"] {{ background: #ffffff; border-radius: 12px; padding: 1.1rem 1.2rem; }}
[data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: .85rem; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; }}
.stTabs [role="tablist"] {{ gap: 1.4rem; border-bottom: 1px solid #dce1e8; }}
.stTabs [role="tab"] {{ padding: .55rem 0; font-weight: 600; }}
</style>
"""


_panel_count = [0]


def panel():
    """A white, bordered box. The key lets CSS give it a white background."""
    _panel_count[0] += 1
    return st.container(border=True, key=f"panel_{_panel_count[0]}")


def section(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>'
                + (f'<div class="section-note">{note}</div>' if note else ""), unsafe_allow_html=True)


def stat_cards(cards: list[tuple[str, str, str]]) -> None:
    """A row of number cards that wraps on small screens. Each card: (label, value, note)."""
    html_cards = "".join(
        f'<div class="stat"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        f'<div class="stat-note">{html.escape(note)}</div></div>'
        for label, value, note in cards)
    st.markdown(f'<div class="stats">{html_cards}</div>', unsafe_allow_html=True)


def risk_pill(level: str) -> str:
    c = RISK_COLORS.get(level, MUTED)
    return (f'<span class="pill" style="background:{c}1a;color:{INK}">'
            f'<span class="dot" style="background:{c}"></span>{html.escape(level)} risk</span>')


# ======================================================================
# 2. Loading (cached so the files are read only once)
# ======================================================================

@st.cache_data
def load_outputs() -> dict:
    dq = REPORTS / "data_quality"
    out = {
        "current": pd.read_csv(REPORTS / "current_forecast.csv",
                               parse_dates=["expected_close_date", "opportunity_created_date",
                                            "last_activity_date"]),
        "periods": pd.read_csv(REPORTS / "backtest_periods.csv", parse_dates=["snapshot_date"]),
        "comparison": pd.read_csv(REPORTS / "method_comparison.csv"),
        "deal_metrics": pd.read_csv(REPORTS / "deal_level_metrics.csv"),
        "intervals": pd.read_csv(REPORTS / "scenario_backtest.csv", parse_dates=["snapshot_date"]),
        "monthly": pd.read_csv(REPORTS / "monthly_forecast.csv"),
        "rep_followup": pd.read_csv(REPORTS / "rep_followup.csv"),
        "concentration": pd.read_csv(REPORTS / "concentration.csv"),
        "dq_issues_clean": pd.read_csv(dq / "clean_issues.csv"),
        "dq_issues_sample": pd.read_csv(dq / "sample_issues.csv"),
        "detection": pd.read_csv(dq / "sample_detection.csv"),
        "quotas": pd.read_csv(config.RAW_DATA_DIR / "quotas.csv"),
        "reps": pd.read_csv(config.RAW_DATA_DIR / "sales_reps.csv"),
    }
    for name in ["scenarios", "insights", "run_info"]:
        out[name] = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
    for name in ["clean", "sample"]:
        out[f"dq_{name}"] = json.loads((dq / f"{name}_summary.json").read_text(encoding="utf-8"))
        out[f"dq_{name}_md"] = (dq / f"{name}_report.md").read_text(encoding="utf-8")
    return out


@st.cache_data
def filtered_scenarios(amounts: tuple, probs: tuple, method: str) -> dict:
    hist = pd.read_csv(REPORTS / "scenario_backtest.csv")
    return sa.build_scenarios(np.array(amounts), np.array(probs), hist, method)


# ======================================================================
# 3. Chart helpers
# ======================================================================

def styled(chart: alt.Chart, height: int = 280) -> alt.Chart:
    """Apply the dashboard's quiet chart style: recessive grid and axes, no frame."""
    return (chart.properties(height=height, background="#ffffff",
                             padding={"left": 4, "right": 12, "top": 10, "bottom": 4})
            .configure(font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=INK_2, gridColor=GRID, domainColor=BASELINE,
                            tickColor=BASELINE, labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal", labelPadding=6)
            .configure_legend(orient="top", labelColor=INK_2, labelFontSize=12, title=None,
                              symbolStrokeWidth=3, columnPadding=16, labelLimit=320)
            .configure_axisQuantitative(tickCount=6)
            .configure_title(color=INK, fontSize=14, anchor="start"))


def show(chart: alt.Chart, height: int = 280) -> None:
    st.altair_chart(styled(chart, height), width="stretch", theme=None)


def hero_range_svg(down: float, base: float, up: float, quota: float | None) -> str:
    """The forecast range strip: Downside .. Upside band, Base marker, optional quota line."""
    values = [down, base, up] + ([quota] if quota else [])
    lo, hi = min(values) * 0.9, max(values) * 1.06
    W, x0, x1 = 640, 70, 570

    def x(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    xd, xb, xu = x(down), x(base), x(up)
    parts = [f'<svg viewBox="0 0 {W} 128" width="100%" role="img" '
             f'aria-label="Forecast range from {inr(down)} to {inr(up)}, base {inr(base)}">',
             # axis line
             f'<line x1="{x0}" x2="{x1}" y1="64" y2="64" stroke="{BASELINE}" stroke-width="2"/>',
             # downside..upside band
             f'<rect x="{xd:.1f}" y="48" width="{xu - xd:.1f}" height="32" rx="6" fill="{BLUE_PALE}"/>',
             f'<line x1="{xd:.1f}" x2="{xd:.1f}" y1="44" y2="84" stroke="{BLUE_MID}" stroke-width="2"/>',
             f'<line x1="{xu:.1f}" x2="{xu:.1f}" y1="44" y2="84" stroke="{BLUE_MID}" stroke-width="2"/>',
             # base marker
             f'<rect x="{xb - 4:.1f}" y="38" width="8" height="52" rx="4" fill="{BLUE}"/>',
             f'<text x="{xb:.1f}" y="26" text-anchor="middle" font-size="15" font-weight="700" '
             f'fill="{INK}">Base {inr(base)}</text>',
             # downside / upside labels under the band
             f'<text x="{xd:.1f}" y="108" text-anchor="end" font-size="13" fill="{INK_2}">'
             f'Downside {inr(down)}</text>',
             f'<text x="{xu:.1f}" y="108" text-anchor="start" font-size="13" fill="{INK_2}">'
             f'Upside {inr(up)}</text>']
    if quota:
        xq = x(quota)
        parts.append(f'<line x1="{xq:.1f}" x2="{xq:.1f}" y1="34" y2="94" stroke="{INK}" '
                     f'stroke-width="2" stroke-dasharray="4 3"/>')
        # Put the quota label where it cannot collide with the Base label.
        near_base = abs(xq - xb) < 130
        y, anchor = (124, "middle") if near_base else (26, "middle")
        if near_base and (abs(xq - xd) < 150 or abs(xq - xu) < 150):
            y = 124
        parts.append(f'<text x="{xq:.1f}" y="{y}" text-anchor="{anchor}" font-size="13" '
                     f'fill="{INK}">Quota {inr(quota)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ======================================================================
# Start of the page
# ======================================================================
required = ["current_forecast.csv", "scenarios.json", "backtest_periods.csv"]
if not all((REPORTS / f).exists() for f in required):
    st.error("No forecast results yet. Run the pipeline first, then reload this page:\n\n"
             "`python run_pipeline.py`")
    st.stop()

st.markdown(CSS, unsafe_allow_html=True)
data = load_outputs()
current = data["current"]
scen = data["scenarios"]
run_info = data["run_info"]
quarter = scen["forecast_quarter"]
method = scen["method"]

# ======================================================================
# 4. Sidebar filters (apply to the open pipeline)
# ======================================================================
with st.sidebar:
    st.header("Filters")
    st.caption("These filter the open deals on every tab except Data quality.")
    min_d = current["expected_close_date"].min().date()
    max_d = current["expected_close_date"].max().date()
    date_range = st.date_input("Expected close date", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
    regions = st.multiselect("Region", sorted(current["region"].unique()), placeholder="All regions")
    rep_options = (current[["rep_id", "rep_name"]].drop_duplicates().sort_values("rep_id")
                   .apply(lambda r: f"{r.rep_id} - {r.rep_name}", axis=1).tolist())
    reps = st.multiselect("Sales rep", rep_options, placeholder="All reps")
    stages = st.multiselect("Stage", [s for s in STAGES if s in set(current["stage"])],
                            placeholder="All stages")

view = current.copy()
date_filtered = False
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    date_filtered = (start.date(), end.date()) != (min_d, max_d)
    view = view[(view["expected_close_date"] >= start) & (view["expected_close_date"] <= end)]
if regions:
    view = view[view["region"].isin(regions)]
rep_ids = [r.split(" - ")[0] for r in reps]
if reps:
    view = view[view["rep_id"].isin(rep_ids)]
if stages:
    view = view[view["stage"].isin(stages)]
is_filtered = len(view) != len(current)

with st.sidebar:
    st.divider()
    st.markdown(f"**{len(view):,}** of {len(current):,} open deals shown")
    st.caption(f"Synthetic data, scale *{run_info['scale']}*, seed {run_info['seed']}. "
               "Numbers do not describe a real company.")


def quota_for_view() -> float | None:
    """Quota of the reps in scope. Not shown when stage/date filters make it meaningless."""
    if stages or date_filtered:
        return None
    q = data["quotas"]
    q = q[q["quota_quarter"] == quarter]
    if reps:
        q = q[q["rep_id"].isin(rep_ids)]
    elif regions:
        q = q[q["rep_id"].isin(data["reps"].loc[data["reps"]["region"].isin(regions), "rep_id"])]
    total = float(q["quota_amount"].sum())
    return total or None


# ======================================================================
# Header
# ======================================================================
st.markdown(f'<div class="page-title">Bookings forecast for {quarter_months(quarter)}</div>'
            f'<div class="page-sub">Made on {pd.Timestamp(scen["snapshot_date"]):%d %b %Y} from '
            f'{len(current):,} open deals. Method: '
            f'<span class="method-tag">{METHOD_LABELS[method]}</span> '
            f'(lowest error in the backtest).</div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Forecast analysis", "Pipeline analysis", "Risks",
                "Data quality", "Opportunity explorer"])

# ======================================================================
# 5a. Overview
# ======================================================================
with tabs[0]:
    if view.empty:
        st.info("No open deals match these filters. Clear a filter in the sidebar to see results.")
    else:
        s = (filtered_scenarios(tuple(view["deal_amount"]), tuple(view["forecast_probability"]), method)
             if is_filtered else scen)
        quota = quota_for_view()
        if quota:
            cover = s["base"] / quota
            quota_line = (f"Quota for these reps is {inr(quota)}; the Base forecast covers "
                          f"<b>{cover:.0%}</b> of it.")
        else:
            quota_line = "Quota comparison is hidden while a stage or date filter is on."
        st.markdown(
            f'<div class="hero"><div class="hero-grid"><div>'
            f'<div class="hero-label">Expected bookings, {quarter_months(quarter)}</div>'
            f'<div class="hero-value">{inr(s["base"])}</div>'
            f'<div class="hero-sub">Likely range {inr(s["downside"])} to {inr(s["upside"])}.<br>'
            f'{quota_line}</div></div>'
            f'<div>{hero_range_svg(s["downside"], s["base"], s["upside"], quota)}</div>'
            f'</div></div>', unsafe_allow_html=True)

        stat_cards([
            ("Open pipeline", inr(view["deal_amount"].sum()), "All open deals, before win chances"),
            ("Open opportunities", f"{len(view):,}", "Deals not yet won or lost"),
            ("Stale opportunities", f"{int(view['flag_stale'].sum()):,}",
             f"No calls, emails or meetings for {config.STALE_ACTIVITY_DAYS}+ days"),
            ("High-risk deals", f"{int((view['risk_level'] == 'High').sum()):,}",
             "Two or more risk flags, or large but unlikely")])

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            with panel():
                section("Where the Base forecast comes from",
                        "Expected bookings by stage. Pale bar = full deal value, "
                        "blue = value × win chance.")
                by_stage = (view.groupby("stage")
                            .agg(pipeline=("deal_amount", "sum"), expected=("expected_booking", "sum"),
                                 deals=("opportunity_id", "count"))
                            .reindex([x for x in STAGES if x in set(view["stage"])]).reset_index())
                by_stage["pipeline_cr"] = by_stage["pipeline"] / CRORE
                by_stage["expected_cr"] = by_stage["expected"] / CRORE
                by_stage["label"] = by_stage["expected"].map(inr)
                tip = [alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("deals:Q", title="Deals"),
                       alt.Tooltip("pipeline_cr:Q", title="Pipeline (₹ cr)", format=",.2f"),
                       alt.Tooltip("expected_cr:Q", title="Expected (₹ cr)", format=",.2f")]
                y = alt.Y("stage:N", sort=STAGES, title=None, axis=alt.Axis(labelColor=INK_2, labelFontSize=13))
                base_c = alt.Chart(by_stage).encode(y=y, tooltip=tip)
                show(base_c.mark_bar(color=BLUE_PALE, cornerRadiusEnd=4, height=26)
                     .encode(x=alt.X("pipeline_cr:Q", title="₹ crore"))
                     + base_c.mark_bar(color=BLUE, cornerRadiusEnd=4, height=12)
                     .encode(x="expected_cr:Q")
                     + base_c.mark_text(align="left", dx=6, color=INK, fontWeight=600, fontSize=12)
                     .encode(x="expected_cr:Q", text="label:N"), height=300)
        with right:
            with panel():
                section("Deals to look at first",
                        "High-risk deals, largest first. Full list in the Risks tab.")
                top = view[view["risk_level"] == "High"].nlargest(4, "deal_amount")
                if top.empty:
                    st.success("No high-risk deals in this selection.")
                else:
                    items = "".join(
                        f'<li><div><div class="deal-name">{html.escape(r.account_name)}</div>'
                        f'<div class="deal-meta">{r.opportunity_id}, {html.escape(r.stage)}, '
                        f'{html.escape(r.rep_name)}</div></div>'
                        f'<div class="deal-amt">{inr(r.deal_amount)}<br>{risk_pill(r.risk_level)}</div>'
                        f'<div class="deal-why">{html.escape(r.risk_reasons)}</div></li>'
                        for r in top.itertuples())
                    st.markdown(f'<ul class="deal-list">{items}</ul>', unsafe_allow_html=True)

        with st.expander("How the Downside, Base and Upside are calculated"):
            for a in s["assumptions"]:
                st.markdown("- " + a)
            st.markdown(f"- In past quarters the simulation's range contained the real result "
                        f"{s['simulation_coverage']:.0%} of the time ({s['backtest_periods']} quarters), "
                        "so the range is widened using past forecast errors.")
            st.caption(s["limitation"])

# ======================================================================
# 5b. Forecast analysis
# ======================================================================
with tabs[1]:
    periods = data["periods"]
    with panel():
        section("Backtest: what each method forecast vs what actually happened",
                "Each point is a forecast made at the end of one quarter for the next quarter, "
                "using only data available at the time. Black = actual bookings.")
        default = [m for m in METHODS if m != "historical_stage_rate"]
        chosen = st.multiselect("Methods to show", METHODS, default=default,
                                format_func=lambda m: METHOD_LABELS[m], label_visibility="collapsed")
        long = periods.melt(id_vars=["forecast_quarter"], value_vars=["actual"] + chosen,
                            var_name="series", value_name="value").dropna()
        long["value_cr"] = long["value"] / CRORE
        long["name"] = long["series"].map({**METHOD_LABELS, "actual": "Actual bookings"})
        names = ["Actual bookings"] + [METHOD_LABELS[m] for m in chosen]
        colors = [INK] + [METHOD_COLORS[m] for m in chosen]
        color = alt.Color("name:N", scale=alt.Scale(domain=names, range=colors), sort=names)
        hover = alt.selection_point(fields=["forecast_quarter"], nearest=True, on="pointerover",
                                    empty=False)
        base_c = alt.Chart(long).encode(x=alt.X("forecast_quarter:O", title=None,
                                                axis=alt.Axis(labelAngle=0)))
        lines = base_c.mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=55, filled=True)).encode(
            y=alt.Y("value_cr:Q", title="₹ crore", scale=alt.Scale(zero=False)), color=color,
            strokeDash=alt.condition(alt.datum.series == "actual", alt.value([1, 0]), alt.value([5, 3])))
        wide = periods[["forecast_quarter", "actual"] + chosen].copy()
        tips = [alt.Tooltip("forecast_quarter:O", title="Quarter"),
                alt.Tooltip("actual:Q", title="Actual (₹)", format=",.0f")]
        tips += [alt.Tooltip(f"{m}:Q", title=METHOD_LABELS[m], format=",.0f") for m in chosen]
        rule = (alt.Chart(wide).mark_rule(color=BASELINE, strokeWidth=1.5)
                .encode(x="forecast_quarter:O", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                        tooltip=tips).add_params(hover))
        show(lines + rule, height=320)

    comp = data["comparison"].copy()
    comp["Used for forecast"] = np.where(comp["method"] == method, "✓ yes", "")
    comp["Method"] = comp["method"].map(METHOD_LABELS)
    comp["MAE (₹ cr)"] = comp["MAE"] / CRORE
    comp["RMSE (₹ cr)"] = comp["RMSE"] / CRORE
    comp["WAPE"] = comp["WAPE"] * 100
    comp["Bias"] = comp["Bias"] * 100
    dm = data["deal_metrics"].copy()
    dm["Method"] = dm["method"].map(METHOD_LABELS)

    with panel():
        section("Accuracy of the quarterly total",
                "Lower WAPE is better. Bias above 0 means the method forecast too high.")
        st.dataframe(comp[["Method", "WAPE", "Bias", "MAE (₹ cr)", "RMSE (₹ cr)", "Used for forecast"]],
                     hide_index=True, width="stretch", column_config={
                         "WAPE": st.column_config.NumberColumn(format="%.1f%%"),
                         "Bias": st.column_config.NumberColumn(format="%+.1f%%"),
                         "MAE (₹ cr)": st.column_config.NumberColumn(format="%.2f"),
                         "RMSE (₹ cr)": st.column_config.NumberColumn(format="%.2f")})
        st.write("")
        section("Accuracy on individual deals",
                f"Lower Brier and higher ROC AUC are better. Risk flags use "
                f"{METHOD_LABELS[run_info['risk_method']]}.")
        st.dataframe(dm[["Method", "Brier", "LogLoss", "ROC_AUC", "deals"]], hide_index=True,
                     width="stretch", column_config={
                         "Brier": st.column_config.NumberColumn(format="%.3f"),
                         "LogLoss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                         "ROC_AUC": st.column_config.NumberColumn("ROC AUC", format="%.3f"),
                         "deals": st.column_config.NumberColumn("Deals tested")})

    a, b = st.columns(2, gap="large")
    with a:
        with panel():
            section(f"{quarter_months(quarter)} forecast by month",
                    "Expected bookings by expected close month. Overdue deals count in the first month.")
            mon = data["monthly"].copy()
            mon["label"] = mon["month"].map(
                lambda m: pd.Timestamp(m + "-01").strftime("%b %Y") if m[:2] == "20" else m)
            mon["expected_cr"] = mon["expected_booking"] / CRORE
            mon = mon[mon["expected_booking"] > 0]
            order = mon["label"].tolist()
            bars = alt.Chart(mon).encode(
                x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                tooltip=[alt.Tooltip("label:N", title="Month"), alt.Tooltip("deals:Q", title="Deals"),
                         alt.Tooltip("expected_cr:Q", title="Expected (₹ cr)", format=",.2f")])
            show(bars.mark_bar(color=BLUE, cornerRadiusEnd=4, width=56)
                 .encode(y=alt.Y("expected_cr:Q", title="₹ crore"))
                 + bars.mark_text(dy=-8, color=INK, fontWeight=600)
                 .encode(y="expected_cr:Q", text=alt.Text("expected_cr:Q", format=",.1f")), height=260)
    with b:
        with panel():
            iv = data["intervals"].copy()
            inside = int(iv["inside_P10_P90"].sum())
            section("Did the real result land inside the simulated range?",
                    f"Blue band = simulation P10 to P90. Inside in {inside} of {len(iv)} quarters "
                    "(ideally about 80%), which is why the final range is widened.")
            iv["quarter"] = (iv["snapshot_date"] + pd.offsets.QuarterEnd(1)).map(
                lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}")
            for col in ["P10", "P90", "actual", "expected"]:
                iv[col + "_cr"] = iv[col] / CRORE
            iv["result"] = np.where(iv["inside_P10_P90"], "Inside range", "Outside range")
            bc = alt.Chart(iv).encode(x=alt.X("quarter:O", title=None, axis=alt.Axis(labelAngle=0)))
            tip = [alt.Tooltip("quarter:O", title="Quarter"),
                   alt.Tooltip("P10_cr:Q", title="P10 (₹ cr)", format=",.2f"),
                   alt.Tooltip("P90_cr:Q", title="P90 (₹ cr)", format=",.2f"),
                   alt.Tooltip("actual_cr:Q", title="Actual (₹ cr)", format=",.2f"),
                   alt.Tooltip("result:N", title="Result")]
            show(bc.mark_bar(color=BLUE_PALE, width=26, cornerRadius=4)
                 .encode(y=alt.Y("P10_cr:Q", title="₹ crore", scale=alt.Scale(zero=False)), y2="P90_cr:Q",
                         tooltip=tip)
                 + bc.mark_point(size=90, strokeWidth=2.5).encode(
                     y="actual_cr:Q", tooltip=tip,
                     shape=alt.Shape("result:N", scale=alt.Scale(domain=["Inside range", "Outside range"],
                                                                range=["circle", "diamond"])),
                     color=alt.Color("result:N", scale=alt.Scale(domain=["Inside range", "Outside range"],
                                                                range=[INK, "#d03b3b"])),
                     fill=alt.Color("result:N", legend=None,
                                    scale=alt.Scale(domain=["Inside range", "Outside range"],
                                                    range=[INK, "#ffffff"]))), height=260)

# ======================================================================
# 5c. Pipeline analysis
# ======================================================================
with tabs[2]:
    if view.empty:
        st.info("No open deals match these filters.")
    else:
        with panel():
            section("Regions: forecast against quota",
                    "Pale bar = open pipeline, blue = expected bookings, black tick = "
                    f"{quarter} quota. Values are queried from the SQLite database.")
            ids = tuple(view["opportunity_id"])
            if config.DATABASE_PATH.exists():
                region_tbl = database.query(
                    "SELECT region, COUNT(*) AS deals, SUM(deal_amount) AS pipeline, "
                    "SUM(expected_booking) AS expected FROM current_forecast "
                    f"WHERE opportunity_id IN ({','.join('?' * len(ids))}) GROUP BY region", ids)
            else:
                region_tbl = (view.groupby("region").agg(deals=("opportunity_id", "count"),
                                                         pipeline=("deal_amount", "sum"),
                                                         expected=("expected_booking", "sum")).reset_index())
            q = data["quotas"]
            q = q[q["quota_quarter"] == quarter].merge(data["reps"][["rep_id", "region"]], on="rep_id")
            if reps:
                q = q[q["rep_id"].isin(rep_ids)]
            region_tbl = region_tbl.merge(q.groupby("region")["quota_amount"].sum().rename("quota"),
                                          left_on="region", right_index=True, how="left")
            for col in ["pipeline", "expected", "quota"]:
                region_tbl[col + "_cr"] = region_tbl[col] / CRORE
            region_tbl["coverage"] = region_tbl["expected"] / region_tbl["quota"]
            region_tbl = region_tbl.sort_values("expected", ascending=False)
            order = region_tbl["region"].tolist()
            tip = [alt.Tooltip("region:N", title="Region"), alt.Tooltip("deals:Q", title="Open deals"),
                   alt.Tooltip("pipeline_cr:Q", title="Pipeline (₹ cr)", format=",.2f"),
                   alt.Tooltip("expected_cr:Q", title="Expected (₹ cr)", format=",.2f"),
                   alt.Tooltip("quota_cr:Q", title="Quota (₹ cr)", format=",.2f"),
                   alt.Tooltip("coverage:Q", title="Forecast / quota", format=".0%")]
            rc = alt.Chart(region_tbl).encode(y=alt.Y("region:N", sort=order, title=None,
                                                      axis=alt.Axis(labelColor=INK_2, labelFontSize=13)),
                                              tooltip=tip)
            layers = (rc.mark_bar(color=BLUE_PALE, height=26, cornerRadiusEnd=4)
                      .encode(x=alt.X("pipeline_cr:Q", title="₹ crore"))
                      + rc.mark_bar(color=BLUE, height=12, cornerRadiusEnd=4).encode(x="expected_cr:Q"))
            if not stages and not date_filtered:
                layers = layers + rc.mark_tick(color=INK, thickness=3, size=34).encode(x="quota_cr:Q")
            show(layers, height=60 * len(region_tbl) + 40)

        a, b = st.columns(2, gap="large")
        with a:
            with panel():
                section("Open deals by stage", "Bar length = pipeline value; hover for counts.")
                sd = (view.groupby("stage").agg(value=("deal_amount", "sum"), deals=("opportunity_id", "count"))
                      .reset_index())
                sd["value_cr"] = sd["value"] / CRORE
                sc = alt.Chart(sd).encode(
                    x=alt.X("stage:N", sort=STAGES, title=None, axis=alt.Axis(labelAngle=0)),
                    tooltip=[alt.Tooltip("stage:N", title="Stage"), alt.Tooltip("deals:Q", title="Deals"),
                             alt.Tooltip("value_cr:Q", title="Pipeline (₹ cr)", format=",.2f")])
                show(sc.mark_bar(color=BLUE, cornerRadiusEnd=4, width=48)
                     .encode(y=alt.Y("value_cr:Q", title="₹ crore"))
                     + sc.mark_text(dy=-8, color=INK, fontWeight=600)
                     .encode(y="value_cr:Q", text=alt.Text("deals:Q", format=",")), height=260)
                st.caption("Numbers above the bars are deal counts.")
        with b:
            with panel():
                section("How old are the open deals?", "Days since the deal was created.")
                bins = [-1, 30, 60, 90, 180, 10_000]
                labels = ["0–30 days", "31–60", "61–90", "91–180", "180+"]
                ag = view.assign(age=pd.cut(view["opportunity_age_days"], bins=bins, labels=labels))
                ag = (ag.groupby("age", observed=False)
                      .agg(value=("deal_amount", "sum"), deals=("opportunity_id", "count")).reset_index())
                ag["age"] = ag["age"].astype(str)
                ag["value_cr"] = ag["value"] / CRORE
                acn = alt.Chart(ag).encode(
                    x=alt.X("age:N", sort=labels, title=None, axis=alt.Axis(labelAngle=0)),
                    tooltip=[alt.Tooltip("age:N", title="Age"), alt.Tooltip("deals:Q", title="Deals"),
                             alt.Tooltip("value_cr:Q", title="Pipeline (₹ cr)", format=",.2f")])
                show(acn.mark_bar(color=BLUE, cornerRadiusEnd=4, width=48)
                     .encode(y=alt.Y("value_cr:Q", title="₹ crore"))
                     + acn.mark_text(dy=-8, color=INK, fontWeight=600)
                     .encode(y="value_cr:Q", text=alt.Text("deals:Q", format=",")), height=260)
                st.caption("Numbers above the bars are deal counts.")

        with panel():
            section("Pipeline by sales rep", "Top 15 reps by open pipeline in this selection.")
            rep_tbl = (view.groupby(["rep_id", "rep_name", "region"])
                       .agg(open_deals=("opportunity_id", "count"), pipeline=("deal_amount", "sum"),
                            expected=("expected_booking", "sum"),
                            flagged=("risk_flag_count", lambda s: int((s > 0).sum())))
                       .reset_index().sort_values("pipeline", ascending=False).head(15))
            rep_tbl["pipeline_l"] = rep_tbl["pipeline"] / LAKH
            rep_tbl["expected_l"] = rep_tbl["expected"] / LAKH
            rep_tbl["flagged_share_pct"] = rep_tbl["flagged"] / rep_tbl["open_deals"] * 100
            st.dataframe(rep_tbl[["rep_id", "rep_name", "region", "open_deals", "pipeline_l", "expected_l",
                                  "flagged_share_pct"]], hide_index=True, width="stretch", column_config={
                "rep_id": "Rep ID", "rep_name": "Name", "region": "Region",
                "open_deals": st.column_config.NumberColumn("Open deals"),
                "pipeline_l": st.column_config.NumberColumn("Pipeline (₹ lakh)", format="%.1f"),
                "expected_l": st.column_config.NumberColumn("Expected (₹ lakh)", format="%.1f"),
                "flagged_share_pct": st.column_config.ProgressColumn(
                    "Deals with a risk flag", min_value=0, max_value=100, format="%.0f%%")})

# ======================================================================
# 5d. Risks
# ======================================================================
with tabs[3]:
    if view.empty:
        st.info("No open deals match these filters.")
    else:
        short = {"flag_stale": "Stale", "flag_overdue": "Overdue",
                 "flag_large_low_prob": "Large but unlikely", "flag_slip_risk": "Likely to slip"}
        stat_cards([(short[f], f"{int(view[f].sum()):,} deals",
                     f"{FLAG_LABELS[f]}, worth {inr(view.loc[view[f], 'deal_amount'].sum())}")
                    for f in FLAG_LABELS])
        sb = data["insights"].get("slip_rule_backtest")
        if sb:
            st.caption(f"The slip-risk rule was checked on past quarters: flagged deals slipped "
                       f"{sb['flagged_slip_rate']:.0%} of the time, against {sb['overall_slip_rate']:.0%} "
                       "for all deals expected in the quarter.")

        with panel():
            section("Flagged deals", "Every flagged open deal, with the reasons. Largest first.")
            levels = st.segmented_control("Risk level", ["High", "Medium"], default=["High", "Medium"],
                                          selection_mode="multi", label_visibility="collapsed")
            fl = view[view["risk_level"].isin(levels or [])].sort_values("deal_amount", ascending=False)
            fl = fl.assign(risk=fl["risk_level"].map(lambda v: {"High": "▲ High", "Medium": "■ Medium"}[v]),
                           amount_l=fl["deal_amount"] / LAKH, win_pct=fl["win_probability"] * 100)
            st.dataframe(fl[["risk", "opportunity_id", "account_name", "amount_l", "win_pct",
                             "risk_reasons"]],
                         hide_index=True, width="stretch", height=380, column_config={
                             "risk": st.column_config.TextColumn("Risk", width="small"),
                             "opportunity_id": "Opportunity", "account_name": "Account",
                             "amount_l": st.column_config.NumberColumn("Deal (₹ lakh)", format="%.1f"),
                             "win_pct": st.column_config.ProgressColumn(
                                 "Win chance", min_value=0, max_value=100, format="%.0f%%",
                                 help="Chance of being won this quarter (risk model)"),
                             "risk_reasons": st.column_config.TextColumn("Why", width="large")})

        a, b = st.columns(2, gap="large")
        with a:
            with panel():
                section("Pipeline concentration",
                        f"Accounts and reps holding the largest share of open pipeline (all deals). "
                        f"Above {config.CONCENTRATION_SHARE:.0%} is marked as a risk.")
                conc = data["concentration"].copy()
                conc["What"] = conc["dimension"].map({"account_id": "Account", "rep_id": "Rep",
                                                      "top_10_deals": "Top 10 deals"})
                conc["Risk"] = np.where(conc["above_threshold"], "▲ Above limit", "")
                conc["share_of_pipeline_pct"] = conc["share_of_pipeline"] * 100
                st.dataframe(conc[["What", "name", "share_of_pipeline_pct", "open_deals", "Risk", "opportunity_ids"]],
                             hide_index=True, width="stretch", column_config={
                                 "name": "Name",
                                 "share_of_pipeline_pct": st.column_config.ProgressColumn(
                                     "Share of pipeline", min_value=0, max_value=100, format="%.0f%%"),
                                 "open_deals": "Deals", "opportunity_ids": "Opportunity IDs"})
        with b:
            with panel():
                section("Reps where a pipeline review is suggested",
                        "At least 4 open deals and half or more flagged. This describes their deals, "
                        "not the person's performance.")
                rf = data["rep_followup"].assign(
                    flagged_share_pct=lambda d: d["flagged_share"] * 100)
                st.dataframe(rf[["rep_id", "rep_name", "region", "open_deals", "flagged_deals",
                                 "flagged_share_pct", "flagged_opportunity_ids"]],
                             hide_index=True, width="stretch", column_config={
                                 "rep_id": "Rep ID", "rep_name": "Name", "region": "Region",
                                 "open_deals": "Open", "flagged_deals": "Flagged",
                                 "flagged_share_pct": st.column_config.ProgressColumn(
                                     "Share flagged", min_value=0, max_value=100, format="%.0f%%"),
                                 "flagged_opportunity_ids": "Flagged opportunity IDs"})

# ======================================================================
# 5e. Data quality
# ======================================================================
with tabs[4]:
    st.caption("Data quality is checked on the whole dataset, so the sidebar filters do not apply here.")
    a, b = st.columns(2, gap="large")
    for col, key, title, note in [
            (a, "clean", "Main dataset", "The clean data used for forecasting."),
            (b, "sample", "Test sample with planted errors",
             "A small copy with 16 deliberate mistakes, to prove the checks work.")]:
        summ = data[f"dq_{key}"]
        with col, panel():
            section(title, note)
            m = st.columns(3)
            m[0].metric("Rows checked", f"{summ['total_rows_checked']:,}")
            m[1].metric("Errors", summ["errors"])
            m[2].metric("Warnings", summ["warnings"])
            if summ["categories"]:
                cats = pd.DataFrame(summ["categories"])
                cats["name"] = cats["issue_type"].str.replace("_", " ").str.capitalize()
                cc = alt.Chart(cats).encode(
                    y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK_2, labelLimit=260)),
                    tooltip=[alt.Tooltip("name:N", title="Issue"), alt.Tooltip("count:Q", title="Count"),
                             alt.Tooltip("recommendation:N", title="How to fix")])
                show(cc.mark_bar(color="#d03b3b", height=14, cornerRadiusEnd=4)
                     .encode(x=alt.X("count:Q", title="Issues found", axis=alt.Axis(tickMinStep=1))),
                     height=24 * len(cats) + 40)
            else:
                st.success("No problems found in this dataset.")
            d1, d2 = st.columns(2)
            d1.download_button("Report (.md)", data[f"dq_{key}_md"],
                               file_name=f"{key}_data_quality_report.md", key=f"md_{key}",
                               icon=":material/download:", width="stretch")
            d2.download_button("Issue list (.csv)", data[f"dq_issues_{key}"].to_csv(index=False),
                               file_name=f"{key}_issues.csv", key=f"csv_{key}",
                               icon=":material/download:", width="stretch")

    det = data["detection"].copy()
    with panel():
        section("Were all planted errors found?",
                f"{int(det['detected'].sum())} of {len(det)} planted errors were detected by the checks.")
        det["Found"] = np.where(det["detected"], "✓ Found", "✗ Missed")
        st.dataframe(det[["Found", "issue_type", "table_name", "record_id", "description"]],
                     hide_index=True, width="stretch", column_config={
                         "issue_type": "Error type", "table_name": "Table", "record_id": "Record",
                         "description": st.column_config.TextColumn("What was broken", width="large")})

# ======================================================================
# 5f. Opportunity explorer
# ======================================================================
with tabs[5]:
    c1, c2 = st.columns([3, 2])
    search = c1.text_input("Search", placeholder="Opportunity ID, account or rep name",
                           label_visibility="collapsed")
    risk_only = c2.toggle("Only deals with a risk flag")
    tbl = view
    if search:
        s_ = search.lower()
        tbl = tbl[tbl["opportunity_id"].str.lower().str.contains(s_, regex=False)
                  | tbl["account_name"].str.lower().str.contains(s_, regex=False)
                  | tbl["rep_name"].str.lower().str.contains(s_, regex=False)]
    if risk_only:
        tbl = tbl[tbl["risk_flag_count"] > 0]
    tbl = tbl.sort_values("expected_booking", ascending=False).assign(
        amount_l=lambda d: d["deal_amount"] / LAKH,
        expected_l=lambda d: d["expected_booking"] / LAKH,
        risk=lambda d: d["risk_level"].map({"High": "▲ High", "Medium": "■ Medium", "Low": "● Low"}),
        prob_pct=lambda d: d["forecast_probability"] * 100)
    st.caption(f"{len(tbl):,} deals. Click a column header to sort.")
    cols = ["opportunity_id", "account_name", "rep_name", "stage", "amount_l",
            "prob_pct", "expected_l", "expected_close_date", "last_activity_date",
            "days_since_last_activity", "risk", "risk_reasons"]
    st.dataframe(tbl[cols], hide_index=True, width="stretch", height=520, column_config={
        "opportunity_id": st.column_config.TextColumn("Opportunity", pinned=True),
        "account_name": "Account", "rep_name": "Rep", "stage": "Stage",
        "amount_l": st.column_config.NumberColumn("Deal (₹ lakh)", format="%.1f"),
        "prob_pct": st.column_config.ProgressColumn(
            "Probability", min_value=0, max_value=100, format="%.0f%%",
            help="Probability used for the Base forecast"),
        "expected_l": st.column_config.NumberColumn("Expected (₹ lakh)", format="%.1f",
                                                    help="Deal amount × probability"),
        "expected_close_date": st.column_config.DateColumn("Expected close", format="DD MMM YYYY"),
        "last_activity_date": st.column_config.DateColumn("Last activity", format="DD MMM YYYY"),
        "days_since_last_activity": st.column_config.NumberColumn(
            "Days quiet", help="Days since the last call, email or meeting"),
        "risk": "Risk", "risk_reasons": st.column_config.TextColumn("Risk reasons", width="large")})
    st.download_button("Download these deals (.csv)", tbl[cols].to_csv(index=False),
                       file_name="opportunities.csv", icon=":material/download:")
