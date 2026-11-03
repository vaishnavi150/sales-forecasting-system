"""
run_pipeline.py - runs the WHOLE project, step by step, with one command.

    python run_pipeline.py                  (starter scale)
    python run_pipeline.py --scale full     (full scale)
    python run_pipeline.py --skip-generate  (reuse existing data)

Steps
-----
 1. Generate synthetic data (+ error sample + ground truth)
 2. Validate clean data and the error sample, write data-quality reports
 3. Build historical snapshots + features (only past information)
 4. Backtest: baselines vs ML models, quarter by quarter
 5. Choose methods using the backtest metrics
 6. Live forecast for the next quarter + Downside/Base/Upside scenarios
 7. Business insights and risk flags
 8. Save everything to reports/, data/processed/ and the SQLite database
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import business_insights as bi
from src import data_validation as dv
from src import database, evaluation, forecasting
from src import scenario_analysis as sa
from src.data_generation import load_tables, run_generation
from src.snapshots import build_all_snapshots, horizon_end

CRORE = 1e7


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    return obj


def make_figures(periods, scen, stage_dist, best_method) -> None:
    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forecast vs actual per quarter
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(periods))
    ax.plot(x, periods["actual"] / CRORE, "o-", color="black", label="Actual", linewidth=2)
    for m in forecasting.METHODS:
        ax.plot(x, periods[m] / CRORE, "o--", label=forecasting.METHOD_LABELS[m], alpha=0.8)
    ax.set_xticks(x, periods["forecast_quarter"])
    ax.set_ylabel("Bookings (INR crore)")
    ax.set_title("Backtest: forecast vs actual bookings (synthetic data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "forecast_vs_actual.png", dpi=120)
    plt.close(fig)

    # 2. Scenarios
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [scen["downside"], scen["base"], scen["upside"]]
    bars = ax.bar(["Downside", "Base", "Upside"], np.array(vals) / CRORE,
                  color=["#c0504d", "#4f81bd", "#9bbb59"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / CRORE, f"{v / CRORE:,.1f}", ha="center", va="bottom")
    ax.set_ylabel("INR crore")
    ax.set_title(f"Next-quarter scenarios ({forecasting.METHOD_LABELS[best_method]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "scenarios.png", dpi=120)
    plt.close(fig)

    # 3. Open pipeline by stage
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stage_dist.index, stage_dist.values / CRORE, color="#4f81bd")
    ax.set_ylabel("Open pipeline (INR crore)")
    ax.set_title("Open pipeline by stage")
    fig.tight_layout()
    fig.savefig(fig_dir / "pipeline_by_stage.png", dpi=120)
    plt.close(fig)


def monthly_breakdown(flagged: pd.DataFrame, live_date) -> pd.DataFrame:
    """Expected bookings of the forecast quarter split by expected close month."""
    h_end = horizon_end(live_date)
    start = pd.Timestamp(live_date) + pd.Timedelta(1, unit="D")
    close = flagged["expected_close_date"].clip(lower=start)  # overdue -> first month
    month = close.dt.to_period("M").astype(str)
    month = month.where(close <= h_end, "Later than quarter")
    return (flagged.assign(month=month).groupby("month")
            .agg(expected_booking=("expected_booking", "sum"), deals=("opportunity_id", "count"))
            .reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full sales forecasting pipeline.")
    parser.add_argument("--scale", choices=list(config.SCALES), default=None)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    cfg = config.get_config(args.scale)
    config.ensure_directories()
    started = time.time()
    reports = config.REPORTS_DIR

    # 1 -------------------------------------------------------------------
    if not args.skip_generate:
        step(f"1. Generating synthetic data (scale={cfg.scale.name}, seed={cfg.seed})")
        manifest = run_generation(cfg.scale, cfg.seed)
        for k, v in manifest["row_counts"].items():
            print(f"   {k:<24}{v:>9,}")
    else:
        step("1. Skipping generation (using existing files)")

    # 2 -------------------------------------------------------------------
    step("2. Validating data")
    dq_dir = reports / "data_quality"
    clean_raw = dv.load_for_validation(config.RAW_DATA_DIR)
    clean_issues = dv.validate_tables(clean_raw)
    clean_summary = dv.build_summary(clean_issues, clean_raw, "clean dataset (data/raw)")
    dv.write_report(clean_issues, clean_summary, dq_dir, "clean")
    print(f"   Clean data : {clean_summary['total_rows_checked']:,} rows, "
          f"{clean_summary['errors']} errors, {clean_summary['warnings']} warnings")
    sample_raw = dv.load_for_validation(config.SAMPLE_DATA_DIR)
    sample_issues = dv.validate_tables(sample_raw)
    sample_summary = dv.build_summary(sample_issues, sample_raw, "error sample (data/sample)")
    dv.write_report(sample_issues, sample_summary, dq_dir, "sample")
    labels = pd.read_csv(config.GROUND_TRUTH_DIR / "anomaly_labels.csv")
    detection = dv.compare_with_labels(sample_issues, labels)
    detection.to_csv(dq_dir / "sample_detection.csv", index=False)
    print(f"   Error sample: {sample_summary['errors']} errors, {sample_summary['warnings']} warnings; "
          f"injected anomalies detected: {int(detection['detected'].sum())}/{len(detection)}")

    # 3 -------------------------------------------------------------------
    step("3. Building snapshots and features")
    tables = load_tables(config.RAW_DATA_DIR)
    tables.update(load_tables(config.GROUND_TRUTH_DIR, ["opportunity_outcomes"]))
    snapshots = build_all_snapshots(tables, config.SNAPSHOT_DATES + [pd.Timestamp(config.LIVE_SNAPSHOT_DATE)])
    snap_issues = dv.validate_tables({"forecast_snapshots": snapshots})
    print(f"   {len(snapshots):,} snapshot rows; time-consistency issues: {len(snap_issues)}")
    snapshots.to_csv(config.PROCESSED_DATA_DIR / "forecast_snapshots.csv", index=False,
                     date_format="%Y-%m-%d")
    frames = forecasting.prepare_frames(tables, config.SNAPSHOT_DATES)

    # 4 -------------------------------------------------------------------
    step("4. Backtesting baselines and ML models")
    deals, periods = forecasting.backtest(frames, cfg.seed)
    deals.to_csv(config.PROCESSED_DATA_DIR / "backtest_deal_predictions.csv", index=False,
                 date_format="%Y-%m-%d")
    periods.to_csv(reports / "backtest_periods.csv", index=False, date_format="%Y-%m-%d")
    show = periods[["forecast_quarter", "actual"] + forecasting.METHODS].copy()
    for c in ["actual"] + forecasting.METHODS:
        show[c] = (show[c] / CRORE).round(2)
    print("   Forecast vs actual (INR crore):")
    print(show.to_string(index=False))

    # 5 -------------------------------------------------------------------
    step("5. Comparing methods")
    best_method, comparison = forecasting.choose_best_method(periods)
    risk_method, deal_metrics = forecasting.choose_risk_model(deals)
    comparison.to_csv(reports / "method_comparison.csv", index=False)
    deal_metrics.to_csv(reports / "deal_level_metrics.csv", index=False)
    seg = forecasting.segment_results(deals, "region")
    seg_metrics = []
    for region, g in seg.dropna().groupby("region"):
        for m in forecasting.METHODS:
            seg_metrics.append({"region": region, "method": m,
                                **evaluation.aggregate_metrics(g[m], g["actual"])})
    pd.DataFrame(seg_metrics).to_csv(reports / "segment_metrics_by_region.csv", index=False)
    print("   Quarterly totals (lower WAPE is better):")
    print(comparison.assign(MAE_cr=comparison["MAE"] / CRORE)
          [["method", "MAE_cr", "WAPE", "Bias", "periods"]].round(3).to_string(index=False))
    print("   Individual deals (lower Brier is better, higher AUC is better):")
    print(deal_metrics.round(3).to_string(index=False))
    print(f"   -> Total forecast method (lowest WAPE): {best_method}")
    print(f"   -> Deal risk method (lowest Brier)    : {risk_method}")

    # 6 -------------------------------------------------------------------
    step("6. Live forecast and scenarios")
    live_date = pd.Timestamp(config.LIVE_SNAPSHOT_DATE)
    live = forecasting.live_forecast(tables, frames, live_date, cfg.seed)
    intervals = sa.backtest_intervals(deals, best_method, seed=cfg.seed)
    intervals.to_csv(reports / "scenario_backtest.csv", index=False, date_format="%Y-%m-%d")
    scen = sa.build_scenarios(live["deal_amount"], live[f"p_{best_method}"], intervals,
                              best_method, cfg.seed)
    scen["forecast_quarter"] = forecasting.quarter_of(horizon_end(live_date))
    scen["snapshot_date"] = live_date.strftime("%Y-%m-%d")
    # Scenario totals for every method, for comparison on the dashboard.
    scen["all_methods_base"] = {m: float((live["deal_amount"] * live[f"p_{m}"]).sum())
                                for m in forecasting.METHODS}
    (reports / "scenarios.json").write_text(json.dumps(to_jsonable(scen), indent=2), encoding="utf-8")
    print(f"   Forecast quarter {scen['forecast_quarter']} (open deals: {len(live)})")
    print(f"   Downside {scen['downside'] / CRORE:,.2f} cr | Base {scen['base'] / CRORE:,.2f} cr | "
          f"Upside {scen['upside'] / CRORE:,.2f} cr")
    print(f"   Simulation P10-P90 contained the actual in "
          f"{scen['simulation_coverage']:.0%} of {scen['backtest_periods']} backtest quarters")

    # 7 -------------------------------------------------------------------
    step("7. Business insights")
    flagged = bi.flag_opportunities(live, f"p_{risk_method}", f"p_{best_method}", live_date)
    flagged = flagged.merge(tables["sales_reps"][["rep_id", "rep_name"]], on="rep_id", how="left")
    flagged = flagged.merge(tables["accounts"][["account_id", "account_name"]], on="account_id", how="left")
    reps_follow = bi.rep_followup(flagged, tables["sales_reps"])
    gaps = bi.region_gaps(flagged, tables["quotas"], tables["sales_reps"], scen["forecast_quarter"])
    conc = bi.concentration(flagged, tables["accounts"])
    summary = bi.insight_summary(flagged)
    summary["slip_rule_backtest"] = bi.slip_rule_backtest(deals, tables["opportunity_outcomes"])
    monthly = monthly_breakdown(flagged, live_date)
    keep = ["opportunity_id", "account_id", "account_name", "rep_id", "rep_name", "region",
            "industry", "account_size", "product", "stage", "deal_amount", "opportunity_created_date",
            "expected_close_date", "opportunity_age_days", "days_until_expected_close",
            "last_activity_date", "days_since_last_activity", "num_calls", "num_emails",
            "num_meetings", "num_followups", "close_date_pushes", "stage_conversion_rate",
            "win_probability", "forecast_probability", "expected_booking",
            *bi.FLAG_LABELS, "risk_flag_count", "risk_level", "risk_reasons"]
    current = flagged[keep].sort_values("expected_booking", ascending=False)
    current.to_csv(reports / "current_forecast.csv", index=False, date_format="%Y-%m-%d")
    reps_follow.to_csv(reports / "rep_followup.csv", index=False)
    gaps.to_csv(reports / "region_gaps.csv", index=False)
    conc.to_csv(reports / "concentration.csv", index=False)
    monthly.to_csv(reports / "monthly_forecast.csv", index=False)
    (reports / "insights.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    for k, v in summary.items():
        if isinstance(v, dict) and "label" in v:
            print(f"   {v['label']:<40}{v['count']:>5} deals  ({v['pipeline_value'] / CRORE:,.1f} cr)")
    sb = summary["slip_rule_backtest"]
    print(f"   Slip rule check (history): flagged deals slipped {sb['flagged_slip_rate']:.0%} "
          f"vs {sb['overall_slip_rate']:.0%} of all deals expected in-quarter")
    print(f"   Reps with >=50% of open deals flagged: {len(reps_follow)}")

    # 8 -------------------------------------------------------------------
    step("8. Saving database and charts")
    db_tables = dict(tables)
    db_tables["forecast_snapshots"] = snapshots
    counts = database.create_database(db_tables)
    database.save_table(current, "current_forecast")
    database.save_table(periods, "backtest_periods")
    print(f"   SQLite database: {config.DATABASE_PATH.name} ({sum(counts.values()):,} rows)")
    stage_dist = (current.groupby("stage")["deal_amount"].sum()
                  .reindex(config.OPEN_STAGE_NAMES).fillna(0))
    make_figures(periods, scen, stage_dist, best_method)

    run_info = {
        "scale": cfg.scale.name, "seed": cfg.seed,
        "forecast_method": best_method, "risk_method": risk_method,
        "live_snapshot_date": live_date.strftime("%Y-%m-%d"),
        "forecast_quarter": scen["forecast_quarter"],
        "clean_errors": clean_summary["errors"], "clean_warnings": clean_summary["warnings"],
        "sample_anomalies_detected": f"{int(detection['detected'].sum())}/{len(detection)}",
        "runtime_seconds": round(time.time() - started, 1),
    }
    (reports / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"\nDone in {run_info['runtime_seconds']} s. Start the dashboard with:"
          f"\n    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
