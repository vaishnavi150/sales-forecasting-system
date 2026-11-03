"""
src/scenario_analysis.py - Downside, Base and Upside forecasts.

TWO WAYS TO MEASURE UNCERTAINTY
-------------------------------
1. Simulation ("Monte Carlo")
   For every open deal, flip a weighted coin: heads (won) with the deal's
   probability. Add up the won amounts = one possible quarter. Repeat
   10,000 times. The 10th, 50th and 90th percentiles of those totals are
   the P10 / P50 / P90 outcomes.
   LIMITATION: it assumes deals win or lose independently. In real life
   a bad economy hurts many deals at once, so this range is usually TOO
   NARROW. That is why we also use method 2.

2. Historical error ("empirical")
   In the backtest we know how far each past forecast was from reality.
   If past actuals ranged from 85% to 120% of the forecast, we apply the
   same range to today's forecast. With only a few past quarters this is
   a rough estimate, and we say so.

FINAL SCENARIOS (documented assumptions)
----------------------------------------
    Base     = expected bookings of the chosen method (sum of amount x probability)
    Downside = the LOWER of: simulation P10, historical-error 10th percentile
    Upside   = the HIGHER of: simulation P90, historical-error 90th percentile
Taking the wider of the two keeps the range honest when one method is
too optimistic about certainty. If there are fewer than
MIN_PERIODS_FOR_EMPIRICAL past quarters, only the simulation is used and
the result is marked as not calibrated.

We also CHECK the simulation: in the backtest, how often did the real
result land between P10 and P90? (Ideally about 80% of the time.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

MIN_PERIODS_FOR_EMPIRICAL = 4


def simulate_totals(amounts, probabilities, runs: int = config.SIMULATION_RUNS,
                    seed: int = config.RANDOM_SEED, batch: int = 1000) -> np.ndarray:
    """Return `runs` simulated quarter totals."""
    a = np.asarray(amounts, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if a.shape != p.shape:
        raise ValueError("amounts and probabilities must have the same length")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be between 0 and 1")
    rng = np.random.default_rng(seed)
    totals = []
    done = 0
    while done < runs:            # work in batches to limit memory use
        n = min(batch, runs - done)
        wins = rng.random((n, len(a))) < p
        totals.append(wins @ a)
        done += n
    return np.concatenate(totals) if totals else np.zeros(0)


def summarise(totals: np.ndarray) -> dict:
    return {
        "P10": float(np.percentile(totals, config.DOWNSIDE_PERCENTILE)),
        "P50": float(np.percentile(totals, config.BASE_PERCENTILE)),
        "P90": float(np.percentile(totals, config.UPSIDE_PERCENTILE)),
        "mean": float(np.mean(totals)),
    }


def backtest_intervals(deal_predictions: pd.DataFrame, method: str,
                       runs: int = 2000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """For each past snapshot: simulated P10/P50/P90 vs the actual result."""
    rows = []
    col = f"p_{method}"
    for d, grp in deal_predictions.groupby("snapshot_date"):
        if grp[col].isna().any():
            continue
        s = summarise(simulate_totals(grp["deal_amount"], grp[col], runs, seed))
        actual = float(grp["actual_booking"].sum())
        expected = float((grp["deal_amount"] * grp[col]).sum())
        rows.append({"snapshot_date": d, "expected": expected, **s, "actual": actual,
                     "actual_to_expected": actual / expected if expected else np.nan,
                     "inside_P10_P90": bool(s["P10"] <= actual <= s["P90"])})
    return pd.DataFrame(rows)


def build_scenarios(amounts, probabilities, interval_history: pd.DataFrame,
                    method: str, seed: int = config.RANDOM_SEED) -> dict:
    """Combine simulation and historical error into Downside / Base / Upside."""
    totals = simulate_totals(amounts, probabilities, config.SIMULATION_RUNS, seed)
    sim = summarise(totals)
    base = float(np.sum(np.asarray(amounts) * np.asarray(probabilities)))
    ratios = interval_history["actual_to_expected"].dropna().to_numpy()
    n = len(ratios)
    result = {
        "method": method,
        "base": base,
        "simulation": sim,
        "backtest_periods": int(n),
        "simulation_coverage": (float(interval_history["inside_P10_P90"].mean())
                                if n else None),
        "assumptions": [
            "Base = sum of deal amount x win probability for open deals.",
            "Simulation treats each deal as an independent win/loss event.",
            "Historical-error range applies past actual/forecast ratios to today's base.",
            "Downside/Upside use the wider of the simulation and historical-error ranges.",
            "Deals created after the forecast date are not included (pipeline-only forecast).",
        ],
    }
    if n >= MIN_PERIODS_FOR_EMPIRICAL:
        lo = float(np.quantile(ratios, 0.10))
        hi = float(np.quantile(ratios, 0.90))
        result["empirical"] = {"ratio_P10": lo, "ratio_P90": hi,
                               "downside": base * lo, "upside": base * hi}
        result["downside"] = min(sim["P10"], base * lo)
        result["upside"] = max(sim["P90"], base * hi)
        result["calibrated"] = True
        result["limitation"] = (f"Historical-error range is based on only {n} past quarters "
                                "of synthetic data; treat it as a rough guide.")
    else:
        result["empirical"] = None
        result["downside"] = sim["P10"]
        result["upside"] = sim["P90"]
        result["calibrated"] = False
        result["limitation"] = (f"Only {n} past quarters available (< {MIN_PERIODS_FOR_EMPIRICAL}); "
                                "range comes from simulation only and is NOT calibrated.")
    return result
