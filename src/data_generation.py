"""
src/data_generation.py - creates realistic SYNTHETIC sales data.

WHAT THIS FILE DOES
-------------------
1. Creates master (parent) tables first:
       territories -> sales reps -> accounts -> stages
2. Then creates child / event tables using valid parent IDs:
       opportunities -> stage history -> activities
3. Then DERIVES tables from those records (never invents them separately):
       historical bookings (from won deals) and quotas
4. Saves the "answers" (final outcomes) in a SEPARATE ground-truth folder.
5. Builds a small SAMPLE dataset with deliberately injected, labelled errors
   so we can prove that data validation works.

HOW OUTCOMES ARE DECIDED (documented rules, not coin flips)
-----------------------------------------------------------
Every deal gets a hidden "win score" made from:
  * the sales rep's hidden skill
  * the deal's hidden "engagement" (how interested the customer is)
  * account size, industry and product
Win probability = sigmoid(win score). Engaged deals also get MORE activity
(calls, emails, meetings), and lost deals usually "go dark" (activity stops)
before they are lost. That is why activity features can predict outcomes.

Deals can SLIP (close later than expected). Low engagement increases the
chance of slipping. Most reps then push the expected close date forward;
some forget, which creates "overdue" opportunities.

Anything that happens after DATA_END_DATE is cut off ("censored"), exactly
like a real CRM export taken on that day. Those deals stay "open".

The hidden scores are NEVER saved. Only things a real CRM would record
are written to files.

RUN IT
------
    python -m src.data_generation                 (starter scale)
    python -m src.data_generation --scale full
    python -m src.data_generation --check-reproducibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import ScaleConfig

# ----------------------------------------------------------------------
# Name lists for fictional people and companies (any match to a real
# person or company is accidental).
# ----------------------------------------------------------------------
FIRST_NAMES = [
    "Aarav", "Aditi", "Akhil", "Ananya", "Arjun", "Bhavana", "Chaitanya",
    "Deepika", "Divya", "Gopal", "Harika", "Ishaan", "Kavya", "Kiran",
    "Lakshmi", "Madhav", "Meera", "Naveen", "Nikhil", "Pooja", "Pranav",
    "Priya", "Rahul", "Ramya", "Ravi", "Sai", "Sandeep", "Sneha",
    "Srinivas", "Swathi", "Tarun", "Uma", "Varun", "Vikram", "Yamini",
]
LAST_NAMES = [
    "Reddy", "Rao", "Sharma", "Naidu", "Iyer", "Patel", "Gupta", "Kumar",
    "Varma", "Menon", "Das", "Joshi", "Pillai", "Chowdary", "Nair",
    "Singh", "Mehta", "Bose", "Kulkarni", "Shetty",
]
COMPANY_WORDS_1 = [
    "Godavari", "Krishna", "Deccan", "Lotus", "Sapphire", "Banyan", "Indus",
    "Horizon", "Summit", "Pioneer", "Silverline", "Greenfield", "Bluewave",
    "Sunrise", "Evergreen", "Northstar", "Coral", "Granite", "Meridian",
    "Vertex", "Orbit", "Crescent", "Harbor", "Peak", "Trident",
]
COMPANY_WORDS_2 = [
    "Tech", "Care", "Learning", "Motors", "Retail", "Capital", "Logix",
    "Agro", "Pharma", "Foods", "Infra", "Energy", "Textiles", "Health",
    "Digital", "Metals", "Labs", "Networks", "Steel", "Water",
]
COMPANY_SUFFIXES = ["Pvt Ltd", "Ltd", "Solutions", "Enterprises", "Group", "Industries"]

# ----------------------------------------------------------------------
# Generation assumptions (documented in docs/data_dictionary.md)
# ----------------------------------------------------------------------
# Typical deal size in INR by account size (median of a log-normal curve).
MEDIAN_DEAL_AMOUNT = {"Small": 400_000, "Medium": 1_200_000,
                      "Large": 4_000_000, "Enterprise": 12_000_000}
# Typical sales cycle in days by account size.
MEDIAN_CYCLE_DAYS = {"Small": 45, "Medium": 75, "Large": 110, "Enterprise": 150}
# Bigger accounts are harder to win.
SIZE_WIN_EFFECT = {"Small": 0.25, "Medium": 0.10, "Large": -0.10, "Enterprise": -0.30}
ACCOUNT_SIZE_WEIGHTS = [0.40, 0.32, 0.20, 0.08]
# Monthly seasonality for new opportunities (Jan..Dec).
MONTH_WEIGHTS = [1.10, 1.00, 0.85, 1.20, 1.00, 0.90, 1.10, 1.05, 0.90, 1.15, 1.00, 0.70]
# Base quarterly quota in INR by team (before growth and random variation).
TEAM_QUOTA = {"Inside Sales": 2_000_000, "Field Sales": 4_200_000,
              "Enterprise Sales": 7_000_000}
ACTIVITY_TYPE_WEIGHTS = np.array([0.30, 0.45, 0.12, 0.13])  # Call, Email, Meeting, Follow-up

STAGE_BY_NAME = {s.name: s for s in config.STAGES}
OPEN_STAGES = config.OPEN_STAGE_NAMES  # Prospecting .. Negotiation

DATE_FORMAT = "%Y-%m-%d"


# ======================================================================
# Small helper functions
# ======================================================================

def _day_to_date(day_numbers) -> pd.Series:
    """Convert 'days since DATA_START_DATE' integers to real dates."""
    origin = pd.Timestamp(config.DATA_START_DATE)
    return origin + pd.to_timedelta(pd.Series(day_numbers, dtype="int64"), unit="D")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def quarter_label(d) -> str:
    """date(2026, 2, 10) -> '2026Q1'."""
    d = pd.Timestamp(d)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _end_day() -> int:
    return (config.DATA_END_DATE - config.DATA_START_DATE).days


# ======================================================================
# 1. MASTER TABLES
# ======================================================================

def generate_territories(scale: ScaleConfig) -> pd.DataFrame:
    """Territories are spread evenly over the regions (round-robin)."""
    rows = []
    counter = {r: 0 for r in config.REGIONS}
    for i in range(scale.num_territories):
        region = config.REGIONS[i % len(config.REGIONS)]
        counter[region] += 1
        rows.append({
            "territory_id": config.TERRITORY_ID_FORMAT.format(n=i + 1),
            "territory_name": f"{region} Territory {counter[region]}",
            "region": region,
        })
    return pd.DataFrame(rows)


def generate_reps(scale: ScaleConfig, territories: pd.DataFrame,
                  rng: np.random.Generator):
    """
    Returns (reps, rep_territory_history, hidden_traits).

    hidden_traits holds each rep's secret skill and activity level.
    It is used to simulate outcomes and is NEVER saved to disk.
    """
    n = scale.num_reps
    terr_ids = territories["territory_id"].tolist()
    terr_region = dict(zip(territories["territory_id"], territories["region"]))

    # The first rep of each territory is a long-serving rep who never moves,
    # so every territory always has at least one active rep.
    first_territory = [terr_ids[i % len(terr_ids)] for i in range(n)]
    first_territory[len(terr_ids):] = list(rng.choice(terr_ids, size=max(0, n - len(terr_ids))))

    skill = rng.normal(0.0, 0.5, n)
    activity_level = rng.lognormal(0.0, 0.3, n)
    teams = rng.choice(config.TEAMS, size=n, p=[0.45, 0.35, 0.20])

    reps, history = [], []
    data_start = pd.Timestamp(config.DATA_START_DATE)
    for i in range(n):
        rep_id = config.REP_ID_FORMAT.format(n=i + 1)
        anchor = i < len(terr_ids)
        if anchor or rng.random() < 0.85:
            joining = data_start - pd.Timedelta(int(rng.integers(60, 2500)), unit="D")
        else:  # new joiner during 2024-2025
            joining = data_start + pd.Timedelta(int(rng.integers(30, 700)), unit="D")
        territory = first_territory[i]
        # Historical performance: a 1-5 rating given BEFORE our data period.
        # It is a noisy reflection of skill (ratings are imperfect).
        rating = float(np.clip(np.round(3 + 1.6 * skill[i] + rng.normal(0, 0.6), 1), 1, 5))

        # About 10% of non-anchor reps change territory (same region) in 2025.
        move_day = None
        if not anchor and rng.random() < 0.10:
            same_region = [t for t in terr_ids
                           if terr_region[t] == terr_region[territory] and t != territory]
            if same_region:
                move_day = pd.Timestamp(2025, 1, 1) + pd.Timedelta(int(rng.integers(0, 365)), unit="D")
                if move_day <= joining:
                    move_day = None
        if move_day is None:
            history.append({"rep_id": rep_id, "territory_id": territory,
                            "start_date": joining, "end_date": pd.NaT})
            current = territory
        else:
            new_territory = str(rng.choice(same_region))
            history.append({"rep_id": rep_id, "territory_id": territory,
                            "start_date": joining, "end_date": move_day - pd.Timedelta(1, unit="D")})
            history.append({"rep_id": rep_id, "territory_id": new_territory,
                            "start_date": move_day, "end_date": pd.NaT})
            current = new_territory

        reps.append({
            "rep_id": rep_id,
            "rep_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "region": terr_region[current],
            "territory_id": current,
            "team": teams[i],
            "joining_date": joining,
            "historical_performance": rating,
        })

    reps_df = pd.DataFrame(reps)
    hist_df = pd.DataFrame(history)
    hist_df.insert(0, "assignment_id",
                   [f"RTA-{i + 1:06d}" for i in range(len(hist_df))])
    hidden = pd.DataFrame({"rep_id": reps_df["rep_id"], "skill": skill,
                           "activity_level": activity_level})
    return reps_df, hist_df, hidden


def generate_accounts(scale: ScaleConfig, territories: pd.DataFrame,
                      rng: np.random.Generator) -> pd.DataFrame:
    n = scale.num_accounts
    all_names = [f"{a} {b} {c}" for a in COMPANY_WORDS_1
                 for b in COMPANY_WORDS_2 for c in COMPANY_SUFFIXES]
    if n > len(all_names):
        raise ValueError(f"Cannot create {n} unique account names (max {len(all_names)}).")
    names = rng.choice(all_names, size=n, replace=False)
    terr = rng.choice(territories["territory_id"].to_numpy(), size=n)
    terr_region = dict(zip(territories["territory_id"], territories["region"]))
    return pd.DataFrame({
        "account_id": [config.ACCOUNT_ID_FORMAT.format(n=i + 1) for i in range(n)],
        "account_name": names,
        "industry": rng.choice(config.INDUSTRIES, size=n),
        "region": [terr_region[t] for t in terr],
        "territory_id": terr,
        "account_size": rng.choice(config.ACCOUNT_SIZES, size=n, p=ACCOUNT_SIZE_WEIGHTS),
    })


def generate_stages() -> pd.DataFrame:
    return pd.DataFrame([{
        "stage_id": s.stage_id, "stage_name": s.name, "stage_order": s.order,
        "default_probability": s.default_probability, "is_closed": int(s.is_closed),
    } for s in config.STAGES])


# ======================================================================
# 2. OPPORTUNITIES, STAGE HISTORY AND ACTIVITIES
# ======================================================================

def _creation_day_weights() -> np.ndarray:
    """Weight for every possible creation day: seasonality + growth + fewer on weekends."""
    days = pd.date_range(config.DATA_START_DATE, config.DATA_END_DATE, freq="D")
    month_w = np.array(MONTH_WEIGHTS)[days.month.to_numpy() - 1]
    growth = 1 + 0.15 * (days.year.to_numpy() - config.DATA_START_DATE.year
                         + days.dayofyear.to_numpy() / 365.0)
    weekday = np.where(days.dayofweek.to_numpy() >= 5, 0.2, 1.0)
    w = month_w * growth * weekday
    return w / w.sum()


def _active_reps_by_territory(history: pd.DataFrame, reps: pd.DataFrame):
    """territory_id -> arrays (rep_index, start_day, end_day) for fast lookups."""
    origin = pd.Timestamp(config.DATA_START_DATE)
    rep_index = {r: i for i, r in enumerate(reps["rep_id"])}
    out = {}
    for terr, grp in history.groupby("territory_id"):
        start = (grp["start_date"] - origin).dt.days.to_numpy()
        end = grp["end_date"].fillna(pd.Timestamp("2100-01-01"))
        end = (end - origin).dt.days.to_numpy()
        out[terr] = (grp["rep_id"].map(rep_index).to_numpy(), start, end)
    return out


def simulate_opportunities(scale: ScaleConfig, reps: pd.DataFrame,
                           rep_history: pd.DataFrame, hidden: pd.DataFrame,
                           accounts: pd.DataFrame, rng: np.random.Generator):
    """
    Simulate each opportunity's whole life, then cut off at DATA_END_DATE.

    Returns (opportunities, stage_history, activities, outcomes).
    """
    n = scale.num_opportunities
    end_day = _end_day()
    day_weights = _creation_day_weights()
    created_days = np.sort(rng.choice(len(day_weights), size=n, p=day_weights))

    # Some accounts buy much more often than others (pipeline concentration).
    acc_weights = rng.gamma(0.8, 1.0, len(accounts))
    acc_weights /= acc_weights.sum()
    acc_idx = rng.choice(len(accounts), size=n, p=acc_weights)

    industry_effect = dict(zip(config.INDUSTRIES, rng.normal(0, 0.2, len(config.INDUSTRIES))))
    product_effect = dict(zip(config.PRODUCTS, rng.normal(0, 0.15, len(config.PRODUCTS))))
    skill = hidden["skill"].to_numpy()
    act_level = hidden["activity_level"].to_numpy()
    rep_ids = reps["rep_id"].to_numpy()
    join_day = (reps["joining_date"] - pd.Timestamp(config.DATA_START_DATE)).dt.days.to_numpy()
    terr_reps = _active_reps_by_territory(rep_history, reps)
    rep_region = dict(zip(reps["rep_id"], reps["region"]))
    # Each rep has a "CRM hygiene" level: how often they update slipped close dates.
    hygiene = np.clip(rng.normal(0.75, 0.15, len(reps)), 0.3, 0.98)

    opp_rows, event_rows, act_rows, outcome_rows = [], [], [], []

    for i in range(n):
        created = int(created_days[i])
        acc = accounts.iloc[acc_idx[i]]
        size = acc["account_size"]

        # --- choose an active rep in the account's territory ---------------
        r_idx, r_start, r_end = terr_reps[acc["territory_id"]]
        ok = (r_start <= created) & (r_end >= created) & (join_day[r_idx] <= created)
        candidates = r_idx[ok]
        if len(candidates) == 0:  # fallback: any rep already working in the region
            candidates = np.array([j for j, rid in enumerate(rep_ids)
                                   if rep_region[rid] == acc["region"] and join_day[j] <= created])
        rep = int(rng.choice(candidates))

        # --- deal basics ---------------------------------------------------
        product = str(rng.choice(config.PRODUCTS))
        amount = float(np.round(MEDIAN_DEAL_AMOUNT[size] * rng.lognormal(0, 0.5), -3))
        amount = max(amount, 50_000.0)
        planned = max(14, int(round(MEDIAN_CYCLE_DAYS[size] * rng.lognormal(0, 0.3))))
        expected_close = created + planned

        # --- hidden outcome rule ------------------------------------------
        engagement = rng.normal(0, 1)
        score = (-0.55 + 1.2 * skill[rep] + 0.8 * engagement + SIZE_WIN_EFFECT[size]
                 + industry_effect[acc["industry"]] + product_effect[product])
        won = rng.random() < _sigmoid(score)

        # --- timing and slippage ------------------------------------------
        slips = rng.random() < _sigmoid(-0.5 - 0.8 * engagement)
        factor = rng.uniform(1.15, 2.2) if slips else rng.uniform(0.75, 1.05)
        if won:
            last_stage = 4
            total = max(7, int(round(planned * factor)))
            close = created + total
            # Quarter-end "hockey stick": some won deals are pushed to close
            # in the last two weeks of their quarter.
            if rng.random() < 0.30:
                q_end = (pd.Timestamp(config.DATA_START_DATE) + pd.Timedelta(close, unit="D")
                         + pd.offsets.QuarterEnd(0))
                q_end_day = (q_end - pd.Timestamp(config.DATA_START_DATE)).days
                close = max(close, q_end_day - int(rng.integers(0, 14)))
        else:
            # Lost deals usually die in early stages; engaged deals go further.
            base = np.array([0.40, 0.30, 0.18, 0.12])
            w = base * np.exp(0.3 * engagement * (np.arange(1, 5) - 2.5))
            last_stage = int(rng.choice([1, 2, 3, 4], p=w / w.sum()))
            close = created + max(7, int(round(planned * factor * rng.uniform(0.5, 1.4))))
        total = close - created

        # --- stage events (fractions of the total cycle) -------------------
        fractions = [0.0, rng.uniform(0.05, 0.25), rng.uniform(0.30, 0.50),
                     rng.uniform(0.55, 0.85)]
        events = []  # (day, order_within_day, event_type, stage, expected_close)
        for s in range(last_stage):
            day = created + int(round(total * fractions[s]))
            events.append([day, 0, "stage_change", OPEN_STAGES[s], None])

        # --- close-date push (slippage seen in the CRM) --------------------
        if close > expected_close and rng.random() < hygiene[rep]:
            push_day = max(created + 1, expected_close - int(rng.integers(0, 10)))
            new_expected = max(push_day + 1, close + int(rng.integers(-7, 21)))
            events.append([push_day, 1, "close_date_change", None, new_expected])

        final_stage = config.WON_STAGE if won else config.LOST_STAGE
        events.append([close, 2, "stage_change", final_stage, close])
        events.sort(key=lambda e: (e[0], e[1]))

        # Fill in the stage and expected close date in effect at each event.
        current_stage, current_expected = OPEN_STAGES[0], expected_close
        for e in events:
            if e[3] is None:
                e[3] = current_stage
            current_stage = e[3]
            if e[4] is None:
                e[4] = current_expected
            current_expected = e[4]

        # --- activities ----------------------------------------------------
        if won:
            last_active = close
        else:
            dark = int(rng.integers(15, 61)) if rng.random() < 0.75 else 0
            last_active = max(created, close - dark)
        span = last_active - created + 1
        rate_per_week = 1.5 * act_level[rep] * np.exp(0.4 * engagement)
        n_act = rng.poisson(rate_per_week * span / 7.0)
        if n_act > 0:
            a_days = created + rng.integers(0, span, n_act)
            p = ACTIVITY_TYPE_WEIGHTS * np.array([1, 1, np.exp(0.3 * engagement), 1])
            a_types = rng.choice(4, size=n_act, p=p / p.sum())
            keys, counts = np.unique(a_days * 4 + a_types, return_counts=True)
            for k, c in zip(keys, counts):
                d = int(k // 4)
                if d <= end_day:
                    act_rows.append((i, d, int(k % 4), int(c)))

        # --- censor at DATA_END_DATE (the "export date") -------------------
        visible = [e for e in events if e[0] <= end_day]
        for seq, e in enumerate(visible):
            event_rows.append((i, seq + 1, e[0], e[2], e[3], e[4]))
        last = visible[-1]
        is_closed = last[3] in (config.WON_STAGE, config.LOST_STAGE)
        final_status = ("Won" if last[3] == config.WON_STAGE else
                        "Lost" if last[3] == config.LOST_STAGE else "Open")

        opp_rows.append({
            "_i": i, "_created": created,
            "account_id": acc["account_id"], "rep_id": rep_ids[rep],
            "product": product, "deal_amount": amount,
            "currency": config.BASE_CURRENCY,
            "_expected_close": last[4], "opportunity_stage": last[3],
            "current_probability": STAGE_BY_NAME[last[3]].default_probability,
            "is_closed": int(is_closed), "_last_modified": last[0],
        })
        outcome_rows.append({
            "_i": i, "final_status": final_status,
            "_close": close if is_closed else None,
            "won_amount": amount if final_status == "Won" else 0.0,
            "_initial_expected": expected_close,
            "slipped": (int(close > expected_close) if is_closed else None),
        })

    # --- give opportunities their IDs (OPP-<creation year>-<counter>) ------
    opps = pd.DataFrame(opp_rows)
    created_dates = _day_to_date(opps["_created"])
    years = created_dates.dt.year
    counters = years.groupby(years).cumcount() + 1
    opp_ids = [config.OPPORTUNITY_ID_FORMAT.format(year=y, n=c)
               for y, c in zip(years, counters)]
    id_of = dict(zip(opps["_i"], opp_ids))

    opportunities = pd.DataFrame({
        "opportunity_id": opp_ids,
        "account_id": opps["account_id"],
        "rep_id": opps["rep_id"],
        "product": opps["product"],
        "opportunity_created_date": created_dates,
        "expected_close_date": _day_to_date(opps["_expected_close"]),
        "opportunity_stage": opps["opportunity_stage"],
        "deal_amount": opps["deal_amount"],
        "currency": opps["currency"],
        "current_probability": opps["current_probability"],
        "is_closed": opps["is_closed"],
        "last_modified_date": _day_to_date(opps["_last_modified"]),
    })

    ev = pd.DataFrame(event_rows, columns=["_i", "seq", "day", "event_type",
                                           "stage", "expected_close"])
    stage_history = pd.DataFrame({
        "stage_history_id": [config.STAGE_HISTORY_ID_FORMAT.format(opportunity_id=id_of[i], n=s)
                             for i, s in zip(ev["_i"], ev["seq"])],
        "opportunity_id": ev["_i"].map(id_of),
        "change_date": _day_to_date(ev["day"]),
        "event_type": ev["event_type"],
        "stage": ev["stage"],
        "stage_probability": ev["stage"].map(lambda s: STAGE_BY_NAME[s].default_probability),
        "expected_close_date": _day_to_date(ev["expected_close"]),
    })

    act = pd.DataFrame(act_rows, columns=["_i", "day", "type", "count"])
    act["opportunity_id"] = act["_i"].map(id_of)
    act = act.sort_values(["day", "opportunity_id", "type"], kind="mergesort").reset_index(drop=True)
    rep_of_opp = dict(zip(opportunities["opportunity_id"], opportunities["rep_id"]))
    activities = pd.DataFrame({
        "activity_id": [config.ACTIVITY_ID_FORMAT.format(n=k + 1) for k in range(len(act))],
        "opportunity_id": act["opportunity_id"],
        "rep_id": act["opportunity_id"].map(rep_of_opp),
        "activity_date": _day_to_date(act["day"]),
        "activity_type": np.array(config.ACTIVITY_TYPES)[act["type"].to_numpy()],
        "activity_count": act["count"],
    })

    out = pd.DataFrame(outcome_rows)
    outcomes = pd.DataFrame({
        "opportunity_id": out["_i"].map(id_of),
        "final_status": out["final_status"],
        "close_date": pd.to_datetime(
            pd.Timestamp(config.DATA_START_DATE)
            + pd.to_timedelta(out["_close"], unit="D")),
        "won_amount": out["won_amount"],
        "initial_expected_close_date": _day_to_date(out["_initial_expected"]),
        "slipped": out["slipped"].astype("Int64"),
    })
    return opportunities, stage_history, activities, outcomes


# ======================================================================
# 3. DERIVED TABLES (calculated from records, never invented)
# ======================================================================

def derive_bookings(opportunities: pd.DataFrame, stage_history: pd.DataFrame,
                    accounts: pd.DataFrame) -> pd.DataFrame:
    """One booking per WON deal. Booking date = date of the 'Closed Won' event."""
    won_events = stage_history[stage_history["stage"] == config.WON_STAGE]
    b = won_events[["opportunity_id", "change_date"]].merge(
        opportunities[["opportunity_id", "rep_id", "account_id", "deal_amount", "currency"]],
        on="opportunity_id")
    b = b.merge(accounts[["account_id", "territory_id"]], on="account_id")
    b = b.sort_values(["change_date", "opportunity_id"]).reset_index(drop=True)
    return pd.DataFrame({
        "booking_id": [config.BOOKING_ID_FORMAT.format(n=k + 1) for k in range(len(b))],
        "opportunity_id": b["opportunity_id"],
        "rep_id": b["rep_id"],
        "account_id": b["account_id"],
        "territory_id": b["territory_id"],
        "booking_date": b["change_date"],
        "booking_quarter": b["change_date"].map(quarter_label),
        "booking_amount": b["deal_amount"],
        "currency": b["currency"],
    })


def generate_quotas(reps: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Quarterly quota per rep, set by team, yearly growth and some variation."""
    rows = []
    for _, rep in reps.iterrows():
        for year in config.QUOTA_YEARS:
            for q in range(1, 5):
                q_start = pd.Timestamp(year, 3 * q - 2, 1)
                q_end = q_start + pd.offsets.QuarterEnd(0)
                if rep["joining_date"] > q_end:
                    continue
                share = 1.0
                if rep["joining_date"] > q_start:  # joined mid-quarter -> pro-rate
                    share = (q_end - rep["joining_date"]).days / (q_end - q_start).days
                amount = (TEAM_QUOTA[rep["team"]] * (1.06 ** (year - 2024))
                          * rng.uniform(0.9, 1.1) * share)
                rows.append({
                    "quota_id": config.QUOTA_ID_FORMAT.format(rep_id=rep["rep_id"], year=year, quarter=q),
                    "rep_id": rep["rep_id"],
                    "quota_quarter": f"{year}Q{q}",
                    "quarter_start": q_start,
                    "quarter_end": q_end.normalize(),
                    "quota_amount": float(np.round(amount, -3)),
                    "currency": config.BASE_CURRENCY,
                })
    return pd.DataFrame(rows)


# ======================================================================
# 4. MAIN ENTRY POINT
# ======================================================================

def generate_all(scale: ScaleConfig, seed: int = config.RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """
    Build every clean table plus the ground-truth outcomes.
    Same scale + same seed  ->  exactly the same tables.
    """
    rng = np.random.default_rng(seed)
    territories = generate_territories(scale)
    reps, rep_history, hidden = generate_reps(scale, territories, rng)
    accounts = generate_accounts(scale, territories, rng)
    stages = generate_stages()
    opportunities, stage_history, activities, outcomes = simulate_opportunities(
        scale, reps, rep_history, hidden, accounts, rng)
    bookings = derive_bookings(opportunities, stage_history, accounts)
    quotas = generate_quotas(reps, rng)
    return {
        "territories": territories,
        "sales_reps": reps,
        "rep_territory_history": rep_history,
        "accounts": accounts,
        "stages": stages,
        "opportunities": opportunities,
        "stage_history": stage_history,
        "activities": activities,
        "quotas": quotas,
        "historical_bookings": bookings,
        "opportunity_outcomes": outcomes,   # GROUND TRUTH - saved separately
    }


GROUND_TRUTH_TABLES = {"opportunity_outcomes"}


def save_tables(tables: dict[str, pd.DataFrame], folder: Path) -> dict[str, str]:
    """Save each table as CSV and return {file name: SHA-256 checksum}."""
    folder.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, df in tables.items():
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False, date_format=DATE_FORMAT, lineterminator="\n")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def load_tables(folder: Path, names=None) -> dict[str, pd.DataFrame]:
    """Read CSV tables back. Columns ending in '_date' are parsed as dates."""
    tables = {}
    for path in sorted(folder.glob("*.csv")):
        if names and path.stem not in names:
            continue
        df = pd.read_csv(path)
        for col in df.columns:
            if col.endswith("_date") or col in ("quarter_start", "quarter_end"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        tables[path.stem] = df
    return tables


def run_generation(scale: ScaleConfig, seed: int = config.RANDOM_SEED,
                   raw_dir: Path = config.RAW_DATA_DIR,
                   truth_dir: Path = config.GROUND_TRUTH_DIR,
                   sample_dir: Path = config.SAMPLE_DATA_DIR) -> dict:
    """Generate, save, build the error sample, and write a manifest."""
    tables = generate_all(scale, seed)
    clean = {k: v for k, v in tables.items() if k not in GROUND_TRUTH_TABLES}
    truth = {k: v for k, v in tables.items() if k in GROUND_TRUTH_TABLES}
    checksums = save_tables(clean, raw_dir)
    checksums.update(save_tables(truth, truth_dir))

    # Small dataset with labelled, deliberately injected errors.
    from src.error_injection import build_error_sample  # local import avoids a cycle
    sample_tables, labels = build_error_sample(tables, seed=seed, size=scale.num_error_rows)
    save_tables(sample_tables, sample_dir)
    save_tables({"anomaly_labels": labels}, truth_dir)

    manifest = {
        "generator_command": f"python -m src.data_generation --scale {scale.name}",
        "seed": seed,
        "scale": scale.__dict__,
        "data_period": [str(config.DATA_START_DATE), str(config.DATA_END_DATE)],
        "currency": config.BASE_CURRENCY,
        "source": "100% synthetic data created by src/data_generation.py. "
                  "No external or personal data was used.",
        "row_counts": {k: int(len(v)) for k, v in tables.items()},
        "sample_row_counts": {k: int(len(v)) for k, v in sample_tables.items()},
        "injected_anomalies": int(len(labels)),
        "sha256": checksums,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def check_reproducibility(scale: ScaleConfig, seed: int = config.RANDOM_SEED) -> bool:
    """Generate twice in temporary folders and compare checksums."""
    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            tables = generate_all(scale, seed)
            results.append(save_tables(tables, Path(tmp)))
    same = results[0] == results[1]
    print("Run 1 vs Run 2 checksums identical:", same)
    for name in sorted(results[0]):
        flag = "OK " if results[0][name] == results[1][name] else "DIFF"
        print(f"  [{flag}] {name}: {results[0][name][:16]}...")
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sales data.")
    parser.add_argument("--scale", choices=list(config.SCALES), default=None,
                        help="starter (default) or full")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--check-reproducibility", action="store_true",
                        help="generate twice and compare checksums")
    args = parser.parse_args()
    cfg = config.get_config(args.scale)
    config.ensure_directories()
    if args.check_reproducibility:
        ok = check_reproducibility(cfg.scale, args.seed)
        raise SystemExit(0 if ok else 1)
    manifest = run_generation(cfg.scale, args.seed)
    print(f"Generated '{cfg.scale.name}' dataset with seed {args.seed}:")
    for name, count in manifest["row_counts"].items():
        print(f"  {name:<24} {count:>8,} rows")
    print(f"Injected anomalies in sample dataset: {manifest['injected_anomalies']}")


if __name__ == "__main__":
    main()
