"""
config.py - One central place for all project settings.

WHY THIS FILE EXISTS
--------------------
Instead of typing numbers like "100 reps" or "30 days" inside many
different Python files, we keep every setting HERE. If we want to
change something later (for example, switch from the small "starter"
dataset to the big "full" dataset), we change it in one place only.

HOW TO CHOOSE THE DATASET SIZE (SCALE)
--------------------------------------
By default the project uses the "starter" scale (small and fast).
To use the "full" scale, set an environment variable before running:

    Windows PowerShell:   $env:SFS_SCALE = "full"
    Windows CMD:          set SFS_SCALE=full

No code needs to change. Every other module simply calls get_config().

QUICK CHECK
-----------
Run   python config.py   to print a summary of the active settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ======================================================================
# 1. FOLDER PATHS
# ======================================================================
# Path(__file__) is the location of THIS file (config.py).
# .resolve().parent gives the project's main folder, so the paths below
# work no matter which folder you run Python from.

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"                  # clean generated tables
PROCESSED_DATA_DIR = DATA_DIR / "processed"      # features for modelling
SAMPLE_DATA_DIR = DATA_DIR / "sample"            # small dataset with injected errors
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"     # actual outcomes + anomaly labels
REPORTS_DIR = PROJECT_ROOT / "reports"           # quality reports, metrics, charts
SQL_DIR = PROJECT_ROOT / "sql"                   # database schema
DATABASE_PATH = DATA_DIR / "sales_forecasting.db"  # the SQLite database file

# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================
# A "seed" makes random numbers repeatable: same seed -> same data,
# every single time. This lets anyone re-create our exact dataset.

RANDOM_SEED = 20261022

# ======================================================================
# 3. TIME PERIOD
# ======================================================================
# History covers 8 quarters: Q4 2024 to Q3 2026.
# Opportunities start being created a little earlier (Jan 2024) so the
# pipeline is already "full" when the first snapshot is taken.

DATA_START_DATE = date(2024, 1, 1)
DATA_END_DATE = date(2026, 9, 30)

# Forecast snapshot dates = the last day of each quarter.
# On each date we take a "photo" of what was known on that day.
SNAPSHOT_DATES: list[date] = [
    date(2024, 9, 30),    # end of Q3 2024 -> forecasts Q4 2024
    date(2024, 12, 31),   # end of Q4 2024 -> forecasts Q1 2025
    date(2025, 3, 31),    # end of Q1 2025 -> forecasts Q2 2025
    date(2025, 6, 30),    # end of Q2 2025 -> forecasts Q3 2025
    date(2025, 9, 30),    # end of Q3 2025 -> forecasts Q4 2025
    date(2025, 12, 31),   # end of Q4 2025 -> forecasts Q1 2026
    date(2026, 3, 31),    # end of Q1 2026 -> forecasts Q2 2026
    date(2026, 6, 30),    # end of Q2 2026 -> forecasts Q3 2026
]

FORECAST_HORIZON_MONTHS = 3   # we forecast the NEXT quarter (3 months)

# ======================================================================
# 4. CURRENCY
# ======================================================================
# The main clean dataset uses ONE currency. Other currencies appear
# only in the small error dataset, to test the "mixed currency" check.

BASE_CURRENCY = "INR"
ALLOWED_CURRENCIES = ["INR"]

# ======================================================================
# 5. SALES STAGES
# ======================================================================
# Each stage has an order (1 = earliest) and a DEFAULT probability.
# These defaults are only a starting assumption. Later phases calculate
# REAL conversion rates from historical data.


@dataclass(frozen=True)
class Stage:
    stage_id: str          # e.g. "STG-01"
    name: str              # e.g. "Prospecting"
    order: int             # position in the sales process
    default_probability: float  # between 0.0 and 1.0
    is_closed: bool        # True for "Closed Won" and "Closed Lost"


STAGES: list[Stage] = [
    Stage("STG-01", "Prospecting",   1, 0.10, False),
    Stage("STG-02", "Qualification", 2, 0.20, False),
    Stage("STG-03", "Proposal",      3, 0.50, False),
    Stage("STG-04", "Negotiation",   4, 0.75, False),
    Stage("STG-05", "Closed Won",    5, 1.00, True),
    Stage("STG-06", "Closed Lost",   6, 0.00, True),
]

STAGE_NAMES = [s.name for s in STAGES]
OPEN_STAGE_NAMES = [s.name for s in STAGES if not s.is_closed]
WON_STAGE = "Closed Won"
LOST_STAGE = "Closed Lost"

# ======================================================================
# 6. LOOKUP VALUES (allowed categories)
# ======================================================================

REGIONS = ["North", "South", "East", "West", "Central"]
INDUSTRIES = [
    "Healthcare", "Education", "Manufacturing", "Retail",
    "Finance", "Technology", "Logistics", "Government",
]
ACCOUNT_SIZES = ["Small", "Medium", "Large", "Enterprise"]
TEAMS = ["Inside Sales", "Field Sales", "Enterprise Sales"]
ACTIVITY_TYPES = ["Call", "Email", "Meeting", "Follow-up"]
PRODUCTS = ["Analytics Suite", "CRM Cloud", "HR Platform", "Security Pack"]

# ======================================================================
# 7. ID FORMATS
# ======================================================================
# Python format strings. Example:
#   REP_ID_FORMAT.format(n=1)  ->  "REP-000001"
# Zero-padding keeps IDs the same length and sorted correctly.

REP_ID_FORMAT = "REP-{n:06d}"
ACCOUNT_ID_FORMAT = "ACC-{n:06d}"
TERRITORY_ID_FORMAT = "TER-{n:03d}"
OPPORTUNITY_ID_FORMAT = "OPP-{year}-{n:06d}"           # year = creation year
STAGE_HISTORY_ID_FORMAT = "STG-{opportunity_id}-{n:03d}"
ACTIVITY_ID_FORMAT = "ACT-{n:08d}"
QUOTA_ID_FORMAT = "QUOTA-{rep_id}-{year}Q{quarter}"
BOOKING_ID_FORMAT = "BKG-{n:06d}"
FORECAST_ID_FORMAT = "FCST-{snapshot_date}-{n:03d}"    # snapshot_date = YYYY-MM-DD

# ======================================================================
# 8. BUSINESS RULE THRESHOLDS
# ======================================================================
# Used later for risk detection. Each one is a documented assumption.

STALE_ACTIVITY_DAYS = 30           # no activity for 30+ days -> stale
LARGE_DEAL_PERCENTILE = 0.90       # top 10% of deal amounts = "large"
LOW_WIN_PROBABILITY = 0.30         # below 30% = "low" win chance
CONCENTRATION_SHARE = 0.20         # one account/rep > 20% of pipeline -> risk

# Scenario simulation settings (used in Phase 8)
SIMULATION_RUNS = 10_000
DOWNSIDE_PERCENTILE = 10           # P10
BASE_PERCENTILE = 50               # P50 (median)
UPSIDE_PERCENTILE = 90             # P90

# ======================================================================
# 9. DATASET SCALE (starter vs full)
# ======================================================================
# The SAME generator code uses either set of numbers. Only the counts
# change, never the logic. That is how the project "scales" without
# rewriting code.


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    num_territories: int
    num_reps: int
    num_accounts: int
    num_opportunities: int
    num_error_rows: int = 50   # size of the small error-test dataset


SCALES: dict[str, ScaleConfig] = {
    "starter": ScaleConfig(
        name="starter",
        num_territories=10,
        num_reps=100,
        num_accounts=500,
        num_opportunities=5_000,
    ),
    "full": ScaleConfig(
        name="full",
        num_territories=25,
        num_reps=2_000,
        num_accounts=2_000,
        num_opportunities=20_000,
    ),
}

DEFAULT_SCALE = "starter"


@dataclass(frozen=True)
class ProjectConfig:
    """Everything a module needs, bundled into one object."""
    scale: ScaleConfig
    seed: int = RANDOM_SEED
    snapshot_dates: list[date] = field(default_factory=lambda: list(SNAPSHOT_DATES))


def get_scale_name() -> str:
    """Read the scale from the SFS_SCALE environment variable (default: starter)."""
    return os.environ.get("SFS_SCALE", DEFAULT_SCALE).strip().lower()


def get_config(scale_name: str | None = None) -> ProjectConfig:
    """
    Return the active project settings.

    scale_name: "starter" or "full". If not given, we use the
                SFS_SCALE environment variable, or "starter".
    """
    name = (scale_name or get_scale_name()).strip().lower()
    if name not in SCALES:
        valid = ", ".join(SCALES)
        raise ValueError(f"Unknown scale '{name}'. Choose one of: {valid}")
    return ProjectConfig(scale=SCALES[name])


def ensure_directories() -> None:
    """Create the data and report folders if they do not exist yet."""
    for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
                   GROUND_TRUTH_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 10. SELF-CHECK
# ======================================================================
# This block only runs when you type:  python config.py
# It does NOT run when another file imports config.

if __name__ == "__main__":
    cfg = get_config()
    s = cfg.scale
    print("Sales Forecasting System - active configuration")
    print("-" * 50)
    print(f"Project folder : {PROJECT_ROOT}")
    print(f"Scale          : {s.name}")
    print(f"Territories    : {s.num_territories}")
    print(f"Sales reps     : {s.num_reps}")
    print(f"Accounts       : {s.num_accounts}")
    print(f"Opportunities  : {s.num_opportunities}")
    print(f"Random seed    : {cfg.seed}")
    print(f"Currency       : {BASE_CURRENCY}")
    print(f"Data period    : {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"Snapshots      : {len(cfg.snapshot_dates)} "
          f"({cfg.snapshot_dates[0]} to {cfg.snapshot_dates[-1]})")
    print(f"Stages         : {', '.join(STAGE_NAMES)}")
    print(f"Example IDs    : {REP_ID_FORMAT.format(n=1)}, "
          f"{OPPORTUNITY_ID_FORMAT.format(year=2026, n=1)}, "
          f"{FORECAST_ID_FORMAT.format(snapshot_date='2026-01-31', n=1)}")
