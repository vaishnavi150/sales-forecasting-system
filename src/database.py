"""
src/database.py - stores the tables in a local SQLite database.

SQLite is a database that lives in ONE file (data/sales_forecasting.db).
No server is needed. We use it so the data can be queried with SQL,
and so the database itself enforces rules (unique IDs, foreign keys,
"close date >= creation date", and so on) from sql/schema.sql.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

import config

# Parents first, so foreign-key checks pass while inserting.
LOAD_ORDER = [
    "territories", "sales_reps", "rep_territory_history", "accounts", "stages",
    "opportunities", "stage_history", "activities", "quotas",
    "historical_bookings", "opportunity_outcomes", "forecast_snapshots",
]


def _dates_to_text(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite has no date type, so dates are saved as 'YYYY-MM-DD' text."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def connect(db_path: Path = config.DATABASE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_database(tables: dict[str, pd.DataFrame],
                    db_path: Path = config.DATABASE_PATH,
                    schema_path: Path = config.SQL_DIR / "schema.sql") -> dict[str, int]:
    """Create all tables from schema.sql and load the data. Returns row counts."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(schema_path.read_text())
        counts = {}
        for name in LOAD_ORDER:
            if name in tables:
                _dates_to_text(tables[name]).to_sql(name, conn, if_exists="append", index=False)
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise ValueError(f"Foreign-key problems found in database: {problems[:5]}")
    return counts


def save_table(df: pd.DataFrame, name: str, db_path: Path = config.DATABASE_PATH) -> None:
    """Save an output table (for example the current forecast), replacing any old copy."""
    with connect(db_path) as conn:
        _dates_to_text(df).to_sql(name, conn, if_exists="replace", index=False)


def query(sql: str, params: tuple = (), db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    """Run a SQL SELECT and return the result as a DataFrame."""
    with connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)
