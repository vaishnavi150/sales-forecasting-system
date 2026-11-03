-- ====================================================================
-- sql/schema.sql - SQLite database schema for the Sales Forecasting System
--
-- Every table has a PRIMARY KEY (unique ID) and FOREIGN KEYs for every
-- relationship. Dates are stored as TEXT in YYYY-MM-DD format.
-- Tables are created parents-first so foreign keys can be checked.
--
-- Cardinality (1:N = one-to-many):
--   territories 1:N sales_reps (current territory), 1:N accounts
--   sales_reps  N:M territories over time -> bridge table rep_territory_history
--   accounts    1:N opportunities
--   sales_reps  1:N opportunities, 1:N activities, 1:N quotas
--   opportunities 1:N stage_history, 1:N activities
--   opportunities 1:0..1 historical_bookings  (only won deals have a booking)
--   opportunities 1:0..1 opportunity_outcomes (ground truth, kept separate)
--   forecast_snapshots: one row per (snapshot, open opportunity)
-- ====================================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS current_forecast;
DROP TABLE IF EXISTS forecast_snapshots;
DROP TABLE IF EXISTS opportunity_outcomes;
DROP TABLE IF EXISTS historical_bookings;
DROP TABLE IF EXISTS quotas;
DROP TABLE IF EXISTS activities;
DROP TABLE IF EXISTS stage_history;
DROP TABLE IF EXISTS opportunities;
DROP TABLE IF EXISTS stages;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS rep_territory_history;
DROP TABLE IF EXISTS sales_reps;
DROP TABLE IF EXISTS territories;

CREATE TABLE territories (
    territory_id   TEXT PRIMARY KEY,              -- TER-001
    territory_name TEXT NOT NULL,
    region         TEXT NOT NULL CHECK (region IN ('North','South','East','West','Central'))
);

CREATE TABLE sales_reps (
    rep_id                 TEXT PRIMARY KEY,      -- REP-000001
    rep_name               TEXT NOT NULL,
    region                 TEXT NOT NULL,
    territory_id           TEXT NOT NULL REFERENCES territories(territory_id),  -- current
    team                   TEXT NOT NULL,
    joining_date           TEXT NOT NULL,
    historical_performance REAL CHECK (historical_performance BETWEEN 1 AND 5)
);

-- Bridge table: which territory a rep covered, and when (effective dates).
CREATE TABLE rep_territory_history (
    assignment_id TEXT PRIMARY KEY,               -- RTA-000001
    rep_id        TEXT NOT NULL REFERENCES sales_reps(rep_id),
    territory_id  TEXT NOT NULL REFERENCES territories(territory_id),
    start_date    TEXT NOT NULL,
    end_date      TEXT                             -- NULL = still current
);

CREATE TABLE accounts (
    account_id   TEXT PRIMARY KEY,                -- ACC-000001
    account_name TEXT NOT NULL UNIQUE,
    industry     TEXT NOT NULL,
    region       TEXT NOT NULL,
    territory_id TEXT NOT NULL REFERENCES territories(territory_id),
    account_size TEXT NOT NULL CHECK (account_size IN ('Small','Medium','Large','Enterprise'))
);

CREATE TABLE stages (
    stage_id            TEXT PRIMARY KEY,         -- STG-01
    stage_name          TEXT NOT NULL UNIQUE,
    stage_order         INTEGER NOT NULL,
    default_probability REAL NOT NULL CHECK (default_probability BETWEEN 0 AND 1),
    is_closed           INTEGER NOT NULL CHECK (is_closed IN (0,1))
);

-- Current state of each opportunity at the data export date.
-- The final outcome is NOT stored here; see opportunity_outcomes.
CREATE TABLE opportunities (
    opportunity_id           TEXT PRIMARY KEY,    -- OPP-2026-000001
    account_id               TEXT NOT NULL REFERENCES accounts(account_id),
    rep_id                   TEXT NOT NULL REFERENCES sales_reps(rep_id),
    product                  TEXT NOT NULL,
    opportunity_created_date TEXT NOT NULL,
    expected_close_date      TEXT NOT NULL,
    opportunity_stage        TEXT NOT NULL REFERENCES stages(stage_name),
    deal_amount              REAL NOT NULL CHECK (deal_amount >= 0),
    currency                 TEXT NOT NULL,
    current_probability      REAL NOT NULL CHECK (current_probability BETWEEN 0 AND 1),
    is_closed                INTEGER NOT NULL CHECK (is_closed IN (0,1)),
    last_modified_date       TEXT NOT NULL,
    CHECK (expected_close_date >= opportunity_created_date)
);

-- Every stage change / close-date change is a NEW row. History is never overwritten.
CREATE TABLE stage_history (
    stage_history_id    TEXT PRIMARY KEY,         -- STG-OPP-2026-000001-001
    opportunity_id      TEXT NOT NULL REFERENCES opportunities(opportunity_id),
    change_date         TEXT NOT NULL,
    event_type          TEXT NOT NULL CHECK (event_type IN ('stage_change','close_date_change')),
    stage               TEXT NOT NULL REFERENCES stages(stage_name),
    stage_probability   REAL NOT NULL,
    expected_close_date TEXT NOT NULL
);

CREATE TABLE activities (
    activity_id    TEXT PRIMARY KEY,              -- ACT-00000001
    opportunity_id TEXT NOT NULL REFERENCES opportunities(opportunity_id),
    rep_id         TEXT NOT NULL REFERENCES sales_reps(rep_id),
    activity_date  TEXT NOT NULL,
    activity_type  TEXT NOT NULL CHECK (activity_type IN ('Call','Email','Meeting','Follow-up')),
    activity_count INTEGER NOT NULL CHECK (activity_count >= 1)
);

CREATE TABLE quotas (
    quota_id      TEXT PRIMARY KEY,               -- QUOTA-REP-000001-2026Q1
    rep_id        TEXT NOT NULL REFERENCES sales_reps(rep_id),
    quota_quarter TEXT NOT NULL,
    quarter_start TEXT NOT NULL,
    quarter_end   TEXT NOT NULL,
    quota_amount  REAL NOT NULL CHECK (quota_amount > 0),
    currency      TEXT NOT NULL
);

-- Derived from won opportunities (one booking per won deal).
CREATE TABLE historical_bookings (
    booking_id      TEXT PRIMARY KEY,             -- BKG-000001
    opportunity_id  TEXT NOT NULL UNIQUE REFERENCES opportunities(opportunity_id),
    rep_id          TEXT NOT NULL REFERENCES sales_reps(rep_id),
    account_id      TEXT NOT NULL REFERENCES accounts(account_id),
    territory_id    TEXT NOT NULL REFERENCES territories(territory_id),
    booking_date    TEXT NOT NULL,
    booking_quarter TEXT NOT NULL,
    booking_amount  REAL NOT NULL,
    currency        TEXT NOT NULL
);

-- GROUND TRUTH: used only for training labels and evaluation, never as a feature.
CREATE TABLE opportunity_outcomes (
    opportunity_id              TEXT PRIMARY KEY REFERENCES opportunities(opportunity_id),
    final_status                TEXT NOT NULL CHECK (final_status IN ('Won','Lost','Open')),
    close_date                  TEXT,
    won_amount                  REAL NOT NULL,
    initial_expected_close_date TEXT NOT NULL,
    slipped                     INTEGER
);

-- What was known about each open deal on each snapshot date.
CREATE TABLE forecast_snapshots (
    snapshot_id              TEXT NOT NULL,       -- FCST-2026-03-31-001
    snapshot_date            TEXT NOT NULL,
    opportunity_id           TEXT NOT NULL REFERENCES opportunities(opportunity_id),
    account_id               TEXT NOT NULL,
    rep_id                   TEXT NOT NULL,
    product                  TEXT NOT NULL,
    opportunity_created_date TEXT NOT NULL,
    stage                    TEXT NOT NULL,
    stage_probability        REAL NOT NULL,
    stage_as_of_date         TEXT NOT NULL,
    expected_close_date      TEXT NOT NULL,
    deal_amount              REAL NOT NULL,
    currency                 TEXT NOT NULL,
    weighted_amount          REAL NOT NULL,
    PRIMARY KEY (snapshot_id, opportunity_id),
    CHECK (stage_as_of_date <= snapshot_date)
);
