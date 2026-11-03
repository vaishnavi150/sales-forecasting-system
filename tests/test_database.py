"""Tests for src/database.py - the schema accepts clean data and enforces rules."""

import sqlite3

import pytest

from src import database


def test_clean_data_loads_with_valid_foreign_keys(tiny_tables, tmp_path):
    db = tmp_path / "test.db"
    counts = database.create_database(tiny_tables, db_path=db)
    assert counts["opportunities"] == len(tiny_tables["opportunities"])
    assert counts["activities"] == len(tiny_tables["activities"])
    df = database.query("SELECT COUNT(*) AS n FROM opportunities WHERE is_closed = 1", db_path=db)
    assert df.loc[0, "n"] == int(tiny_tables["opportunities"]["is_closed"].sum())


def test_database_rejects_negative_amount(tiny_tables, tmp_path):
    db = tmp_path / "test.db"
    database.create_database(tiny_tables, db_path=db)
    with database.connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE opportunities SET deal_amount = -1 WHERE rowid = 1")


def test_database_rejects_unknown_rep(tiny_tables, tmp_path):
    db = tmp_path / "test.db"
    database.create_database(tiny_tables, db_path=db)
    with database.connect(db) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE opportunities SET rep_id = 'REP-999999' WHERE rowid = 1")
