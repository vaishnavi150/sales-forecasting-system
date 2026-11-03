"""
tests/conftest.py - shared test data ("fixtures") for all test files.

pytest runs the functions below once and passes their results to any
test that asks for them by name. We use a TINY dataset so the tests run
in seconds; the generator logic is identical to the starter/full scales.
"""

import pytest

import config
from config import ScaleConfig
from src.data_generation import generate_all

TINY = ScaleConfig(name="tiny", num_territories=5, num_reps=25,
                   num_accounts=80, num_opportunities=900, num_error_rows=40)


@pytest.fixture(scope="session")
def tiny_scale():
    return TINY


@pytest.fixture(scope="session")
def tiny_tables():
    """Clean tiny dataset + ground truth (generated once per test run)."""
    return generate_all(TINY, seed=config.RANDOM_SEED)
