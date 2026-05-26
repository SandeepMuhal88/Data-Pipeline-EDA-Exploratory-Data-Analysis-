"""
tests/test_loader.py
--------------------
Unit tests for data_loader.py
Run with: pytest tests/test_loader.py -v
"""

import pytest
import sqlite3
import json
import pandas as pd
from pathlib import Path
from src.data_loader import (
    DataLoader,
    FileNotFoundError,
    UnsupportedFormatError,
    EmptyDatasetError,
    SchemaValidationError,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a minimal IPL-style CSV file."""
    df = pd.DataFrame({
        "id":     [1, 2],
        "season": [2020, 2021],
        "city":   ["Mumbai", "Chennai"],
        "date":   ["2020-09-19", "2021-04-10"],
        "team1":  ["MI", "CSK"],
        "team2":  ["CSK", "MI"],
        "winner": ["MI", "CSK"],
    })
    path = tmp_path / "ipl_matches.csv"
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def sample_json(tmp_path):
    data = [{"match_id": 1, "inning": 1, "batting_team": "MI",
              "bowling_team": "CSK", "over": 1, "ball": 1,
              "batsman": "RG Sharma", "bowler": "DJ Bravo", "total_runs": 4}]
    path = tmp_path / "ipl_deliveries.json"
    path.write_text(json.dumps(data))
    return path

@pytest.fixture
def sample_sqlite(tmp_path):
    path = tmp_path / "ipl.db"
    conn = sqlite3.connect(path)
    pd.DataFrame({"id": [1], "season": [2020], "city": ["Mumbai"],
                  "date": ["2020-09-19"], "team1": ["MI"],
                  "team2": ["CSK"], "winner": ["MI"]}).to_sql("matches", conn, index=False)
    conn.close()
    return path

@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,season,city\n")   # header only, no rows
    return path

# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFileValidation:

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent.csv").load()

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("x")
        with pytest.raises(UnsupportedFormatError):
            DataLoader(str(f)).load()

    def test_empty_csv_raises(self, empty_csv):
        with pytest.raises(EmptyDatasetError):
            DataLoader(str(empty_csv)).load()


class TestCSVLoading:

    def test_loads_csv_successfully(self, sample_csv):
        df = DataLoader(str(sample_csv)).load()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 7)

    def test_schema_validation_passes(self, sample_csv):
        df = DataLoader(str(sample_csv), schema_name="ipl_matches").load()
        assert "winner" in df.columns

    def test_schema_validation_fails_on_missing_col(self, tmp_path):
        bad = pd.DataFrame({"id": [1], "season": [2020]})
        p = tmp_path / "bad.csv"
        bad.to_csv(p, index=False)
        with pytest.raises(SchemaValidationError):
            DataLoader(str(p), schema_name="ipl_matches").load()


class TestJSONLoading:

    def test_loads_json_successfully(self, sample_json):
        df = DataLoader(str(sample_json), schema_name="ipl_deliveries").load()
        assert df.shape[0] == 1
        assert "batsman" in df.columns


class TestSQLiteLoading:

    def test_loads_sqlite_successfully(self, sample_sqlite):
        df = DataLoader(str(sample_sqlite), schema_name="ipl_matches").load()
        assert df.shape[0] == 1
        assert "winner" in df.columns


class TestSummary:

    def test_summary_before_load(self):
        loader = DataLoader("dummy.csv")
        assert "No dataset loaded" in loader.summary()

    def test_summary_after_load(self, sample_csv):
        loader = DataLoader(str(sample_csv))
        loader.load()
        summary = loader.summary()
        assert "ipl_matches.csv" in summary
        assert "2" in summary     # 2 rows

    def test_action_log_populated(self, sample_csv):
        loader = DataLoader(str(sample_csv))
        loader.load()
        log = loader.get_action_log()
        assert "load_complete" in log
        assert "file_exists_ok" in log
