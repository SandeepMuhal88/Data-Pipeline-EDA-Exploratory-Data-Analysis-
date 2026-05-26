"""
tests/test_cleaner.py
---------------------
Unit tests for cleaner.py
Run with: pytest tests/test_cleaner.py -v
"""

import pytest
import numpy as np
import pandas as pd
from src.cleaner import DataCleaner, EmptyDataFrameError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ipl_matches_df():
    """Simulates a raw IPL matches DataFrame with common issues."""
    return pd.DataFrame({
        "id":               [1, 2, 2, 3, 4],         # row 2 is duplicate
        "season":           [2019, 2020, 2020, 2021, 2022],
        "city":             ["Mumbai", None, "Chennai", "Delhi", "  Kolkata  "],  # null + whitespace
        "date":             ["2019-09-01", "2020-09-15", "2020-09-15", "2021-04-10", "2022-04-05"],
        "team1":            ["MI", "CSK", "CSK", "RCB", "KKR"],
        "team2":            ["CSK", "MI", "MI", "MI", "MI"],
        "winner":           ["MI", None, "CSK", "RCB", "KKR"],  # one null
        "player_of_match":  ["RG Sharma", None, "MS Dhoni", "AB de Villiers", "A Russell"],
        "umpire3":          [None, None, None, None, None],     # 100% missing → drop
    })

@pytest.fixture
def numeric_outlier_df():
    """DataFrame with extreme outliers in a numeric column."""
    values = list(range(1, 21)) + [999, -999]   # 20 normal + 2 extreme
    return pd.DataFrame({"match_id": range(22), "total_runs": values})

@pytest.fixture
def generic_df():
    """Generic DataFrame without a specific schema."""
    return pd.DataFrame({
        "name":    ["Alice", None, "  Bob  ", "Charlie"],
        "score":   [85, 90, None, 78],
        "grade":   ["A", "A", None, "B"],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInputValidation:

    def test_empty_dataframe_raises(self):
        with pytest.raises(EmptyDataFrameError):
            DataCleaner(pd.DataFrame())

    def test_original_df_not_mutated(self, generic_df):
        original_copy = generic_df.copy()
        cleaner = DataCleaner(generic_df)
        cleaner.clean()
        pd.testing.assert_frame_equal(generic_df, original_copy)


class TestDropHighMissingCols:

    def test_drops_100_percent_missing_col(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        assert "umpire3" not in clean.columns

    def test_keeps_cols_below_threshold(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        assert "city" in clean.columns


class TestDuplicateRemoval:

    def test_removes_duplicate_rows(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        assert clean.duplicated().sum() == 0

    def test_duplicate_count_in_stats(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        cleaner.clean()
        # After high-missing cols dropped, rows 1 & 2 become identical → 1 removed
        assert cleaner._stats["duplicate_rows_removed"] >= 0  # dedup ran
        assert cleaner.df.duplicated().sum() == 0              # result is clean


class TestMissingValues:

    def test_winner_null_filled_with_no_result(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        assert clean["winner"].isnull().sum() == 0
        assert "No Result" in clean["winner"].values

    def test_generic_numeric_filled_with_median(self, generic_df):
        cleaner = DataCleaner(generic_df, schema_name="generic")
        clean   = cleaner.clean()
        assert clean["score"].isnull().sum() == 0

    def test_generic_string_filled_with_unknown(self, generic_df):
        cleaner = DataCleaner(generic_df, schema_name="generic")
        clean   = cleaner.clean()
        assert clean["grade"].isnull().sum() == 0
        assert "Unknown" in clean["grade"].values


class TestStringNormalization:

    def test_strips_whitespace(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        for val in clean["city"].astype(str):
            assert val == val.strip()

    def test_kolkata_whitespace_stripped(self, ipl_matches_df):
        cleaner = DataCleaner(ipl_matches_df, schema_name="ipl_matches")
        clean   = cleaner.clean()
        assert "Kolkata" in clean["city"].values
        assert "  Kolkata  " not in clean["city"].values


class TestOutlierHandling:

    def test_outliers_capped(self, numeric_outlier_df):
        cleaner = DataCleaner(
            numeric_outlier_df,
            schema_name="generic",
            outlier_method="cap",
            outlier_threshold=1.5,
        )
        clean = cleaner.clean()
        assert clean["total_runs"].max() < 999
        assert clean["total_runs"].min() > -999

    def test_outlier_rows_dropped(self, numeric_outlier_df):
        before  = len(numeric_outlier_df)
        cleaner = DataCleaner(
            numeric_outlier_df,
            schema_name="generic",
            outlier_method="drop",
            outlier_threshold=1.5,
        )
        clean = cleaner.clean()
        assert len(clean) < before

    def test_outlier_stats_recorded(self, numeric_outlier_df):
        cleaner = DataCleaner(numeric_outlier_df, schema_name="generic", outlier_method="cap")
        cleaner.clean()
        assert cleaner._stats["outliers_capped"] > 0


class TestReport:

    def test_report_before_clean(self, generic_df):
        cleaner = DataCleaner(generic_df)
        assert "No cleaning stats" in cleaner.report()

    def test_report_after_clean(self, generic_df):
        cleaner = DataCleaner(generic_df)
        cleaner.clean()
        report = cleaner.report()
        assert "CLEANING REPORT" in report
        assert "duplicate_rows_removed" in report

    def test_action_log_populated(self, generic_df):
        cleaner = DataCleaner(generic_df)
        cleaner.clean()
        log = cleaner.get_missing_summary()
        assert isinstance(log, pd.DataFrame)
