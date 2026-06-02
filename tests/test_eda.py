"""
tests/test_eda_engine.py
------------------------
Unit tests for eda_engine.py
Run with: pytest tests/test_eda_engine.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.eda_engine import EDAEngine, EmptyDataFrameError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ipl_matches_df():
    return pd.DataFrame({
        "season":         [2019, 2020, 2020, 2021, 2022, 2022, 2019, 2021],
        "city":           ["Mumbai","Chennai","Delhi","Kolkata","Mumbai","Chennai","Delhi","Kolkata"],
        "team1":          ["MI","CSK","RCB","KKR","MI","CSK","RCB","KKR"],
        "team2":          ["CSK","MI","KKR","RCB","CSK","MI","KKR","RCB"],
        "winner":         ["MI","CSK","RCB","KKR","MI","CSK","RCB","KKR"],
        "toss_decision":  ["bat","field","bat","field","bat","field","bat","field"],
        "date":           pd.to_datetime(["2019-04-01","2020-09-01","2020-10-01",
                                          "2021-04-01","2022-03-27","2022-04-01",
                                          "2019-05-01","2021-05-01"]),
    })

@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "runs":    np.random.randint(0, 200, 100),
        "wickets": np.random.randint(0, 10,  100),
        "overs":   np.random.uniform(1, 20,  100),
        "team":    np.random.choice(["MI","CSK","RCB"], 100),
    })

@pytest.fixture
def single_col_df():
    return pd.DataFrame({"name": ["Alice","Bob","Charlie","Dave"]})

@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "eda_output")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInputValidation:

    def test_empty_df_raises(self, output_dir):
        with pytest.raises(EmptyDataFrameError):
            EDAEngine(pd.DataFrame(), output_dir=output_dir)

    def test_original_df_not_mutated(self, numeric_df, output_dir):
        original = numeric_df.copy()
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        pd.testing.assert_frame_equal(numeric_df, original)


class TestOutputDir:

    def test_creates_output_dir(self, numeric_df, tmp_path):
        new_dir = str(tmp_path / "brand_new_dir")
        eda = EDAEngine(numeric_df, output_dir=new_dir)
        eda.run()
        assert Path(new_dir).exists()

    def test_charts_saved_as_png(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        results = eda.run()
        for path in results["charts"]:
            assert Path(path).exists()
            assert path.endswith(".png")


class TestDescriptiveStats:

    def test_stats_computed(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        assert eda._stats["numeric_col_count"] == 3
        assert eda._stats["categorical_col_count"] >= 1

    def test_describe_in_stats(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        assert "describe" in eda._stats
        assert isinstance(eda._stats["describe"], pd.DataFrame)

    def test_row_col_counts(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        assert eda._stats["total_rows"] == 100
        assert eda._stats["total_cols"] == 4


class TestMissingHeatmap:

    def test_skipped_when_no_nulls(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert not any("missing" in c for c in charts)

    def test_saved_when_nulls_exist(self, numeric_df, output_dir):
        df = numeric_df.copy()
        df.loc[0:10, "runs"] = np.nan
        eda = EDAEngine(df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("missing" in c for c in charts)


class TestDistributionPlots:

    def test_distribution_chart_created(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("distribution" in c for c in charts)

    def test_skipped_for_no_numeric_cols(self, single_col_df, output_dir):
        eda = EDAEngine(single_col_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert not any("distribution" in c for c in charts)


class TestCorrelationHeatmap:

    def test_correlation_chart_created(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("correlation" in c for c in charts)

    def test_correlation_matrix_in_stats(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        assert "correlation_matrix" in eda._stats
        corr = eda._stats["correlation_matrix"]
        assert corr.shape == (3, 3)           # 3 numeric cols

    def test_skipped_for_single_numeric(self, single_col_df, output_dir):
        df = pd.DataFrame({"score": [1, 2, 3], "name": ["a","b","c"]})
        eda = EDAEngine(df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert not any("correlation" in c for c in charts)


class TestCategoricalPlots:

    def test_categorical_charts_created(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("categorical" in c for c in charts)

    def test_ipl_schema_categorical_cols(self, ipl_matches_df, output_dir):
        eda = EDAEngine(ipl_matches_df, output_dir=output_dir, schema_name="ipl_matches")
        eda.run()
        assert "winner" in eda.get_categorical_cols()
        assert "team1"  in eda.get_categorical_cols()


class TestBoxPlots:

    def test_boxplot_chart_created(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("boxplot" in c for c in charts)


class TestTimeSeries:

    def test_timeseries_saved_for_ipl(self, ipl_matches_df, output_dir):
        eda = EDAEngine(ipl_matches_df, output_dir=output_dir, schema_name="ipl_matches")
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert any("timeseries" in c for c in charts)

    def test_timeseries_skipped_for_generic_no_date(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir, schema_name="generic")
        eda.run()
        charts = [Path(p).name for p in eda._saved_charts]
        assert not any("timeseries" in c for c in charts)


class TestSummary:

    def test_summary_before_run(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        assert "No EDA run" in eda.summary()

    def test_summary_after_run(self, numeric_df, output_dir):
        eda = EDAEngine(numeric_df, output_dir=output_dir)
        eda.run()
        s = eda.summary()
        assert "EDA SUMMARY" in s
        assert "Charts saved" in s

    def test_run_returns_dict(self, numeric_df, output_dir):
        eda     = EDAEngine(numeric_df, output_dir=output_dir)
        results = eda.run()
        assert isinstance(results, dict)
        assert "stats"  in results
        assert "charts" in results
        assert isinstance(results["charts"], list)