from polars_tsutils.gaps import detect_gaps, flag_gaps, coverage

from datetime import datetime
import polars as pl
import pytest


class TestDetectGaps:
    def test_no_gaps_returns_empty(self, regular_df):
        gaps = detect_gaps(regular_df, "ts", "5m")
        assert gaps.height == 0

    def test_finds_single_gap(self, df_with_gap):
        gaps = detect_gaps(df_with_gap, "ts", "5m")
        assert gaps.height == 1

    def test_gap_columns_present(self, df_with_gap):
        gaps = detect_gaps(df_with_gap, "ts", "5m")
        assert set(gaps.columns) == {"gap_start", "gap_end", "gap_seconds", "missing_periods"}

    def test_gap_duration_correct(self, df_with_gap):
        gaps = detect_gaps(df_with_gap, "ts", "5m")
        # Gap from 00:25 to 00:40 = 15 minutes = 900 seconds
        assert gaps["gap_seconds"][0] == pytest.approx(900.0)

    def test_missing_periods_correct(self, df_with_gap):
        gaps = detect_gaps(df_with_gap, "ts", "5m")
        # 900s gap / 300s freq - 1 = 2.0 missing periods
        assert gaps["missing_periods"][0] == pytest.approx(2.0)

    def test_threshold_sensitivity(self, df_with_gap):
        # threshold=3.0 means only gaps > 15 min are reported (our gap is exactly 15 min)
        gaps_strict = detect_gaps(df_with_gap, "ts", "5m", threshold=3.1)
        assert gaps_strict.height == 0

        gaps_loose = detect_gaps(df_with_gap, "ts", "5m", threshold=1.5)
        assert gaps_loose.height == 1


class TestFlagGaps:
    def test_no_flags_on_regular_series(self, regular_df):
        result = flag_gaps(regular_df, "ts", "5m")
        assert result["gap_after"].sum() == 0

    def test_flags_row_before_gap(self, df_with_gap):
        result = flag_gaps(df_with_gap, "ts", "5m")
        # Row at index 5 (00:25) is immediately before the gap
        flagged = result.filter(pl.col("gap_after"))
        assert flagged.height == 1
        assert flagged["ts"][0] == datetime(2024, 1, 1, 0, 25)

    def test_custom_col_name(self, df_with_gap):
        result = flag_gaps(df_with_gap, "ts", "5m", col_name="has_gap")
        assert "has_gap" in result.columns

    def test_last_row_never_flagged(self, df_with_gap):
        result = flag_gaps(df_with_gap, "ts", "5m")
        assert result["gap_after"][-1] == False


class TestCoverage:
    def test_perfect_coverage(self, regular_df):
        assert coverage(regular_df, "ts", "5m") == pytest.approx(1.0)

    def test_coverage_with_gap(self, df_with_gap):
        cov = coverage(df_with_gap, "ts", "5m")
        # 10 rows present, 12 expected over the 55-min span (0..55 in 5-min steps = 12)
        # 00:00 to 01:00 with gap: expected = 13 timestamps, actual = 10
        assert 0.0 < cov < 1.0

    def test_single_row_coverage(self):
        df = pl.DataFrame({"ts": [datetime(2024, 1, 1)], "v": [1.0]})
        assert coverage(df, "ts", "5m") == pytest.approx(1.0)