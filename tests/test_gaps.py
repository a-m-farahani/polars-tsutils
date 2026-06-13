from polars_tsutils.gaps import detect_gaps

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

