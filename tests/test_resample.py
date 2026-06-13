from polars_tsutils.resample import resample_twa, upsample_zoh

from datetime import datetime
import polars as pl
import pytest


class TestResampleTwa:
    def test_exact_boundary_uniform(self, regular_df):
        result = resample_twa(regular_df, "ts", "5m", ["power"])
        # Each bucket has exactly one measurement; TWA = value
        assert result.height == 12
        assert result["power"][0] == pytest.approx(100.0)

    def test_twa_correct_weighting(self, irregular_df):
        """
        Verify TWA for first 5-minute bucket [00:00, 00:05):

        Row 0: power=100 active from 00:00 → 00:03 = 180s
        Row 1: power=200 active from 00:03 → 00:05 = 120s  (next is 00:07, clipped to 00:05)

        TWA = (100*180 + 200*120) / 300 = (18000 + 24000) / 300 = 140.0
        """
        result = resample_twa(irregular_df, "ts", "5m", ["power"])
        first_bucket = result.filter(
            pl.col("ts") == datetime(2026, 1, 1, 0, 0)
        )["power"][0]
        assert first_bucket == pytest.approx(140.0, rel=1e-6)

    def test_output_has_correct_columns(self, regular_df):
        result = resample_twa(regular_df, "ts", "5m", ["power", "voltage"])
        assert set(result.columns) == {"ts", "power", "voltage"}

    def test_label_right(self, irregular_df):
        result = resample_twa(irregular_df, "ts", "5m", ["power"], label="right")
        # First bucket label should be 00:05 with label='right'
        assert result["ts"][0] == datetime(2026, 1, 1, 0, 5)

    def test_label_invalid(self, irregular_df):
        with pytest.raises(ValueError, match="label must be"):
            resample_twa(irregular_df, "ts", "5m", ["power"], label="center")

    def test_no_data_loss(self, regular_df):
        # All rows should produce a non-null TWA bucket.
        result = resample_twa(regular_df, "ts", "5m", ["power"])
        assert result["power"].null_count() == 0

    def test_cross_bucket_measurement(self):
        df = pl.DataFrame({
            "ts": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 10)],
            "power": [100.0, 200.0],
        })
        result = resample_twa(df, "ts", "5m", ["power"])
        # Bucket [00:00, 00:05): power=100 for full 5 min → TWA = 100
        # Bucket [00:05, 00:10): power=100 still in effect → TWA = 100
        b0 = result.filter(pl.col("ts") == datetime(2026, 1, 1, 0, 0))["power"][0]
        b1 = result.filter(pl.col("ts") == datetime(2026, 1, 1, 0, 5))["power"][0]
        assert b0 == pytest.approx(100.0)
        assert b1 == pytest.approx(100.0)


class TestUpsampleZoh:
    def test_regular_series_unchanged(self, regular_df):
        result = upsample_zoh(regular_df, "ts", "5m", ["power"])
        assert result.height == regular_df.height

    def test_fills_missing_timestamps(self):
        df = pl.DataFrame({
            "ts": [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 10)],
            "power": [100.0, 200.0],
        })
        result = upsample_zoh(df, "ts", "5m", ["power"])
        assert result.height == 3  # 00:00, 00:05, 00:10
        # 00:05 should be ZOH-filled with 100.0
        mid = result.filter(pl.col("ts") == datetime(2026, 1, 1, 0, 5))["power"][0]
        assert mid == pytest.approx(100.0)

    def test_default_fills_all_non_time_cols(self, regular_df):
        result = upsample_zoh(regular_df, "ts", "5m")
        assert "power" in result.columns
        assert "voltage" in result.columns

    def test_no_leading_null_when_data_starts_at_boundary(self, regular_df):
        result = upsample_zoh(regular_df, "ts", "5m", ["power"])
        assert result["power"].null_count() == 0
