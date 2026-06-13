from polars_tsutils.rolling import rolling_twa, rolling_zscore

from datetime import datetime
import polars as pl
import pytest


def _make_step_df():
    ts = [datetime(2026, 1, 1, 0, i) for i in range(21)]
    power = [100.0] * 10 + [200.0] * 11
    return pl.DataFrame({"ts": ts, "power": power})


class TestRollingTwa:
    def test_output_col_appended(self, regular_df):
        result = rolling_twa(regular_df, "ts", "power", "10m")
        assert "twa_power" in result.columns

    def test_custom_output_col(self, regular_df):
        result = rolling_twa(regular_df, "ts", "power", "10m", output_col="p_twa")
        assert "p_twa" in result.columns

    def test_stable_signal_equals_value(self):
        # For a constant signal, rolling TWA should equal to that constant.
        df = pl.DataFrame({
            "ts": [datetime(2026, 1, 1, 0, i * 5) for i in range(10)],
            "power": [150.0] * 10,
        })
        result = rolling_twa(df, "ts", "power", "15m")
        non_null = result["twa_power"].drop_nulls()
        assert all(abs(v - 150.0) < 1e-9 for v in non_null.to_list())

    def test_twa_lags_step_change(self):
        """
        The rolling average looks back 10 minutes, so right after the step it still includes 
        old low values and only reaches 200 after a full 10 minutes have passed.
        """
        df = _make_step_df()
        result = rolling_twa(df, "ts", "power", "10m")
        # At t=10 (just after step), TWA should be <200 because window still sees 100s
        twa_at_10 = result.filter(pl.col("ts") == datetime(2026, 1, 1, 0, 10))["twa_power"][0]
        assert twa_at_10 is not None
        assert twa_at_10 < 200.0

        # At t=20, full 10-min window is all 200s → TWA = 200
        twa_at_20 = result.filter(pl.col("ts") == datetime(2026, 1, 1, 0, 20))["twa_power"][0]
        assert twa_at_20 == pytest.approx(200.0, rel=1e-6)

    def test_min_weight_suppresses_early_results(self, regular_df):
        result = rolling_twa(
            regular_df.head(3), "ts", "power", "30m", min_weight=1500
        )
        # First few rows don't have 1500s (25 min) of data in window → null
        assert result["twa_power"][0] is None


class TestRollingZscore:
    def test_output_col_appended(self, regular_df):
        result = rolling_zscore(regular_df, "ts", "power", "30m")
        assert "zscore_power" in result.columns

    def test_stable_signal_zscore_near_zero(self):
        # Constant signal has zero std → z-score should be 0.
        df = pl.DataFrame({
            "ts": [datetime(2026, 1, 1, 0, i * 5) for i in range(10)],
            "power": [100.0] * 10,
        })
        result = rolling_zscore(df, "ts", "power", "30m")
        non_null = result["zscore_power"].drop_nulls()
        assert all(abs(v) < 1e-6 for v in non_null.to_list())

    def test_outlier_has_high_zscore(self):
        # An outlier should produce a large |z-score|.
        ts = [datetime(2026, 1, 1, 0, i * 5) for i in range(12)]
        power = [100.0] * 6 + [100.0, 100.0, 100.0, 100.0, 100.0, 1000.0]
        df = pl.DataFrame({"ts": ts, "power": power})
        result = rolling_zscore(df, "ts", "power", "30m")
        last_z = result["zscore_power"][-1]
        assert last_z is not None
        assert abs(last_z) > 2.0

    def test_min_periods_produces_nulls_at_start(self, regular_df):
        result = rolling_zscore(regular_df.head(5), "ts", "power", "1h", min_periods=4)
        # With a 1h window and min_periods=4, first few rows may not have enough data
        # At minimum, the first row should be null (only 1 point in window)
        assert result["zscore_power"][0] is None
