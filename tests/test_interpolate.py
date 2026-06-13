from polars_tsutils.interpolate import fill_zoh, seed_at_boundary

from datetime import datetime
import polars as pl
import pytest


class TestFillZoh:
    def test_fills_interior_nulls(self, df_with_nulls):
        result = fill_zoh(df_with_nulls, ["power"])
        assert result["power"].null_count() == 0

    def test_leading_nulls_remain(self):
        df = pl.DataFrame({
            "ts": [datetime(2026, 1, 1, 0, i) for i in range(4)],
            "power": [None, None, 100.0, 200.0],
        })
        result = fill_zoh(df, ["power"])
        # Leading nulls cannot be forward-filled
        assert result["power"][0] is None
        assert result["power"][1] is None
        assert result["power"][2] == pytest.approx(100.0)
        assert result["power"][3] == pytest.approx(200.0)

    def test_zoh_carries_correct_value(self, df_with_nulls):
        result = fill_zoh(df_with_nulls, ["power"])
        # Original: [100, None, None, 130, ...]
        # After ZOH: [100, 100, 100, 130, ...]
        assert result["power"][1] == pytest.approx(100.0)
        assert result["power"][2] == pytest.approx(100.0)

    def test_limit_caps_fill_length(self, df_with_nulls):
        result = fill_zoh(df_with_nulls, ["power"], limit=1)
        # Only one null after 100 should be filled
        assert result["power"][1] == pytest.approx(100.0)
        assert result["power"][2] is None  # second consecutive null not filled

    def test_multiple_cols(self, df_with_nulls):
        df = df_with_nulls.with_columns(pl.col("power").alias("voltage"))
        result = fill_zoh(df, ["power", "voltage"])
        assert result["voltage"].null_count() == 0


class TestSeedAtBoundary:
    def test_inserts_seed_row(self, irregular_df):
        boundary = datetime(2026, 1, 1, 0, 5)
        result = seed_at_boundary(irregular_df, "ts", ["power"], boundary)
        assert result.filter(pl.col("ts") == boundary).height == 1

    def test_seed_carries_last_known_value(self, irregular_df):
        boundary = datetime(2026, 1, 1, 0, 5)
        result = seed_at_boundary(irregular_df, "ts", ["power"], boundary)
        seed_power = result.filter(pl.col("ts") == boundary)["power"][0]
        # Last measurement before 00:05 is at 00:03 with power=200
        assert seed_power == pytest.approx(200.0)

    def test_no_duplicate_if_already_exists(self, irregular_df):
        # 00:00 already exists in irregular_df
        boundary = datetime(2026, 1, 1, 0, 0)
        result = seed_at_boundary(irregular_df, "ts", ["power"], boundary)
        assert result.filter(pl.col("ts") == boundary).height == 1

    def test_returns_unchanged_if_no_prior_data(self, irregular_df):
        boundary = datetime(2023, 12, 31)  # before all data
        result = seed_at_boundary(irregular_df, "ts", ["power"], boundary)
        assert result.height == irregular_df.height

    def test_result_is_sorted(self, irregular_df):
        boundary = datetime(2026, 1, 1, 0, 5)
        result = seed_at_boundary(irregular_df, "ts", ["power"], boundary)
        diffs = result["ts"].diff().drop_nulls().dt.total_seconds()
        assert (diffs >= 0).all()
