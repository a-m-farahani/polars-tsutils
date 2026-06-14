from polars_tsutils.align import make_grid, align_to_grid, project_to_grid, outer_join_resample

from datetime import datetime
import polars as pl
import pytest


def _ts(*args) -> datetime:
    return datetime(*args)


def _irregular_df():
    # Readings at 0, 3, 7, 12 minutes - not on a 5-min grid.
    return pl.DataFrame({
        "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,3),
               _ts(2026,1,1,0,7), _ts(2026,1,1,0,12)],
        "power": [100.0, 150.0, 200.0, 250.0],
    })


class TestMakeGrid:
    def test_correct_length(self):
        grid = make_grid(_ts(2026,1,1,0,0), _ts(2026,1,1,1,0), "15m")
        assert grid.height == 5  # 0, 15, 30, 45, 60

    def test_custom_col_name(self):
        grid = make_grid(_ts(2026,1,1), _ts(2026,1,1,0,10), "5m", time_col="timestamp")
        assert "timestamp" in grid.columns

    def test_first_and_last_values(self):
        grid = make_grid(_ts(2026,1,1,0,0), _ts(2026,1,1,0,10), "5m")
        assert grid["ts"][0] == _ts(2026,1,1,0,0)
        assert grid["ts"][-1] == _ts(2026,1,1,0,10)

    def test_single_point(self):
        grid = make_grid(_ts(2026,1,1), _ts(2026,1,1), "5m")
        assert grid.height == 1


class TestAlignToGrid:
    def test_nearest_snaps_correctly(self):
        df = _irregular_df()
        result = align_to_grid(df, "ts", "5m", method="nearest")
        # 00:03 → nearest 5-min grid point is 00:05
        assert _ts(2026,1,1,0,5) in result["ts"].to_list()
        # 00:07 → nearest is 00:05
        assert result["ts"].to_list().count(_ts(2026,1,1,0,5)) == 1  # deduplicated

    def test_floor_always_rounds_down(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,1), _ts(2026,1,1,0,4), _ts(2026,1,1,0,9)],
            "v": [1.0, 2.0, 3.0],
        })
        result = align_to_grid(df, "ts", "5m", method="floor", on_duplicate="last")
        # 00:01 and 00:04 both floor to 00:00; 00:09 floors to 00:05
        assert _ts(2026,1,1,0,0) in result["ts"].to_list()
        assert _ts(2026,1,1,0,5) in result["ts"].to_list()

    def test_ceil_always_rounds_up(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,1), _ts(2026,1,1,0,6)],
            "v": [1.0, 2.0],
        })
        result = align_to_grid(df, "ts", "5m", method="ceil")
        assert _ts(2026,1,1,0,5) in result["ts"].to_list()
        assert _ts(2026,1,1,0,10) in result["ts"].to_list()

    def test_ceil_exact_boundary_unchanged(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5)],
            "v": [1.0, 2.0],
        })
        result = align_to_grid(df, "ts", "5m", method="ceil")
        # Exact grid points should not be pushed forward
        assert _ts(2026,1,1,0,0) in result["ts"].to_list()
        assert _ts(2026,1,1,0,5) in result["ts"].to_list()

    def test_keep_original_preserves_timestamp(self):
        df = _irregular_df()
        result = align_to_grid(df, "ts", "5m", keep_original=True)
        assert "ts_original" in result.columns
        assert result["ts_original"][0] == _ts(2026,1,1,0,0)

    def test_on_duplicate_first(self):
        # 00:03 → 00:05 (2 min away), 00:04 → 00:05 (1 min away): both snap to 00:05
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,3), _ts(2026,1,1,0,4)],
            "v": [10.0, 20.0],
        })
        result = align_to_grid(df, "ts", "5m", method="nearest", on_duplicate="first")
        assert result.height == 1
        assert result["v"][0] == pytest.approx(10.0)

    def test_on_duplicate_last(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,3), _ts(2026,1,1,0,4)],
            "v": [10.0, 20.0],
        })
        result = align_to_grid(df, "ts", "5m", method="nearest", on_duplicate="last")
        assert result.height == 1
        assert result["v"][0] == pytest.approx(20.0)

    def test_on_duplicate_error_raises(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,3), _ts(2026,1,1,0,4)],
            "v": [10.0, 20.0],
        })
        with pytest.raises(ValueError, match="duplicate grid points"):
            align_to_grid(df, "ts", "5m", method="nearest", on_duplicate="error")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            align_to_grid(_irregular_df(), "ts", "5m", method="median")

    def test_already_on_grid_unchanged(self):
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5), _ts(2026,1,1,0,10)],
            "v": [1.0, 2.0, 3.0],
        })
        result = align_to_grid(df, "ts", "5m")
        assert result["ts"].to_list() == df["ts"].to_list()

    def test_result_is_sorted(self):
        df = _irregular_df()
        result = align_to_grid(df, "ts", "5m")
        diffs = result["ts"].diff().drop_nulls().dt.total_seconds()
        assert (diffs >= 0).all()


class TestProjectToGrid:
    def test_output_on_regular_grid(self):
        result = project_to_grid(_irregular_df(), "ts", "5m")
        diffs = result["ts"].diff().drop_nulls().dt.total_seconds()
        assert (diffs == 300.0).all()

    def test_zoh_fill_correct(self):
        # Grid point at 00:05 should carry the 00:03 measurement (power=150).
        result = project_to_grid(_irregular_df(), "ts", "5m")
        val_at_05 = result.filter(pl.col("ts") == _ts(2026,1,1,0,5))["power"][0]
        assert val_at_05 == pytest.approx(150.0)

    def test_custom_t_min_t_max(self):
        result = project_to_grid(
            _irregular_df(), "ts", "5m",
            t_min=_ts(2026,1,1,0,0), t_max=_ts(2026,1,1,0,10)
        )
        assert result["ts"][0] == _ts(2026,1,1,0,0)
        assert result["ts"][-1] == _ts(2026,1,1,0,10)

    def test_tolerance_produces_nulls_beyond_window(self):
        # With tolerance='4m', a grid point >4 min from any measurement gets null.
        df = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,20)],
            "power": [100.0, 200.0],
        })
        result = project_to_grid(df, "ts", "5m", tolerance="4m")
        # 00:05 is 5 min after 00:00 - beyond tolerance - should be null
        val_at_05 = result.filter(pl.col("ts") == _ts(2026,1,1,0,5))["power"][0]
        assert val_at_05 is None

    def test_selects_specified_value_cols(self):
        df = _irregular_df().with_columns(pl.lit(1.0).alias("extra"))
        result = project_to_grid(df, "ts", "5m", value_cols=["power"])
        assert "extra" not in result.columns
        assert "power" in result.columns



class TestOuterJoinResample:
    def _inv_df(self):
        # Inverter power sampled every 3 minutes.
        return pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,3), _ts(2026,1,1,0,6),
                   _ts(2026,1,1,0,9), _ts(2026,1,1,0,12)],
            "power": [100.0, 110.0, 120.0, 130.0, 140.0],
        })

    def _irr_df(self):
        # Irradiance sampled every 10 minutes.
        return pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,10)],
            "irradiance": [500.0, 600.0],
        })

    def test_output_on_regular_grid(self):
        result = outer_join_resample(self._inv_df(), self._irr_df(), "ts", "5m")
        diffs = result["ts"].diff().drop_nulls().dt.total_seconds()
        assert (diffs == 300.0).all()

    def test_both_signals_present(self):
        result = outer_join_resample(self._inv_df(), self._irr_df(), "ts", "5m")
        assert "power" in result.columns
        assert "irradiance" in result.columns

    def test_grid_spans_both_series(self):
        result = outer_join_resample(self._inv_df(), self._irr_df(), "ts", "5m")
        assert result["ts"].min() == _ts(2026,1,1,0,0)
        assert result["ts"].max() == _ts(2026,1,1,0,10)

    def test_conflicting_cols_disambiguated(self):
        """Both series have a 'value' column - should get suffixes."""
        df_a = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5)],
            "value": [1.0, 2.0],
        })
        df_b = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5)],
            "value": [10.0, 20.0],
        })
        result = outer_join_resample(df_a, df_b, "ts", "5m")
        assert "value_a" in result.columns
        assert "value_b" in result.columns
        assert "value" not in result.columns

    def test_custom_suffixes(self):
        df_a = pl.DataFrame({"ts": [_ts(2026,1,1,0,0)], "v": [1.0]})
        df_b = pl.DataFrame({"ts": [_ts(2026,1,1,0,0)], "v": [2.0]})
        result = outer_join_resample(df_a, df_b, "ts", "5m", suffixes=("_inv", "_irr"))
        assert "v_inv" in result.columns
        assert "v_irr" in result.columns

    def test_zoh_fill_from_each_series(self):
        result = outer_join_resample(self._inv_df(), self._irr_df(), "ts", "5m")
        # At 00:05, irradiance should be ZOH from 00:00 reading (500.0)
        irr_at_05 = result.filter(pl.col("ts") == _ts(2026,1,1,0,5))["irradiance"][0]
        assert irr_at_05 == pytest.approx(500.0)

    def test_custom_t_min_t_max(self):
        result = outer_join_resample(
            self._inv_df(), self._irr_df(), "ts", "5m",
            t_min=_ts(2026,1,1,0,0), t_max=_ts(2026,1,1,0,5),
        )
        assert result["ts"][-1] == _ts(2026,1,1,0,5)
        assert result.height == 2

    def test_non_overlapping_series(self):
        # Series that don't overlap in time - grid spans both, ZOH carries each forward.
        df_a = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5)],
            "power": [100.0, 110.0],
        })
        df_b = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,10), _ts(2026,1,1,0,15)],
            "irradiance": [500.0, 600.0],
        })
        result = outer_join_resample(df_a, df_b, "ts", "5m")
        assert result["ts"].min() == _ts(2026,1,1,0,0)
        assert result["ts"].max() == _ts(2026,1,1,0,15)
        # irradiance has no data before 00:10, so first two rows are null
        assert result.filter(pl.col("ts") == _ts(2026,1,1,0,0))["irradiance"][0] is None
        assert result.filter(pl.col("ts") == _ts(2026,1,1,0,5))["irradiance"][0] is None
        # ZOH carries power forward into irradiance-only period (correct behavior)
        assert result.filter(pl.col("ts") == _ts(2026,1,1,0,15))["power"][0] == pytest.approx(110.0)

    def test_tolerance_limits_zoh_carry_forward(self):
        # With tolerance, ZOH fill is capped - stale values become null.
        df_a = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,0), _ts(2026,1,1,0,5)],
            "power": [100.0, 110.0],
        })
        df_b = pl.DataFrame({
            "ts": [_ts(2026,1,1,0,10), _ts(2026,1,1,0,15)],
            "irradiance": [500.0, 600.0],
        })
        result = outer_join_resample(df_a, df_b, "ts", "5m", tolerance="6m")
        # power at 00:15 is >6 min from last measurement (00:05) → null
        assert result.filter(pl.col("ts") == _ts(2026,1,1,0,15))["power"][0] is None