from .utils import parse_duration

import polars as pl


def detect_gaps(df: pl.DataFrame, time_col: str, expected_freq: str, *, threshold: float = 1.5) -> pl.DataFrame:
    """
    Finds all gaps in a time-series.

    A gap is detected when the interval between two consecutive timestamps exceeds ``threshold * expected_freq``.

    Parameters
    ---
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    expected_freq:
        Expected sampling interval, e.g. '5m'.
    threshold:
        Multiplier over ``expected_freq`` that defines a gap.  Default: ``1.5``.

    Returns
    -------
    polars.DataFrame
        Columns: ``gap_start``, ``gap_end``, ``gap_seconds``, ``missing_periods``.
    """
    
    td = parse_duration(expected_freq)
    min_gap_s = td.total_seconds() * threshold
    sorted_df = df.sort(time_col)

    return (
        sorted_df.select(
            pl.col(time_col).alias("gap_start"),
            pl.col(time_col).shift(-1).alias("gap_end"),
            (pl.col(time_col).shift(-1) - pl.col(time_col))
            .dt.total_seconds()
            .alias("gap_seconds"),
        )
        .drop_nulls()
        .filter(pl.col("gap_seconds") > min_gap_s)
        .with_columns(
            (pl.col("gap_seconds") / td.total_seconds() - 1)
            .round(1)
            .alias("missing_periods")
        )
    )


def flag_gaps(df: pl.DataFrame, time_col: str, expected_freq: str, *, col_name: str = "gap_after", threshold: float = 1.5) -> pl.DataFrame:
    """
    Adds a boolean column that is ``True`` on rows immediately *before* a gap.
    Useful for masking or inspecting continuity breaks.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    expected_freq:
        Expected sampling interval, e.g. ``'5m'``.
    col_name:
        Name of the new boolean column. Default ``'gap_after'``.
    threshold:
        Gap detection sensitivity. See :func:`detect_gaps`.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``col_name`` appended.
    """

    td = parse_duration(expected_freq)
    min_gap_s = td.total_seconds() * threshold

    return df.sort(time_col).with_columns(
        (
            (pl.col(time_col).shift(-1) - pl.col(time_col))
            .dt.total_seconds()
            > min_gap_s
        )
        .fill_null(False)
        .alias(col_name)
    )


def coverage(df: pl.DataFrame, time_col: str, expected_freq: str) -> float:
    """
    Extracts data coverage: fraction of expected timestamps that are present.
    
    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    expected_freq:
        Expected sampling frequency, e.g. ``'5m'``.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.  ``1.0`` means no gaps.
    """

    td = parse_duration(expected_freq)
    time_min = df[time_col].min()
    time_max = df[time_col].max()
    span_s = (time_max - time_min).total_seconds()
    if span_s == 0:
        return 1.0

    expected_count = int(span_s / td.total_seconds()) + 1
    actual_count = df[time_col].n_unique()

    return min(actual_count / expected_count, 1.0)
