from .utils import parse_duration

import polars as pl


def rolling_twa(df: pl.DataFrame, time_col: str, value_col: str, window: str, *, output_col: str | None = None, min_weight: float = 0.0) -> pl.DataFrame:
    """
    Computes a **rolling TWA** over a time-based window.

    For each row ``i``, the TWA is computed over the interval
    ``[ts[i] - window, ts[i]]`` using ZOH )each measurement holds its value until the next one).

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    value_col:
        Value column, aggregation is performed on this values.
    window:
        Look-back window duration: ``'15m'``, ``'1h'``, etc.
    output_col:
        Name of the result column.  Defaults to ``'twa_{value_col}'``.
    min_weight:
        Minimum coverage (seconds) required to remove a value instead of
        ``null``.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``output_col`` appended.
    """

    td = parse_duration(window)
    out_col = output_col or f"twa_{value_col}"

    df = df.sort(time_col)
    times = df[time_col].to_list()
    values = df[value_col].cast(pl.Float64).to_list()
    n = len(times)

    results: list[float | None] = []
    left = 0 

    for i in range(n):
        t_end = times[i]
        t_start = t_end - td

        while left < i and (
            (times[left + 1] if left + 1 <= i else t_end) <= t_start
        ):
            left += 1

        weighted_sum = 0.0
        total_weight = 0.0

        for j in range(left, i + 1):
            if values[j] is None:
                continue

            # Active interval for row j: [times[j], times[j+1]) clipped to [t_start, t_end]
            t_next = times[j + 1] if j + 1 < n else t_end
            eff_start = max(times[j], t_start)
            eff_end = min(t_next, t_end)

            if eff_end <= eff_start:
                continue

            weight = (eff_end - eff_start).total_seconds()
            weighted_sum += values[j] * weight
            total_weight += weight

        if total_weight > min_weight:
            results.append(weighted_sum / total_weight)
        else:
            results.append(None)

    return df.with_columns(pl.Series(out_col, results, dtype=pl.Float64))


def rolling_zscore(df: pl.DataFrame, time_col: str, value_col: str, window: str, *, output_col: str | None = None, min_periods: int = 2) -> pl.DataFrame:
    """
    Computes a **rolling Z-score** over a time-based window.

    For each row, Z = (value - rolling_mean) / rolling_std, where the rolling
    statistics are computed over all rows within the ``window`` duration.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    value_col:
        Numeric column to score.
    window:
        Look-back window duration: ``'1h'``, ``'24h'``, etc.
    output_col:
        Name of the result column.  Defaults to ``'zscore_{value_col}'``.
    min_periods:
        Minimum number of non-null rows required to compute a score.
        Rows with fewer observations emit ``null``.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with ``output_col`` appended.
    """

    out_col = output_col or f"zscore_{value_col}"

    result = (
        df.sort(time_col)
        .with_columns([
            pl.col(value_col)
            .rolling_mean_by(time_col, window_size=window)
            .alias("_rm"),
            pl.col(value_col)
            .rolling_std_by(time_col, window_size=window)
            .alias("_rs"),
            pl.col(value_col)
            .rolling_sum_by(time_col, window_size=window) 
            .alias("_dummy"),
        ])
    )

    result = result.with_columns([
        pl.col(value_col)
        .is_not_null()
        .cast(pl.Int32)
        .rolling_sum_by(time_col, window_size=window)
        .alias("_cnt")
    ])

    result = result.with_columns(
        pl.when(pl.col("_cnt") < min_periods)
        .then(None)
        .when(pl.col("_rs").is_null() | (pl.col("_rs") == 0.0))
        .then(0.0)
        .otherwise((pl.col(value_col) - pl.col("_rm")) / pl.col("_rs"))
        .alias(out_col)
    ).drop(["_rm", "_rs", "_dummy", "_cnt"])

    return result
