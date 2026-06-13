from .utils import parse_duration

from typing import Sequence
import polars as pl


def resample_twa(df: pl.DataFrame, time_col: str, freq: str, value_cols: Sequence[str], *, label: str = "left") -> pl.DataFrame:
    """
    Resamples an irregular time-series to fixed-frequency using **time-weighted average** (TWA).

    Parameters
    ----------
    df:
        Input DataFrame. Must contain ``time_col`` and ``value_cols``.
    time_col:
        Name of the datetime column.
    freq:
        Bucket width as a compact duration string: ``'5m'``, ``'1h'``, ``'30s'``, etc.
    value_cols:
        Columns to aggregate. Must be numeric.
    label:
        Which side of the bucket to use as the output timestamp.
        ``'left'`` (default) uses the bucket start; ``'right'`` uses bucket end.

    Returns
    -------
    pl.DataFrame
        One row per bucket, with ``time_col`` and TWA for each value column.
        Buckets with no overlapping data have ``null`` values.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> df = pl.DataFrame({
    ...     "ts": [datetime(2024,1,1,0,0), datetime(2024,1,1,0,3), datetime(2024,1,1,0,7)],
    ...     "ac_power": [100.0, 150.0, 200.0],
    ... })
    >>> resample_twa(df, "ts", "5m", ["ac_power"])
    """

    if label.lower() not in ("left", "right"):
        raise ValueError("label must be 'left' or 'right'")

    td = parse_duration(freq)
    df = df.sort(time_col)

    time_min = df[time_col].min()
    time_max = df[time_col].max()

    reference = time_max + td

    df = df.with_columns(
        pl.col(time_col).shift(-1).fill_null(pl.lit(reference)).alias("_next_ts")
    )

    td_s = td.total_seconds()

    buckets: list = []
    t = time_min
    while t <= time_max:
        buckets.append(t)
        t += td

    records = []
    for b_start in buckets:
        b_end = b_start + td

        overlap = df.filter(
            (pl.col(time_col) < pl.lit(b_end))
            & (pl.col("_next_ts") > pl.lit(b_start))
        )

        bucket_ts = b_end if label.lower() == "right" else b_start

        if overlap.is_empty():
            records.append({time_col: bucket_ts, **{c: None for c in value_cols}})
            continue

        weights = overlap.select(
            (
                pl.min_horizontal(pl.col("_next_ts"), pl.lit(b_end))
                - pl.max_horizontal(pl.col(time_col), pl.lit(b_start))
            )
            .dt.total_seconds()
            .alias("_w")
        )["_w"]

        row: dict = {time_col: bucket_ts}
        for col in value_cols:
            vals = overlap[col].cast(pl.Float64)
            twa = float((vals * weights).sum() / td_s)
            row[col] = twa

        records.append(row)

    result = pl.DataFrame(records)
    result = result.with_columns(pl.col(time_col).cast(df[time_col].dtype))
    return result


def upsample_zoh(df: pl.DataFrame, time_col: str, freq: str, value_cols: Sequence[str] | None = None) -> pl.DataFrame:
    """
    Upsample a time-series to a regular grid using **zero-order hold** (forward-fill).
    Missing timestamps are inserted and filled with the last known value.

    Parameters
    ----------
    df:
        Input DataFrame with a datetime ``time_col``.
    time_col:
        Name of the datetime column.
    freq:
        Target grid frequency: ``'1m'``, ``'5m'``, ``'1h'``, etc.
    value_cols:
        Columns to forward-fill. Defaults to all non-time columns.

    Returns
    -------
    pl.DataFrame
        Regular-frequency DataFrame with ZOH-filled values.

    Examples
    --------
    >>> upsample_zoh(df, "ts", "1m", ["power", "voltage"])
    """

    cols = list(value_cols) if value_cols is not None else [
        c for c in df.columns if c != time_col
    ]

    upsampled = (
        df.sort(time_col)
        .upsample(time_column=time_col, every=freq)
    )

    return upsampled.with_columns([
        pl.col(c).forward_fill() for c in cols
    ])
