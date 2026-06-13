from datetime import datetime
from typing import Sequence
import polars as pl


def fill_zoh(df: pl.DataFrame, value_cols: Sequence[str], *, limit: int | None = None) -> pl.DataFrame:
    """
    Fills null values using zero-order hold (forward-fill).

    Parameters
    ----------
    df:
        Input DataFrame.
    value_cols:
        Columns to fill.
    limit:
        Maximum number of consecutive nulls to fill.
        ``limit = None`` (default) fills all consecutive nulls.

    Returns
    -------
    pl.DataFrame
        DataFrame with nulls replaced by ZOH values.

    Examples
    --------
    >>> filled = fill_zoh(df, ["power", "voltage"])
    """

    return df.with_columns([
        pl.col(c).forward_fill(limit=limit) for c in value_cols
    ])


def seed_at_boundary(df: pl.DataFrame, time_col: str, value_cols: Sequence[str], boundary: datetime) -> pl.DataFrame:
    """
    Inserts a row at `boundary` with the most recent values from before that time.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    time_col : str
        Name of the datetime column.
    value_cols : list of str
        Names of the columns whose values should be carried forward.
    boundary : datetime
        The timestamp at which to add the row.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the extra row added (if needed), sorted by time.

    Example
    -------
    >>> seed_at_boundary(df, "timestamp", ["temperature", "pressure"], datetime(2024, 1, 1, 8, 0))
    """

    df = df.sort(time_col)

    if df.filter(pl.col(time_col) == pl.lit(boundary)).height > 0:
        return df

    before = df.filter(pl.col(time_col) < pl.lit(boundary))
    if before.is_empty():
        return df

    seed_row = (
        before.tail(1)
        .select([time_col] + list(value_cols))
        .with_columns(pl.lit(boundary).cast(df[time_col].dtype).alias(time_col))
    )

    after = df.filter(pl.col(time_col) > pl.lit(boundary))

    return pl.concat([before, seed_row, after])
