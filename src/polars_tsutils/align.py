from .utils import parse_duration

from datetime import datetime
import polars as pl


def make_grid(t_min: datetime, t_max: datetime, freq: str, time_col: str = "ts") -> pl.DataFrame:
    """
    Builds a datetime grid as a DataFrame.

    Parameters
    ----------
    t_min:
        First grid point.
    t_max:
        Last grid point.
    freq:
        Grid spacing: ``'1m'``, ``'5m'``, ``'1h'``, etc.
    time_col:
        Name of the datetime column in the output.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``time_col`` at regular ``freq`` intervals.

    Examples
    --------
    >>> from datetime import datetime
    >>> grid = make_grid(datetime(2024,1,1), datetime(2024,1,1,1), "15m")
    >>> grid.height
    5
    """

    return pl.DataFrame({
        time_col: pl.datetime_range(t_min, t_max, interval=freq, eager=True)
    })


def align_to_grid(
        df              : pl.DataFrame,
        time_col        : str,
        freq            : str,
        method          : str = "nearest",
        keep_original   : bool = False,
        on_duplicate    : str = "first") -> pl.DataFrame:
    """
    Moves uneven timestamps to the closest regular time point. 
    Useful for combining data from two sources that record at different times, 
    or before any calculation that expects timestamps to be evenly spaced.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column.
    freq:
        Grid frequency: ``'5m'``, ``'1h'``, etc.
    method:
        * ``'nearest'`` — round to the closest grid point (default).
        * ``'floor'``   — always snap backward (truncate).
        * ``'ceil'``    — always snap forward.
    keep_original:
        If ``True``, preserve the original timestamp in a column named
        ``{time_col}_original`` before snapping.
    on_duplicate:
        * ``'first'`` — keep the first row (default).
        * ``'last'``  — keep the last row.
        * ``'error'`` — raise ``ValueError`` if duplicates exist.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``time_col`` replaced by snapped values.

    Examples
    --------
    >>> snapped = align_to_grid(df, "ts", "5m")
    >>> snapped = align_to_grid(df, "ts", "5m", method="floor", on_duplicate="last")
    """

    if method not in ("nearest", "floor", "ceil"):
        raise ValueError(f"method must be 'nearest', 'floor', or 'ceil', got '{method}'")
    if on_duplicate not in ("first", "last", "error"):
        raise ValueError(f"on_duplicate must be 'first', 'last', or 'error', got '{on_duplicate}'")

    result = df.sort(time_col)

    if keep_original:
        result = result.with_columns(
            pl.col(time_col).alias(f"{time_col}_original")
        )

    if method == "nearest":
        result = result.with_columns(pl.col(time_col).dt.round(freq))
    elif method == "floor":
        result = result.with_columns(pl.col(time_col).dt.truncate(freq))
    else:  # ceil
        td = parse_duration(freq)
        result = result.with_columns(
            pl.when(pl.col(time_col) == pl.col(time_col).dt.truncate(freq))
            .then(pl.col(time_col))
            .otherwise(pl.col(time_col).dt.truncate(freq) + pl.duration(seconds=int(td.total_seconds())))
            .alias(time_col)
        )

    dupes = result.filter(result[time_col].is_duplicated())
    if dupes.height > 0:
        if on_duplicate == "error":
            raise ValueError(
                f"{dupes.height} rows snapped to duplicate grid points. "
                "Use on_duplicate='first' or 'last' to resolve, or increase freq."
            )
        keep = "first" if on_duplicate == "first" else "last"
        result = result.unique(subset=[time_col], keep=keep, maintain_order=True)

    return result.sort(time_col)


def project_to_grid(
        df          : pl.DataFrame,
        time_col    : str,
        freq        : str,
        value_cols  : list[str] | None = None,
        t_min       : datetime | None = None,
        t_max       : datetime | None = None,
        tolerance   : str | None = None) -> pl.DataFrame:
    """
    Place an irregular series onto a regular grid using **zero-order hold**.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_col:
        Datetime column name.
    freq:
        Grid frequency: ``'5m'``, ``'1h'``, etc.
    value_cols:
        Columns to project.  Defaults to all non-time columns.
    t_min:
        Grid start.  Defaults to ``df[time_col].min()``.
    t_max:
        Grid end.  Defaults to ``df[time_col].max()``.
    tolerance:
        Maximum look-back distance.  Grid points with no measurement within
        this window produce ``null`` instead of carrying a stale value forward.
        Format: ``'15m'``, ``'1h'``, etc.

    Returns
    -------
    pl.DataFrame
        Regular-grid DataFrame with ZOH-filled values.

    Examples
    --------
    >>> regular = project_to_grid(raw_df, "ts", "5m")
    >>> regular = project_to_grid(raw_df, "ts", "5m", tolerance="30m")
    """

    cols = list(value_cols) if value_cols is not None else [
        c for c in df.columns if c != time_col
    ]

    _t_min = t_min if t_min is not None else df[time_col].min()
    _t_max = t_max if t_max is not None else df[time_col].max()

    grid = make_grid(_t_min, _t_max, freq, time_col) # type: ignore

    join_kwargs: dict = {"on": time_col, "strategy": "backward"}
    if tolerance is not None:
        td = parse_duration(tolerance)
        join_kwargs["tolerance"] = f"{int(td.total_seconds())}s"

    return grid.join_asof(
        df.sort(time_col).select([time_col] + cols),
        **join_kwargs,
    )


def outer_join_resample(
        df_a      : pl.DataFrame,
        df_b      : pl.DataFrame,
        time_col  : str,
        freq      : str,
        suffixes  : tuple[str, str] = ("_a", "_b"),
        tolerance : str | None = None,
        t_max     : datetime | None = None,
        t_min     : datetime | None = None,
) -> pl.DataFrame:
    """
    Aligns two time-series onto a shared regular grid and joins them.
    Each series is projected onto the grid using **zero-order hold**.

    Parameters
    ----------
    df_a:
        First series.
    df_b:
        Second series.
    time_col:
        Datetime column name (must be the same in both DataFrames).
    freq:
        Output grid frequency: ``'5m'``, ``'1h'``, etc.
    suffixes:
        ``(suffix_a, suffix_b)`` appended to conflicting column names.
        Default ``('_a', '_b')``.
    tolerance:
        Maximum look-back distance for ZOH fill.  Grid points with no
        recent measurement in either series produce ``null``.
    t_min:
        Override grid start (defaults to ``min`` of both series).
    t_max:
        Override grid end (defaults to ``max`` of both series).

    Returns
    -------
    pl.DataFrame
        Regular-grid DataFrame with columns from both series, disambiguated
        with ``suffixes`` where names collide.

    Examples
    --------
    >>> # Inverter power (sampled every 3 min) vs irradiance (every 10 min)
    >>> combined = outer_join_resample(inv_df, irr_df, "ts", "5m")

    >>> # Restrict to overlapping window only
    >>> combined = outer_join_resample(
    ...     inv_df, irr_df, "ts", "5m",
    ...     t_min=datetime(2024,1,1,6), t_max=datetime(2024,1,1,18),
    ... )
    """

    _t_min = t_min if t_min is not None else min(df_a[time_col].min(), df_b[time_col].min()) # type: ignore
    _t_max = t_max if t_max is not None else max(df_a[time_col].max(), df_b[time_col].max()) # type: ignore

    grid = make_grid(_t_min, _t_max, freq, time_col)

    a_value_cols = [c for c in df_a.columns if c != time_col]
    b_value_cols = [c for c in df_b.columns if c != time_col]

    common = set(a_value_cols) & set(b_value_cols)
    df_a_r = df_a.rename({c: f"{c}{suffixes[0]}" for c in common})
    df_b_r = df_b.rename({c: f"{c}{suffixes[1]}" for c in common})

    join_kwargs: dict = {"strategy": "backward"}
    if tolerance is not None:
        td = parse_duration(tolerance)
        join_kwargs["tolerance"] = f"{int(td.total_seconds())}s"

    return (
        grid
        .join_asof(df_a_r.sort(time_col), on=time_col, **join_kwargs)
        .join_asof(df_b_r.sort(time_col), on=time_col, **join_kwargs)
    )