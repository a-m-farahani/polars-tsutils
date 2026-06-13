# polars-tsutils

Time-series utilities for [Polars](https://pola.rs).   
Time-weighted average resampling, zero-order hold interpolation, gap detection, rolling statistics, signal quality checks, ...

Built for SCADA, metering, and industrial sensor data where **a measurement holds its value until the next one arrives** (step-function / ZOH semantics), making naive `mean()` resampling incorrect.


## Why regular average is wrong?

Take a 5‑min bucket with two values:

| Timestamp | Power |
|-----------|-------|
| 00:00     | 100 W |
| 00:04     | 200 W |

- Naive mean: `(100+200)/2 = 150 W`  
- Time‑weighted mean: `(100*240s + 200*60s)/300s = 120 W`


## Duration strings

All `freq` and `window` parameters accept compact duration strings:

| String  |  Duration  |
|---------|------------|
| `"30s"` | 30 seconds |
| `"5m"`  | 5 minutes  |
| `"1h"`  | 1 hour     |
| `"1d"`  | 1 day      |


## Rolling Operations

Two rolling functions with ZOH behaviour:

- `rolling_twa`: Computes a time-weighted average over a look-back window ending at each row. 
A sudden change (e.g., 100 → 200) causes the rolling TWA to transition **gradually** 
over the window duration, not jump instantly.  
- `rolling_zscore`: Computes a rolling Z‑score = (value - rolling_mean) / rolling_std using the same logic.

Both functions use a backward-looking window `[ts - window, ts]` and append the result column to the original DataFrame.


## Gap detection

Functions to find and flag irregular gaps in regularly sampled time series.

- `detect_gaps`: Returns a DataFrame listing all gaps longer than `threshold` $\times$ `expected_freq`. Columns: `gap_start`, `gap_end`, `gap_seconds`, `missing_periods` (number of expected samples missing).
- `flag_gaps`: Adds a boolean column (`True` on the row immediately **before** a gap), useful for masking or visualisation.
- `coverage`: Returns the fraction of expected timestamps present in the data (`1.0` means no gaps).



## API Reference

### `resample`

#### `resample_twa(df, time_col, freq, value_cols, *, label='left')`

Resample to fixed-frequency buckets using **time-weighted average**.

```python
df_5m = ptu.resample_twa(df, "timestamp", "5m", ["power", "current"])
```

#### `upsample_zoh(df, time_col, freq, value_cols=None)`

Upsample to a regular grid by forward-filling the last known value.

```python
df_1m = ptu.upsample_zoh(df, "timestamp", "1m", ["power"])
```


### `gaps`

#### `detect_gaps(df, time_col, expected_freq, *, threshold=1.5)`

Return a DataFrame of gaps: `gap_start`, `gap_end`, `gap_seconds`, `missing_periods`.

```python
gaps = ptu.detect_gaps(df, "timestamp", "5m")
# shape: (n_gaps, 4)
```

#### `flag_gaps(df, time_col, expected_freq, *, col_name='_gap_after', threshold=1.5)`

Add a boolean column that is `True` on rows immediately before a gap.

#### `coverage(df, time_col, expected_freq) → float`

Fraction of expected timestamps present.  `1.0` = no gaps.


### `rolling`

#### `rolling_twa(df, time_col, value_col, window, *, output_col=None, min_weight=0.0)`

Rolling time-weighted average over a time-based window.  Preserves ZOH semantics: the TWA properly lags step changes rather than snapping instantly.

```python
df = ptu.rolling_twa(df, "timestamp", "power", "15m")
# adds column: twa_power
```

#### `rolling_zscore(df, time_col, value_col, window, *, output_col=None, min_periods=2)`

Rolling Z-score anomaly score over a time-based window.

```python
df = ptu.rolling_zscore(df, "timestamp", "power", "1h")
# adds column: zscore_power
```



## Development

```bash
git clone https://github.com/a-m-farahani/polars-tsutils
cd polars-tsutils
pip install -e ".[dev]"
pytest
```