# polars-tsutils

Time-series utilities for [Polars](https://pola.rs): time-weighted average resampling, zero-order hold interpolation, gap detection, rolling statistics, signal quality checks, ...

Built for SCADA, metering, and industrial sensor data where a measurement *holds its value* until the next one arrives (step-function / ZOH semantics), making naive `mean()` resampling incorrect.

---

## Time-Weighted Average (TWA):

Consider a 5-minute bucket `[00:00, 00:05)` containing two samples:

| Timestamp | DC Power |
|-----------|----------|
| 00:00     |  100 W   |
| 00:03     |  200 W   |

**Naive mean:** `(100 + 200) / 2 = 150 W`

**Time-weighted average:** `(100×180s + 200×120s) / 300s = 140 W`

---

## Duration string format

All `freq` / `window` parameters accept compact duration strings:

| String | Duration      |
|--------|---------------|
| `"30s"` | 30 seconds   |
| `"5m"`  | 5 minutes    |
| `"1h"`  | 1 hour       |
| `"1d"`  | 1 day        |

---

## Development

```bash
git clone https://github.com/a-m-farahani/polars-tsutils
cd polars-tsutils
pip install -e ".[dev]"
pytest
```
