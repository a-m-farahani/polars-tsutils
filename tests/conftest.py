from datetime import datetime, timedelta
import pytest
import polars as pl


def make_ts(start: str, n: int, freq_minutes: int = 5) -> list[datetime]:
    base = datetime.fromisoformat(start)
    return [base + timedelta(minutes=i * freq_minutes) for i in range(n)]


@pytest.fixture
def regular_df():
    """Regular 5-minute series, no gaps."""
    ts = make_ts("2026-01-01 00:00", 12)  # 1 hour
    return pl.DataFrame({
        "ts": ts,
        "power": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                  160.0, 170.0, 180.0, 190.0, 200.0, 210.0],
        "voltage": [230.0] * 12,
    })


@pytest.fixture
def irregular_df():
    """Irregular timestamps (not on 5-minute boundary)."""
    ts = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 3),   # 3 min
        datetime(2026, 1, 1, 0, 7),   # 4 min
        datetime(2026, 1, 1, 0, 12),  # 5 min
        datetime(2026, 1, 1, 0, 17),  # 5 min
    ]
    return pl.DataFrame({
        "ts": ts,
        "power": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def df_with_gap():
    """5-minute series with a 15-minute gap in the middle."""
    ts = (
        make_ts("2026-01-01 00:00", 6)       # 00:00–00:25
        + make_ts("2026-01-01 00:40", 4)     # skip 00:30–00:35, resume 00:40
    )
    return pl.DataFrame({
        "ts": ts,
        "power": list(range(10, 110, 10)),
    })