import re
from datetime import timedelta

FREQ_RE = re.compile(r"^(\d+)(s|m|h|d)$", re.IGNORECASE)
UNIT_MAP = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_duration(freq: str) -> timedelta:
    """
    Parses a compact duration string into a ``datetime.timedelta``.

    Supported freqs: ``s`` (seconds), ``m`` (minutes), ``h`` (hours), ``d`` (days).

    Examples
    --------
    >>> parse_duration("5m")
    datetime.timedelta(seconds=300)
    >>> parse_duration("1h")
    datetime.timedelta(seconds=3600)
    """
    
    m = FREQ_RE.match(freq.strip())
    if not m:
        raise ValueError(
            f"Cannot parse frequency '{freq}'. "
            "Expected format: integer followed by s/m/h/d — e.g. '5m', '1h', '30s', '1d'."
        )
    value, unit = int(m.group(1)), m.group(2).lower()
    return timedelta(**{UNIT_MAP[unit]: value})