"""Project-wide shared pure utility functions (no I/O, no state).

Single implementation site for safe_float / utc_now.
"""

from __future__ import annotations

import math

from datetime import datetime, timezone
from typing import Any


def safe_float(
    value: Any,
) -> float | None:
    """Convert any value to a finite float; return None when not convertible or non-finite."""

    try:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def utc_now() -> datetime:
    """Current UTC time (tzinfo-aware)."""

    return datetime.now(timezone.utc)


def geodesic_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) on the IUGG mean Earth radius
    (6371.0088 km). Project-wide single implementation.
    """
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))
