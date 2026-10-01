"""Operational threshold-dashboard baseline using authoritative gauge bands.

This is intentionally stronger than the fault-injection ``NaiveBaseline``:
it never substitutes a missing stage or threshold with zero and it implements
the core function of a public flood dashboard—show the observed gauge stage
against the gauge's published flood categories. It does not attempt spatial
exposure, equity, routing, or allocation.
"""

from __future__ import annotations

from typing import Any

from .historical_validation import flood_category
from .naive_baseline import _call_json
from .utils import safe_float


class ThresholdDashboardBaseline:
    name = "authoritative_threshold_dashboard"

    async def run(self, mcp: Any, station_id: str) -> dict[str, Any]:
        observation = await _call_json(
            mcp, "get_flood_observation", {"station_id": station_id}
        )
        gauge = await _call_json(
            mcp, "get_nwps_gauge", {"station_id": station_id}
        )
        stage = safe_float((observation.get("observation") or {}).get("water_level"))
        raw_categories = gauge.get("flood_categories") or {}
        categories = {
            name: safe_float((raw_categories.get(name) or {}).get("stage"))
            for name in ("action", "minor", "moderate", "major")
        }
        categories = {name: value for name, value in categories.items() if value is not None}
        gaps = []
        if stage is None:
            gaps.append("observed_stage")
        if "action" not in categories:
            gaps.append("authoritative_flood_categories")
        if gaps:
            return {
                "status": "unavailable",
                "baseline": self.name,
                "data_gaps": gaps,
                "hydrologic_category": None,
            }
        return {
            "status": "ok",
            "baseline": self.name,
            "station_id": station_id,
            "observed_stage_ft": stage,
            "hydrologic_category": flood_category(stage, categories),
            "thresholds_ft": categories,
            "source": gauge.get("source") or "NWPS gauge metadata",
            "capabilities": [
                "gauge monitoring",
                "authoritative threshold classification",
            ],
            "out_of_scope": [
                "flood extent",
                "population exposure",
                "equity",
                "capacity-aware transfer planning",
                "resource allocation",
            ],
        }
