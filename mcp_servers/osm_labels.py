"""Human-readable OSM labels with an optional authoritative local override."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_OVERRIDE_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "osm_label_overrides.json"
)


def load_label_overrides(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    configured = path or os.getenv("OSM_LABEL_OVERRIDES_PATH") or DEFAULT_OVERRIDE_PATH
    try:
        payload = json.loads(Path(configured).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if payload.get("schema_version") != 1:
        return {}
    buildings = payload.get("buildings")
    return buildings if isinstance(buildings, dict) else {}


def _address(tags: dict[str, Any]) -> str | None:
    full = str(tags.get("addr:full") or "").strip()
    if full:
        return full
    number = str(tags.get("addr:housenumber") or "").strip()
    street = str(tags.get("addr:street") or "").strip()
    if number and street:
        return f"{number} {street}"
    return street or None


def osm_display_label(
    tags: dict[str, Any],
    *,
    feature_type: str,
    osm_type: str,
    osm_id: Any,
    lat: float,
    lon: float,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return label, address, and provenance without showing a raw OSM id."""
    key = f"{osm_type}/{osm_id}"
    override = (overrides or {}).get(key) or {}
    override_name = str(override.get("name") or "").strip()
    override_address = str(override.get("address") or "").strip()
    if override_name or override_address:
        return {
            "label": override_name or override_address,
            "address": override_address or _address(tags),
            "label_source": "verified_local_override",
        }

    for field in ("name", "official_name", "short_name", "brand", "operator", "addr:housename"):
        value = str(tags.get(field) or "").strip()
        if value:
            return {
                "label": value,
                "address": _address(tags),
                "label_source": f"osm:{field}",
            }

    address = _address(tags)
    if address:
        return {
            "label": address,
            "address": address,
            "label_source": "osm:address",
        }
    readable_type = feature_type.replace("_", " ").title()
    return {
        "label": f"{readable_type} near {lat:.5f}, {lon:.5f}",
        "address": None,
        "label_source": "coordinate_fallback",
    }
