"""Facility-type policy shared by allocation engines."""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_ELIGIBLE_FACILITY_TYPES = ("shelter", "hospital")


def normalize_facility_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "evacuation_shelter": "shelter",
        "emergency_shelter": "shelter",
        "clinic": "hospital",
        "medical_center": "hospital",
        "fire": "fire_station",
        "firestation": "fire_station",
    }
    return aliases.get(text, text)


def normalize_facility_types(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if values is None:
        values = DEFAULT_ELIGIBLE_FACILITY_TYPES
    if isinstance(values, str):
        values = values.split(",")
    normalized = tuple(
        dict.fromkeys(
            kind for kind in (normalize_facility_type(value) for value in values) if kind
        )
    )
    if not normalized:
        raise ValueError("eligible_facility_types must contain at least one type")
    return normalized
