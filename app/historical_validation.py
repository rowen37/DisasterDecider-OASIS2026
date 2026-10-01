"""Independent historical-event validation with explicit scope limits."""

from __future__ import annotations

import json
import time
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def flood_category(stage_ft: float, categories: dict[str, float]) -> str:
    ordered = ("major", "moderate", "minor", "action")
    for name in ordered:
        threshold = categories.get(name)
        if threshold is not None and stage_ft >= float(threshold):
            return name
    return "below_action"


SUPPORTED_MANIFEST_SCHEMA_VERSION = 2
HYDROLOGIC_STRATA = (
    "below_action",
    "action",
    "minor",
    "moderate",
    "major",
)
_REQUIRED_EVENT_FIELDS = (
    "event_id",
    "validation_tier",
    "severity_stratum",
    "observed_peak_stage_ft",
    "flood_categories_ft",
    "expected_hydrologic_category",
)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    schema_version = manifest.get("schema_version")
    if schema_version != SUPPORTED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported validation manifest schema_version "
            f"{schema_version!r}; expected "
            f"{SUPPORTED_MANIFEST_SCHEMA_VERSION}"
        )
    events = manifest.get("events") or []
    development_ids = {
        item.get("event_id") for item in events if item.get("role") == "development"
    }
    validation_events = [
        item for item in events if item.get("role") == "held_out_validation"
    ]
    if not validation_events:
        raise ValueError("manifest has no held-out validation event")

    results = []
    stratum_counts = {name: 0 for name in HYDROLOGIC_STRATA}
    full_agent_event_ids = []
    for event in validation_events:
        event_id = event.get("event_id")
        missing = [
            field for field in _REQUIRED_EVENT_FIELDS if event.get(field) is None
        ]
        if missing:
            raise ValueError(
                f"validation event {event_id!r} is missing required "
                f"fields: {', '.join(missing)}"
            )
        if event_id in development_ids:
            raise ValueError(f"event {event_id} appears in development and validation")
        stratum = event["severity_stratum"]
        if stratum not in HYDROLOGIC_STRATA:
            raise ValueError(
                f"validation event {event_id!r} has unsupported "
                f"severity_stratum {stratum!r}"
            )
        if event["validation_tier"] not in {"hydrologic", "full_agent"}:
            raise ValueError(
                f"validation event {event_id!r} has unsupported "
                f"validation_tier {event['validation_tier']!r}"
            )
        categories = event["flood_categories_ft"]
        category_values = [float(categories[name]) for name in HYDROLOGIC_STRATA[1:]]
        if category_values != sorted(category_values) or len(set(category_values)) != 4:
            raise ValueError(
                f"validation event {event_id!r} flood categories must be "
                "strictly increasing: action < minor < moderate < major"
            )
        observed = float(event["observed_peak_stage_ft"])
        predicted = flood_category(observed, categories)
        expected = event["expected_hydrologic_category"]
        if expected != stratum:
            raise ValueError(
                f"validation event {event_id!r} severity_stratum must match "
                "expected_hydrologic_category"
            )
        stratum_counts[stratum] += 1
        if event["validation_tier"] == "full_agent":
            full_agent_event_ids.append(event_id)
        results.append(
            {
                "event_id": event_id,
                "independent_of_development_set": True,
                "validation_tier": event["validation_tier"],
                "severity_stratum": stratum,
                "observed_peak_stage_ft": observed,
                "predicted_hydrologic_category": predicted,
                "expected_hydrologic_category": expected,
                "passed": predicted == expected,
                "validated_scope": event.get("validated_scope") or [],
                "not_validated": event.get("not_validated") or [],
                "source": event.get("observation_source"),
            }
        )
    return {
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "schema_version": schema_version,
        "development_event_ids": sorted(development_ids),
        "held_out_event_count": len(results),
        "severity_stratum_counts": stratum_counts,
        "balanced_severity_coverage": all(stratum_counts.values()),
        "full_agent_event_ids": full_agent_event_ids,
        "results": results,
    }


def validate_full_agent_output(
    event: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """Check end-to-end artifacts without claiming that they are accurate.

    This validation proves that the real Agent chain completed and produced
    the required, auditable artifacts.  It deliberately does not treat a
    successful run as ground-truth validation of SAR extent, CDRI bands, or
    travel-time estimates.
    """
    if event.get("validation_tier") != "full_agent":
        raise ValueError(
            f"event {event.get('event_id')!r} is not a full_agent event"
        )

    structured = output.get("structured") or {}
    optimization = output.get("optimization") or {}
    requirements = event.get("full_agent_acceptance") or {}
    tolerance = float(requirements.get("stage_tolerance_ft", 0.02))
    checks = [
        {
            "name": "run_completed",
            "passed": output.get("status") == "completed",
            "observed": output.get("status"),
        },
        {
            "name": "station_matches",
            "passed": str(structured.get("station_id")) == str(event["station_id"]),
            "observed": structured.get("station_id"),
        },
        {
            "name": "historical_mode",
            "passed": structured.get("assessment_mode") == "historical",
            "observed": structured.get("assessment_mode"),
        },
        {
            "name": "peak_stage_matches",
            "passed": (
                structured.get("water_level") is not None
                and abs(
                    float(structured["water_level"])
                    - float(event["observed_peak_stage_ft"])
                )
                <= tolerance
            ),
            "observed": structured.get("water_level"),
        },
        {
            "name": "hydrologic_category_matches",
            "passed": (
                structured.get("hydrologic_category")
                == event["expected_hydrologic_category"]
            ),
            "observed": structured.get("hydrologic_category"),
        },
    ]

    allowed_extent = requirements.get("allowed_extent_provenance") or []
    if allowed_extent:
        checks.append(
            {
                "name": "operational_extent_produced",
                "passed": structured.get("extent_provenance") in allowed_extent,
                "observed": structured.get("extent_provenance"),
            }
        )
    if requirements.get("require_population_exposure", False):
        checks.append(
            {
                "name": "population_exposure_produced",
                "passed": (
                    structured.get("population_affected") is not None
                    and int(structured["population_affected"]) > 0
                ),
                "observed": structured.get("population_affected"),
            }
        )
    if requirements.get("require_resource_optimization", False):
        checks.extend(
            [
                {
                    "name": "resource_optimization_completed",
                    "passed": optimization.get("status") == "optimized",
                    "observed": optimization.get("status"),
                },
                {
                    "name": "pareto_recommendation_produced",
                    "passed": bool(optimization.get("recommended_plan_id")),
                    "observed": optimization.get("recommended_plan_id"),
                },
                {
                    "name": "choice_scenarios_produced",
                    "passed": len(optimization.get("plan_scenarios") or []) >= 2,
                    "observed": len(optimization.get("plan_scenarios") or []),
                },
            ]
        )

    passed = all(check["passed"] for check in checks)
    return {
        "event_id": event["event_id"],
        "status": "passed" if passed else "failed",
        "checks": checks,
        "claim_limit": (
            "Execution/artifact acceptance only; does not validate extent "
            "accuracy, CDRI bands, or travel-time accuracy."
        ),
    }


def refresh_usgs_peak(event: dict[str, Any]) -> dict[str, Any]:
    """Fetch the official series and return its peak without editing files."""
    payload = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(  # noqa: S310 -- manifest contains reviewed USGS URLs
                event["observation_source"],
                timeout=60,
            ) as response:
                payload = json.loads(response.read())
            break
        except (HTTPException, OSError, URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (2**attempt))
    if payload is None:
        raise RuntimeError(
            f"USGS refresh failed after 3 attempts: {last_error}"
        ) from last_error
    values = []
    for series in payload.get("value", {}).get("timeSeries", []):
        for group in series.get("values", []):
            for item in group.get("value", []):
                try:
                    values.append((float(item["value"]), item.get("dateTime")))
                except (KeyError, TypeError, ValueError):
                    continue
    if not values:
        raise ValueError("USGS response contained no numeric values")
    peak = max(item[0] for item in values)
    timestamps = [timestamp for value, timestamp in values if value == peak]
    return {
        "observed_peak_stage_ft": peak,
        "observed_peak_time_range": [timestamps[0], timestamps[-1]],
    }


def load_default_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "validation_events.json"
    return json.loads(path.read_text(encoding="utf-8"))
