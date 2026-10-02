"""Community-requirement matching for resource-allocation plans.

This module deliberately separates two different facts:

* facility capabilities are sourced, relatively static attributes; and
* event requirements are editable operator inputs for the current run.

Unknown capability data stays unknown.  It is never converted to ``False``
and the absence of a live community feed is never interpreted as no need.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CAPABILITIES_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "community_facilities.json"
)

CAPABILITY_LABELS = {
    "emergency_rescue": "Emergency rescue capability",
    "high_water_rescue": "High-water rescue",
    "pickup_service": "Pickup service",
    "accessible_transport": "Accessible transport",
    "temporary_shelter": "Temporary shelter",
    "overnight_shelter": "Overnight accommodation",
    "wheelchair_accessible": "Wheelchair accessible",
    "family_support": "Children and family support",
    "elder_support": "Older-adult support",
    "medical_support": "Medical support",
    "pet_accommodation": "Pet accommodation",
    "backup_power": "Backup power",
    "food_water": "Food and drinking water",
    "multilingual_support": "Multilingual support",
}

REQUIREMENT_DEFINITIONS = {
    "emergency_rescue": {
        "label": "Emergency rescue and evacuation",
        "capabilities": ("emergency_rescue", "high_water_rescue"),
    },
    "safe_transport": {
        "label": "Safe transport and pickup",
        "capabilities": ("pickup_service", "accessible_transport"),
    },
    "temporary_shelter": {
        "label": "Temporary shelter and safe stay",
        "capabilities": ("temporary_shelter", "overnight_shelter"),
    },
    "special_population_support": {
        "label": "Support for children, older adults, or disabled people",
        "capabilities": (
            "wheelchair_accessible",
            "family_support",
            "elder_support",
        ),
    },
}


def normalize_community_requirements(values: Any) -> list[str]:
    """Return unique supported event-requirement identifiers."""

    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if value in REQUIREMENT_DEFINITIONS and value not in result:
            result.append(value)
    return result


def _tristate(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "verified", "available"}:
            return True
        if normalized in {"no", "false", "unavailable"}:
            return False
    return None


def load_facility_capability_overrides(
    path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load sourced facility overrides keyed by id or exact facility name."""

    resolved = Path(
        path
        or os.getenv("COMMUNITY_FACILITIES_PATH", "")
        or DEFAULT_CAPABILITIES_PATH
    )
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("community facility configuration must use schema_version 1")
    entries = payload.get("facilities")
    if not isinstance(entries, list):
        raise ValueError("community facility configuration requires a facilities list")

    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        facility_id = str(entry.get("facility_id") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not facility_id and not name:
            raise ValueError("each community facility entry needs facility_id or name")
        source = str(entry.get("source") or "").strip()
        if not source:
            raise ValueError("each community facility entry needs a source")
        capabilities = entry.get("capabilities")
        if not isinstance(capabilities, dict):
            raise ValueError("community facility capabilities must be an object")
        normalized = {
            key: _tristate(value)
            for key, value in capabilities.items()
            if key in CAPABILITY_LABELS
        }
        record = {
            "capabilities": normalized,
            "source": source,
            "source_url": entry.get("source_url"),
            "last_verified_at": entry.get("last_verified_at"),
            "verification_status": entry.get("verification_status", "provided"),
        }
        if facility_id:
            result[f"id:{facility_id}"] = record
        if name:
            result[f"name:{name.casefold()}"] = record
    return result


def _baseline_capabilities(allocation: dict[str, Any]) -> tuple[dict[str, bool | None], list[dict[str, Any]]]:
    """Derive only capabilities that follow directly from sourced facility type/tags."""

    capabilities = {key: None for key in CAPABILITY_LABELS}
    evidence: list[dict[str, Any]] = []
    facility_type = str(allocation.get("type") or "").strip().lower()
    tags = allocation.get("tags") if isinstance(allocation.get("tags"), dict) else {}

    if facility_type == "hospital":
        capabilities["medical_support"] = True
        evidence.append({"capability": "medical_support", "source": "OSM amenity=hospital"})
    elif facility_type == "fire_station":
        capabilities["emergency_rescue"] = True
        evidence.append({"capability": "emergency_rescue", "source": "OSM amenity=fire_station"})
    elif facility_type == "shelter" and tags.get("amenity") == "shelter":
        # An OSM assembly point is not necessarily habitable shelter.  Only
        # the explicit amenity=shelter tag supports this baseline claim.
        capabilities["temporary_shelter"] = True
        evidence.append({
            "capability": "temporary_shelter",
            "source": "OSM amenity=shelter",
        })

    wheelchair = str(tags.get("wheelchair") or "").strip().lower()
    if wheelchair in {"yes", "no"}:
        capabilities["wheelchair_accessible"] = wheelchair == "yes"
        evidence.append({
            "capability": "wheelchair_accessible",
            "source": f"OSM wheelchair={wheelchair}",
        })
    return capabilities, evidence


def enrich_resource_plans(
    resources: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Attach sourced, tri-state capabilities to every allocation in-place."""

    configured = overrides if overrides is not None else load_facility_capability_overrides()
    for plan in resources:
        if not isinstance(plan, dict):
            continue
        for allocation in plan.get("allocations") or []:
            if not isinstance(allocation, dict):
                continue
            capabilities, evidence = _baseline_capabilities(allocation)
            existing = allocation.get("community_capabilities")
            if isinstance(existing, dict):
                for key, value in existing.items():
                    if key in capabilities:
                        capabilities[key] = _tristate(value)
                existing_evidence = allocation.get(
                    "community_capability_evidence"
                )
                if isinstance(existing_evidence, list):
                    evidence.extend(
                        item for item in existing_evidence
                        if isinstance(item, dict)
                    )
                else:
                    evidence.append({
                        "source": "resource record",
                        "capabilities": sorted(existing),
                    })

            facility_id = str(allocation.get("facility_id") or "").strip()
            name = str(allocation.get("resource") or "").strip()
            override = None
            if facility_id:
                override = configured.get(f"id:{facility_id}")
            if override is None and name:
                override = configured.get(f"name:{name.casefold()}")
            if override:
                capabilities.update(override["capabilities"])
                evidence.append({
                    "source": override["source"],
                    "source_url": override.get("source_url"),
                    "last_verified_at": override.get("last_verified_at"),
                    "verification_status": override.get("verification_status"),
                })
            allocation["community_capabilities"] = capabilities
            allocation["community_capability_evidence"] = evidence
    return resources


def _capability_status(
    allocations: list[dict[str, Any]], capability: str
) -> str:
    values = [
        (item.get("community_capabilities") or {}).get(capability)
        for item in allocations
        if isinstance(item, dict)
    ]
    if any(value is True for value in values):
        return "supported"
    known = [value for value in values if value is not None]
    if known and all(value is False for value in known) and len(known) == len(values):
        return "unsupported"
    return "unknown"


def annotate_plan_community(
    plans: list[dict[str, Any]],
    requirements: Any,
    *,
    source: str | None = None,
    note: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate event requirements against each plan's facility capabilities."""

    normalized = normalize_community_requirements(requirements)
    for plan in plans:
        allocations = [
            item for item in (plan.get("allocations") or [])
            if isinstance(item, dict)
        ]
        capability_status = {
            key: _capability_status(allocations, key)
            for key in CAPABILITY_LABELS
        }
        satisfied: list[str] = []
        unmet: list[str] = []
        unknown: list[str] = []
        requirement_details = []
        for requirement in normalized:
            definition = REQUIREMENT_DEFINITIONS[requirement]
            statuses = [
                capability_status[key]
                for key in definition["capabilities"]
            ]
            if "supported" in statuses:
                status = "satisfied"
                satisfied.append(requirement)
            elif statuses and all(item == "unsupported" for item in statuses):
                status = "unmet"
                unmet.append(requirement)
            else:
                status = "unknown"
                unknown.append(requirement)
            requirement_details.append({
                "requirement": requirement,
                "label": definition["label"],
                "status": status,
                "matching_capabilities": list(definition["capabilities"]),
            })

        labels = []
        for key, status in capability_status.items():
            if status == "supported":
                labels.append({"code": key, "label": CAPABILITY_LABELS[key], "status": status})
        for detail in requirement_details:
            if detail["status"] != "satisfied":
                labels.append({
                    "code": detail["requirement"],
                    "label": detail["label"],
                    "status": detail["status"],
                })

        total = len(normalized)
        summary = {
            "requirements": normalized,
            "requirement_details": requirement_details,
            "source": (source or "").strip() or None,
            "note": (note or "").strip()[:500] or None,
            "live_feed_used": False,
            "satisfied_requirements": satisfied,
            "unmet_requirements": unmet,
            "unknown_requirements": unknown,
            "community_requirements_met": len(satisfied),
            "community_requirements_total": total,
            "community_unmet_count": len(unmet),
            "community_unknown_count": len(unknown),
            "community_fit": round(len(satisfied) / total, 4) if total else None,
            "capability_status": capability_status,
            "labels": labels,
        }
        plan["community_summary"] = summary
        plan["community_requirements_met"] = len(satisfied)
        plan["community_requirements_total"] = total
        plan["community_unmet_count"] = len(unmet)
        plan["community_unknown_count"] = len(unknown)
        plan["community_fit"] = summary["community_fit"]
    return plans
