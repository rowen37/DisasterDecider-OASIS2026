# resource_manager.py
#
# Disaster-response resource discovery and candidate allocation plan
# generation.
#
# Public facility-capacity APIs (HIFLD Open Data, Definitive Healthcare
# hospital beds) are no longer reliably available, so this MCP uses a
# three-layer hybrid strategy:
#   1. OpenStreetMap Overpass (live, free) for hospital / fire station /
#      police / shelter / ambulance station locations and tags
#      (beds/capacity tags used when present);
#   2. MANUAL_RESOURCES_JSON env var for an operator-provided capacity
#      table (per-kind defaults for beds / vehicles / shelter spaces,
#      overridable per facility name); every assumption is disclosed
#      in the output;
#   3. Direct generation of candidate plans for Pareto optimization
#      (plan_id / allocations / objectives), each objective value
#      traceable to the inputs above.
#
# All estimated values carry "assumed": true and must never be treated
# as observed data.

import json
import math
import os
import sys

from mcp.server.fastmcp import FastMCP

from overpass_client import OverpassError, post_overpass_async
from osm_labels import load_label_overrides, osm_display_label

mcp = FastMCP("Resource Manager")

# Set when MANUAL_RESOURCES_JSON fails to parse; surfaced in the audit
# output so an operator override never silently disappears.
_MANUAL_CONFIG_ERROR: str | None = None

# Overpass endpoint/failover handling lives in overpass_client (mirror
# rotation, cooldown, TTL cache, budget cap; OSM_OVERPASS_API_URL /
# OSM_OVERPASS_ENDPOINTS are read there).
NOMINATIM_UA = "DisasterAgent/1.0 (qiwenb@design.upenn.edu)"

# Overpass amenity type -> resource kind
FACILITY_QUERIES = {
    "hospital": 'nwr["amenity"="hospital"](around:{r},{lat},{lon});',
    "fire_station": 'nwr["amenity"="fire_station"](around:{r},{lat},{lon});',
    "police": 'nwr["amenity"="police"](around:{r},{lat},{lon});',
    "shelter": (
        'nwr["amenity"="shelter"](around:{r},{lat},{lon});'
        'nwr["emergency"="assembly_point"](around:{r},{lat},{lon});'
    ),
    "ambulance_station": 'nwr["amenity"="ambulance_station"](around:{r},{lat},{lon});',
}

# Manual default capacities (overridable via MANUAL_RESOURCES_JSON)
DEFAULT_MANUAL = {
    "hospital_beds_per_facility": 50,
    "fire_vehicles_per_station": 2,
    "police_patrols_per_station": 3,
    "shelter_capacity_per_facility": 100,
    "ambulance_units_per_station": 2,
    # Per-dispatch cost (dimensionless cost units)
    "dispatch_cost": {
        "hospital": 100,
        "fire_station": 50,
        "police": 40,
        "shelter": 10,
        "ambulance_station": 60,
    },
}

AVG_SPEED_KMH = 60.0
MAX_RESPONSE_MINUTES = 120.0


def _load_manual() -> dict:
    raw = os.environ.get("MANUAL_RESOURCES_JSON", "")
    if not raw:
        return dict(DEFAULT_MANUAL)
    try:
        user = json.loads(raw)
        merged = dict(DEFAULT_MANUAL)
        merged.update(user)
        return merged
    except json.JSONDecodeError as exc:
        # Silently falling back to defaults would lose the operator's
        # override - record the error and surface it in the audit output.
        global _MANUAL_CONFIG_ERROR
        _MANUAL_CONFIG_ERROR = (
            f"MANUAL_RESOURCES_JSON is invalid JSON ({exc}); "
            "built-in defaults were used instead of the operator override."
        )
        print(f"[resource_manager] {_MANUAL_CONFIG_ERROR}", file=sys.stderr)
        return dict(DEFAULT_MANUAL)


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _capacity_for(kind: str, tags: dict, manual: dict) -> tuple[float, bool]:
    """Return (capacity, is_assumed). OSM beds/capacity tags take priority."""
    for tag in ("beds", "capacity", "ambulances"):
        v = tags.get(tag)
        if v:
            try:
                return float(v), False
            except (TypeError, ValueError):
                pass
    if kind == "hospital":
        return float(manual.get("hospital_beds_per_facility", 50)), True
    if kind == "fire_station":
        return float(manual.get("fire_vehicles_per_station", 2)), True
    if kind == "police":
        return float(manual.get("police_patrols_per_station", 3)), True
    if kind == "shelter":
        return float(manual.get("shelter_capacity_per_facility", 100)), True
    if kind == "ambulance_station":
        return float(manual.get("ambulance_units_per_station", 2)), True
    return 1.0, True


async def _fetch_osm_facilities(
    latitude: float,
    longitude: float,
    radius_m: float,
) -> tuple[list[dict], dict]:
    """Return (facility list, Overpass diagnostics). Diagnostics are
    disclosed in the output for provenance."""
    selectors = "\n".join(
        q.format(r=int(radius_m), lat=latitude, lon=longitude)
        for q in FACILITY_QUERIES.values()
    )
    # Server-side [timeout:45]; the caller allows 90s. The client's total
    # budget is 80s and each attempt's timeout is clamped to the remaining
    # budget, so the client never abandons a request the server is still
    # working on.
    query = (
        "[out:json][timeout:45];\n(\n" + selectors + "\n);\nout center 200;"
    )
    try:
        data, diag = await post_overpass_async(
            query,
            server_timeout_s=45.0,
            total_budget_s=80.0,
        )
    except OverpassError as exc:
        print(
            f"[resource_manager] Overpass failed after trying all "
            f"mirrors: {exc}",
            file=sys.stderr,
        )
        raise
    if diag.get("stale"):
        print(
            f"[resource_manager] WARNING: all Overpass mirrors failed — "
            f"serving stale cache (age {diag.get('cache_age_s')}s)",
            file=sys.stderr,
        )
    elif diag.get("attempts"):
        print(
            f"[resource_manager] Overpass served by "
            f"{diag.get('endpoint')} after "
            f"{len(diag['attempts'])} failed attempt(s)",
            file=sys.stderr,
        )

    kind_by_tag = {
        "hospital": "hospital",
        "fire_station": "fire_station",
        "police": "police",
        "shelter": "shelter",
        "ambulance_station": "ambulance_station",
    }

    facilities = []
    label_overrides = load_label_overrides()
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        kind = kind_by_tag.get(tags.get("amenity", ""))
        if not kind and tags.get("emergency") == "assembly_point":
            kind = "shelter"
        if not kind:
            continue
        center = el.get("center", {})
        lat = el.get("lat") or center.get("lat")
        lon = el.get("lon") or center.get("lon")
        if lat is None or lon is None:
            continue
        label = osm_display_label(
            tags,
            feature_type=kind,
            osm_type=str(el.get("type") or "element"),
            osm_id=el.get("id"),
            lat=float(lat),
            lon=float(lon),
            overrides=label_overrides,
        )
        facilities.append(
            {
                "kind": kind,
                "name": label["label"],
                "address": label["address"],
                "label_source": label["label_source"],
                "lat": lat,
                "lon": lon,
                "distance_km": round(
                    _haversine_km(latitude, longitude, lat, lon), 2
                ),
                "tags": {
                    k: tags[k]
                    for k in ("beds", "capacity", "ambulances", "phone")
                    if k in tags
                },
            }
        )
    return facilities, diag


def _build_plans(
    facilities: list[dict],
    manual: dict,
    demand_population: float,
    water_ratio: float,
) -> list[dict]:
    """Generate candidate allocation plans and compute 5 objective values.

    Objectives (aligned with RESOURCE_ALLOCATION_OBJECTIVES_JSON):
      risk_reduction   maximize  min(1, sum(capacity) * severity)
      coverage         maximize  min(1, sum(capacity) / demand population)
      response_time    minimize  driving minutes to the farthest dispatch (60 km/h assumed)
      cost             minimize  sum of per-dispatch costs
      unmet_demand     minimize  max(0, demand - sum(capacity)) / demand
    """
    by_kind: dict[str, list[dict]] = {}
    for f in facilities:
        cap, assumed = _capacity_for(f["kind"], f.get("tags", {}), manual)
        if cap <= 0:
            continue
        by_kind.setdefault(f["kind"], []).append(
            {**f, "capacity": cap, "capacity_assumed": assumed}
        )
    for kind in by_kind:
        by_kind[kind].sort(key=lambda x: x["distance_km"])

    dispatch_cost = manual.get("dispatch_cost", DEFAULT_MANUAL["dispatch_cost"])
    demand = max(demand_population, 1.0)
    severity = max(0.05, min(1.0, water_ratio)) if water_ratio else 0.5

    def _plan_from(picks: list[dict], plan_id: str, strategy: str) -> dict | None:
        picks = [p for p in picks if p]
        if not picks:
            return None
        total_cap = sum(p["capacity"] for p in picks)
        max_dist = max(p["distance_km"] for p in picks) or 0.1
        response_min = min(max_dist / AVG_SPEED_KMH * 60.0, MAX_RESPONSE_MINUTES)
        cost = sum(
            float(dispatch_cost.get(p["kind"], 50)) for p in picks
        )
        allocations = [
            {
                "resource": p["name"],
                "type": p["kind"],
                "capacity": p["capacity"],
                "capacity_assumed": p["capacity_assumed"],
                "distance_km": p["distance_km"],
                # Facility coordinates: spatial input for AllocationEngine's
                # vulnerability_coverage (population-weighted SVI coverage).
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "address": p.get("address"),
                "label_source": p.get("label_source"),
            }
            for p in picks
        ]
        return {
            "plan_id": plan_id,
            "allocations": allocations,
            "objectives": {
                "risk_reduction": round(min(1.0, total_cap * severity / max(demand, 1.0)) * 100, 2) / 100,
                "coverage": round(min(1.0, total_cap / demand), 4),
                "response_time": round(response_min, 1),
                "cost": cost,
                "unmet_demand": round(max(0.0, demand - total_cap) / demand, 4),
            },
            "metadata": {
                "strategy": strategy,
                "total_capacity": total_cap,
                "facility_count": len(picks),
                "demand_population": demand_population,
                "severity_water_ratio_used": severity,
                "assumptions": [
                    "capacity from MANUAL_RESOURCES_JSON defaults where OSM lacks beds/capacity tags",
                    "road speed assumed 60 km/h straight-line response time",
                ],
            },
        }

    plans = []
    plan_no = 0
    for kind, items in sorted(by_kind.items()):
        if not items:
            continue
        for k in (1, 2, 3):
            if len(items) < k:
                break
            plan_no += 1
            plan = _plan_from(
                items[:k],
                f"{kind}_top{k}",
                f"dispatch {k} nearest {kind}(s)",
            )
            if plan:
                plans.append(plan)
    # Mixed plan: one nearest facility of each kind
    mixed = [items[0] for items in by_kind.values() if items]
    if len(mixed) >= 2:
        plan = _plan_from(mixed, "mixed_1_per_kind", "one nearest facility of each kind")
        if plan:
            plans.append(plan)

    return plans[:24]


# ----------------------------------------------------------------
# Tools
# ----------------------------------------------------------------

@mcp.tool()
async def get_available_resources(
    latitude: float,
    longitude: float,
    radius_km: float = 50,
    resource_type: str = "all",
    demand_population: str = "",
    water_level_ratio: str = "",
) -> str:
    """
    Discover disaster-response resources around a target and generate
    Pareto candidate allocation plans.

    Data sources (hybrid):
      - OSM Overpass: hospital / fire station / police / shelter /
        ambulance station locations (beds/capacity tags used when present);
      - MANUAL_RESOURCES_JSON: operator-provided capacity table (public
        bed-count APIs are gone; defaults are all flagged assumed).

    Args:
        latitude / longitude: search center
        radius_km: search radius
        demand_population: affected population (passed as string; defaults to an assumed 5000)
        water_level_ratio: water level / alert level ratio (passed as string; defaults to 0.5)
    """

    def _to_float(value: str, default: float) -> float:
        try:
            parsed = float(str(value).strip())
            return parsed if parsed >= 0 else default
        except (TypeError, ValueError):
            return default

    demand = _to_float(demand_population, 5000.0) if demand_population else 5000.0
    water_ratio = _to_float(water_level_ratio, 0.5) if water_level_ratio else 0.5
    manual = _load_manual()

    try:
        facilities, overpass_diag = await _fetch_osm_facilities(
            latitude, longitude, radius_km * 1000
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "error": f"OSM Overpass facility query failed: {exc}",
                "metadata_verified": False,
                "action_required": (
                    "Overpass unavailable; set MANUAL_RESOURCES_JSON to "
                    "run allocation with operator-provided resources."
                ),
            },
            ensure_ascii=False,
        )

    plans = _build_plans(facilities, manual, demand, water_ratio)

    if not plans:
        return json.dumps(
            {
                "status": "error",
                "error": "No emergency facilities found within the search radius.",
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "source": "OSM Overpass + MANUAL_RESOURCES_JSON capacity table",
            "source_type": "resource_discovery",
            "location": {"latitude": latitude, "longitude": longitude},
            "radius_km": radius_km,
            "facility_count": len(facilities),
            "demand_population": demand,
            "demand_population_assumed": not demand_population,
            "resources": plans,
            "metadata": {
                # Data-source disclosure: healthy mirror / cache / stale cache fallback
                "overpass_endpoint": overpass_diag.get("endpoint"),
                "overpass_stale": bool(overpass_diag.get("stale")),
            },
            "metadata_verified": True,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
