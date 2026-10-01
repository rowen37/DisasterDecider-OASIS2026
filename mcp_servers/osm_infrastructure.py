import json
import os
import sys
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

from overpass_client import OverpassError, post_overpass_async

mcp = FastMCP("OSM Infrastructure")


def _log(message: str) -> None:
    print(f"[osm_infrastructure] {message}", file=sys.stderr)


async def _run_overpass(query: str) -> tuple[dict, dict]:
    """Query Overpass via the shared client (mirror rotation, TTL cache,
    budget cap).

    Timeout chain: ops executor global MCP_CALL_TIMEOUT=60s -> tool total
    budget 55s -> per-mirror attempt <= server_timeout+15s, clamped to the
    remaining budget. Server queries use [timeout:25] with a 25+15=40s
    client timeout: the client must not give up before the server, or the
    query keeps running server-side and consumes quota. On a fusion-layer
    retry (re-invocation after a 3s backoff), failed mirrors are inside
    their 90s cooldown, so the retry lands on a healthy mirror.
    Diagnostics are logged for post-hoc review.
    """
    server_timeout = float(
        os.environ.get("OSM_OVERPASS_SERVER_TIMEOUT", "25")
    )
    budget = float(os.environ.get("OSM_OVERPASS_BUDGET_SECONDS", "55"))
    data, diag = await post_overpass_async(
        query,
        server_timeout_s=server_timeout,
        total_budget_s=budget,
    )
    if diag.get("stale"):
        _log(
            f"overpass: ALL mirrors failed — serving stale cache "
            f"(age {diag.get('cache_age_s')}s, within 24h fallback limit)"
        )
    elif diag.get("attempts"):
        _log(
            f"overpass: served by {diag.get('endpoint')} after "
            f"{len(diag['attempts'])} failed attempt(s): "
            f"{diag['attempts']}"
        )
    elif diag.get("cache_hit"):
        _log(f"overpass: cache hit (age {diag.get('cache_age_s')}s)")
    return data, diag


def _error_payload(exc: Exception) -> str:
    if isinstance(exc, OverpassError):
        return f"OSM Overpass API failed: {exc}"
    return f"OSM Overpass API failed: {type(exc).__name__}: {exc}"


@mcp.tool()
async def get_waterway_network(latitude: float, longitude: float, radius_km: float = 15) -> str:
    """
    Get OSM waterway polylines (river / stream / canal) near a point.

    Used by the stage-buffer fallback: when satellite imagery is
    missing or mistimed but the gauge peak confirms flooding, the
    waterway geometry feeds a first-order hydraulic-proximity extent.

    Args:
        latitude: latitude of the anchor point (usually the station)
        longitude: longitude of the anchor point
        radius_km: search radius (km)
    """
    radius_m = radius_km * 1000

    query = f"""
    [out:json][timeout:25];
    (
      way["waterway"~"^(river|canal|stream)$"](around:{int(radius_m)},{latitude},{longitude});
    );
    out geom;
    """

    try:
        data, diag = await _run_overpass(query)
    except Exception as exc:
        return json.dumps({"status": "error", "error": _error_payload(exc)})

    MAX_WATERWAYS = 400
    waterways = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry")
        if not (isinstance(geometry, list) and len(geometry) >= 2):
            continue
        coordinates = [
            [point["lon"], point["lat"]]
            for point in geometry
            if isinstance(point, dict)
            and isinstance(point.get("lon"), (int, float))
            and isinstance(point.get("lat"), (int, float))
        ]
        if len(coordinates) < 2:
            continue
        tags = element.get("tags", {})
        waterways.append({
            "osm_id": element.get("id"),
            "name": tags.get("name"),
            "waterway": tags.get("waterway"),
            "coordinates": coordinates,
        })

    return json.dumps({
        "status": "ok",
        "source": "OSM",
        "source_type": "waterway_network",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "waterways": waterways[:MAX_WATERWAYS],
        "waterway_count": len(waterways),
        "waterways_truncated": len(waterways) > MAX_WATERWAYS,
        "radius_km": radius_km,
        "metadata": {
            "overpass_endpoint": diag.get("endpoint"),
            "overpass_stale": bool(diag.get("stale")),
        },
        "metadata_verified": True,
    })


@mcp.tool()
async def get_road_status(latitude: float, longitude: float, radius_km: float = 10) -> str:
    """
    Get the road network status near a given location (via OSM).

    Args:
        latitude: latitude
        longitude: longitude
        radius_km: search radius (km)
    """
    radius_m = radius_km * 1000

    # Only road tags (type/name/bridge/tunnel) are needed, so `out tags`
    # suffices. A recursive node download (`out body; >; out skel qt;`)
    # can reach hundreds of MB in dense urban areas and is a main cause
    # of Overpass 502s/timeouts.
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"](around:{int(radius_m)},{latitude},{longitude});
    );
    out tags;
    """

    try:
        data, diag = await _run_overpass(query)
    except Exception as exc:
        return json.dumps({"status": "error", "error": _error_payload(exc)})

    MAX_ROADS = 500
    roads = []
    for element in data.get("elements", []):
        if element.get("type") == "way":
            tags = element.get("tags", {})
            roads.append({
                "id": element.get("id"),
                "highway_type": tags.get("highway"),
                "name": tags.get("name"),
                "surface": tags.get("surface"),
                "lanes": tags.get("lanes"),
                "maxspeed": tags.get("maxspeed"),
                "oneway": tags.get("oneway") == "yes",
                "bridge": tags.get("bridge") == "yes",
                "tunnel": tags.get("tunnel") == "yes",
            })

    return json.dumps({
        "status": "ok",
        "source": "OSM",
        "source_type": "road_network",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "measurements": {
            "road_count": {
                "value": len(roads),
                "unit": "count",
                "variable": "road_segment_count"
            },
            "bridge_count": {
                "value": sum(1 for r in roads if r["bridge"]),
                "unit": "count",
                "variable": "bridge_count"
            }
        },
        "metadata": {
            "overpass_endpoint": diag.get("endpoint"),
            "overpass_stale": bool(diag.get("stale")),
            "roads": roads[:MAX_ROADS],
            "roads_truncated": len(roads) > MAX_ROADS,
            "total_road_count": len(roads),
            "radius_km": radius_km
        },
        "metadata_verified": True
    })


@mcp.tool()
async def get_critical_infrastructure(latitude: float, longitude: float, radius_km: float = 10) -> str:
    """
    Get critical infrastructure near a given location (hospitals, fire
    stations, police, schools, shelters).

    Args:
        latitude: latitude
        longitude: longitude
        radius_km: search radius (km)
    """
    radius_m = radius_km * 1000

    # nwr covers node/way/relation; `out center` returns center
    # coordinates for ways/relations (a few hundred elements at most,
    # so the query is cheap)
    query = f"""
    [out:json][timeout:25];
    (
      nwr["amenity"="hospital"](around:{int(radius_m)},{latitude},{longitude});
      nwr["amenity"="fire_station"](around:{int(radius_m)},{latitude},{longitude});
      nwr["amenity"="police"](around:{int(radius_m)},{latitude},{longitude});
      nwr["amenity"="school"](around:{int(radius_m)},{latitude},{longitude});
      nwr["amenity"="shelter"](around:{int(radius_m)},{latitude},{longitude});
      nwr["emergency"="assembly_point"](around:{int(radius_m)},{latitude},{longitude});
    );
    out center;
    """

    try:
        data, diag = await _run_overpass(query)
    except Exception as exc:
        return json.dumps({"status": "error", "error": _error_payload(exc)})

    facilities = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})

        amenity = tags.get("amenity")
        emergency = tags.get("emergency")

        if amenity in {"hospital", "fire_station", "police", "school", "shelter"}:
            facility_type = amenity
        elif emergency == "assembly_point":
            facility_type = "assembly_point"
        else:
            continue

        # Nodes carry lat/lon at the top level; ways/relations carry a center subobject
        center = element.get("center", {})
        lat = element.get("lat") or center.get("lat")
        lon = element.get("lon") or center.get("lon")

        facilities.append({
            "id": element.get("id"),
            "osm_type": element.get("type"),   # node / way / relation
            "type": facility_type,
            "name": tags.get("name"),
            "latitude": lat,
            "longitude": lon,
            "address": tags.get("addr:full"),
            "phone": tags.get("phone"),
            "capacity": tags.get("capacity"),
        })

    return json.dumps({
        "status": "ok",
        "source": "OSM",
        "source_type": "infrastructure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": {"latitude": latitude, "longitude": longitude},
        "measurements": {
            "facility_count": {
                "value": len(facilities),
                "unit": "count",
                "variable": "critical_facility_count"
            },
            "hospital_count": {
                "value": sum(1 for f in facilities if f["type"] == "hospital"),
                "unit": "count",
                "variable": "hospital_count"
            },
            "shelter_count": {
                "value": sum(1 for f in facilities if f["type"] == "shelter"),
                "unit": "count",
                "variable": "shelter_count"
            }
        },
        "metadata": {
            "overpass_endpoint": diag.get("endpoint"),
            "overpass_stale": bool(diag.get("stale")),
            "facilities": facilities,
            "radius_km": radius_km
        },
        "metadata_verified": True
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
