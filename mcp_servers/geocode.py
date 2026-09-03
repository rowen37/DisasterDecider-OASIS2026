import json
import os
import re
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Geocoder")

# Nominatim policy requires a contactable User-Agent; deployments can
# supply one via env var.
_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "DisasterAgent/1.0")
# The data stack (USGS/NWS/Census) covers the US only, so geocoding is
# pinned to the US to avoid upstream hits (e.g. Athens -> Greece) that
# fail everywhere downstream.
_COUNTRY_CODES = os.getenv("GEOCODE_COUNTRY_CODES", "us")


def _candidate_score(query: str, candidate: dict) -> float:
    """Disambiguation score: query-token coverage in display_name plus
    Nominatim importance. Taking the first result alone puts "Manhattan"
    in New York (importance 0.74 vs 0.57 for Kansas); state-qualified
    queries are corrected via token coverage."""
    tokens = [
        t for t in re.findall(r"[a-z]+", query.lower()) if len(t) > 2
    ]
    display = (candidate.get("display_name") or "").lower()
    coverage = sum(1.0 for t in tokens if t in display)
    importance = float(candidate.get("importance") or 0.0)
    return 2.0 * coverage + importance


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2)
        * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _pick_best(
    query: str,
    candidates: list,
    bias_lat: float | None = None,
    bias_lon: float | None = None,
) -> dict | None:
    """Pick the best candidate.

    With a bias point (anchor coordinates, usually a verified USGS
    station location), anchor-based disambiguation applies: prefer the
    highest-scoring candidate within 150 km of the anchor, falling back
    to global scoring only if no candidate is nearby. Station IDs are
    unique while place names are ambiguous labels - the anchor comes
    from the unique key, so it outranks any text-based scoring.
    """
    if not candidates:
        return None
    if bias_lat is not None and bias_lon is not None:
        def _dist(c):
            try:
                return _haversine_km(
                    bias_lat, bias_lon,
                    float(c["lat"]), float(c["lon"]),
                )
            except (KeyError, TypeError, ValueError):
                return float("inf")

        near = [c for c in candidates if _dist(c) <= 150.0]
        pool = near or candidates
        return max(pool, key=lambda c: _candidate_score(query, c))
    return max(
        candidates, key=lambda c: _candidate_score(query, c)
    )


@mcp.tool()
async def geocode_location(
    place_name: str,
    bias_lat: float | None = None,
    bias_lon: float | None = None,
) -> str:
    """
    Convert a place name to latitude/longitude coordinates
    (OpenStreetMap Nominatim).

    Restricted to the US; with multiple candidates, the best is chosen
    by "query-token coverage + importance". Optional bias_lat/bias_lon
    are anchor coordinates (usually a verified hydrologic station):
    for ambiguous names (e.g. Manhattan NY vs KS), candidates within
    150 km of the anchor are preferred.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 5,
        "countrycodes": _COUNTRY_CODES,
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return json.dumps({
            "status": "error",
            "error": f"Nominatim geocoding failed: {exc}",
            "place_name": place_name,
        }, ensure_ascii=False)

    place = _pick_best(place_name, data or [], bias_lat, bias_lon)
    if place is None:
        return f"未找到地名: {place_name}"

    return f"{place_name} 的坐标: 纬度 {place['lat']}, 经度 {place['lon']}"

@mcp.tool()
async def geocode_boundary(
    place_name: str,
    bias_lat: float | None = None,
    bias_lon: float | None = None,
) -> str:
    """
    Get the administrative boundary polygon (GeoJSON Polygon) for a
    place name, so analysis can be clipped to the target city instead
    of an arbitrary rectangular buffer.

    Returns JSON:
    {
      "status": "ok",
      "name": ...,
      "display_name": ...,
      "bbox": [min_lon, min_lat, max_lon, max_lat],
      "geojson": {"type": "Polygon"/"MultiPolygon", ...}  # EPSG:4326
    }
    If no boundary is found, status=unavailable (callers must degrade
    gracefully, never fabricate a boundary).
    """
    import json

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 5,
        "polygon_geojson": 1,
        "countrycodes": _COUNTRY_CODES,
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url, params=params, headers=headers, timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return json.dumps(
            {"status": "unavailable",
             "error": f"geocoder request failed: {exc}"},
            ensure_ascii=False,
        )

    if not data:
        return json.dumps(
            {"status": "unavailable", "error": "place not found"},
            ensure_ascii=False,
        )

    # Same disambiguation scoring as geocode_location (including the
    # station-anchoring bias) so coordinates and boundary refer to
    # the same place.
    place = _pick_best(place_name, data, bias_lat, bias_lon)
    geojson = place.get("geojson")

    if not isinstance(geojson, dict) or geojson.get("type") not in (
        "Polygon",
        "MultiPolygon",
    ):
        return json.dumps(
            {
                "status": "unavailable",
                "error": "no polygon boundary returned",
                "name": place.get("name", place_name),
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "name": place.get("name", place_name),
            "display_name": place.get("display_name"),
            "osm_type": place.get("osm_type"),
            "bbox": place.get("boundingbox"),
            "geojson": geojson,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")