import json
import os
import httpx
import math
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NWS Alerts")

# ------------------------- Helpers -------------------------
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _get_polygon_center(geometry: dict) -> tuple[float, float] | None:
    if not geometry:
        return None
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None

    ring = None
    if geom_type == "Polygon" and coords:
        ring = coords[0]
    elif geom_type == "MultiPolygon" and coords and coords[0]:
        ring = coords[0][0]

    if not ring or len(ring) < 3:
        return None

    # Drop the last repeated point of a closed GeoJSON ring
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    lat_sum = sum(pt[1] for pt in pts)
    lon_sum = sum(pt[0] for pt in pts)
    n = len(pts)
    return (lat_sum / n, lon_sum / n)

# ------------------------- Tools -------------------------
@mcp.tool()
async def get_flood_warnings(latitude: float, longitude: float) -> str:
    """Get flood-related active alerts near a given location."""
    latitude = round(latitude, 4)
    longitude = round(longitude, 4)
    nws_url = "https://api.weather.gov/alerts/active"
    headers = {
        "User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, contact@example.com)")
    }

    # Step 1: resolve the state code for this point
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            point_resp = await client.get(
                f"https://api.weather.gov/points/{latitude},{longitude}",
                headers=headers
            )
            point_resp.raise_for_status()
            point_data = point_resp.json()
            state = point_data["properties"]["relativeLocation"]["properties"]["state"]
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"Failed to resolve state from point: {str(exc)}",
            "metadata_verified": False
        }, ensure_ascii=False)

    # Step 2: fetch all active flood alerts for that state
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                nws_url,
                params={"area": state, "status": "actual"},
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"NWS flood warnings API request failed: {str(exc)}",
            "metadata_verified": False
        }, ensure_ascii=False)

    features = data.get("features", [])
    filtered = []
    for feature in features:
        props = feature.get("properties", {})
        event_name = props.get("event", "")

        if "flood" not in event_name.lower():
            continue

        # Step 3: keep alerts without geometry too; rank those with one by distance
        geometry = feature.get("geometry")
        if geometry:
            center = _get_polygon_center(geometry)
            if center:
                dist = _haversine_km(latitude, longitude, center[0], center[1])
                feature["_distance_to_center_km"] = round(dist, 2)

        filtered.append(feature)

    # Sort by distance (alerts without geometry sort last)
    filtered.sort(key=lambda f: f.get("_distance_to_center_km", float("inf")))

    return json.dumps({
        "status": "ok",
        "source": "NWS",
        "source_type": "flood_warnings",
        "resolved_state": state,
        "location": {"latitude": latitude, "longitude": longitude},
        "filtered_count": len(filtered),
        "warnings": filtered,
        "metadata_verified": True
    }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")