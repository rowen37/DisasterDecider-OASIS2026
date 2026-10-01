import json
import os
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NWS Alerts")

# ------------------------- Tools -------------------------
@mcp.tool()
async def get_flood_warnings(
    latitude: float, longitude: float, radius_km: float | None = None
) -> str:
    """Get flood-related active alerts that apply to the target point.

    radius_km is accepted for older source configurations; it does not
    change this point-specific query.
    """
    latitude = round(latitude, 4)
    longitude = round(longitude, 4)
    nws_url = "https://api.weather.gov/alerts/active"
    headers = {
        "User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, contact@example.com)")
    }

    # NWS resolves polygon and zone alerts for this point. A statewide
    # search would include unrelated warnings elsewhere in the state.
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                nws_url,
                params={
                    "point": f"{latitude},{longitude}",
                    "status": "actual",
                },
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

        filtered.append(feature)

    return json.dumps({
        "status": "ok",
        "source": "NWS",
        "source_type": "flood_warnings",
        "scope": "target_point",
        "location": {"latitude": latitude, "longitude": longitude},
        "filtered_count": len(filtered),
        "warnings": filtered,
        "metadata_verified": True
    }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")
