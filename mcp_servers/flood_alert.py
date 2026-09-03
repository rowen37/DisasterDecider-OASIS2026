import re
import os
import httpx
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Flood Alert")

@mcp.tool()
async def get_flood_observation(
    station_id: str,
    start_dt: str = "",
    end_dt: str = "",
) -> str:
    """
    Get a river stage observation from USGS.

    Without start_dt/end_dt: return the latest instantaneous value
    (live mode). With start_dt/end_dt (ISO 8601 or YYYY-MM-DD):
    return the observation closest to end_dt within the window
    (historical replay — natively supported by the USGS IV API via
    the startDT/endDT query parameters).
    """

    url = os.environ.get("USGS_WATER_API_URL")
    if not url:
        return json.dumps({
            "status": "error",
            "error": "USGS_WATER_API_URL is not configured.",
            "metadata_verified": False,
        }, ensure_ascii=False)

    params = {
        "format": "json",
        "sites": station_id,
        "parameterCd": os.environ.get(
            "USGS_STAGE_PARAMETER_CODE", "00065"
        ),
        "siteStatus": "active",
    }

    # Historical window: empty string = not provided (FastMCP str
    # parameters do not accept None).
    if start_dt and end_dt:
        params["startDT"] = start_dt
        params["endDT"] = end_dt

    try:
        # follow_redirects: waterservices.usgs.gov permanently redirects
        # to nwis.waterservices.usgs.gov; without following, the request
        # fails with 301.
        async with httpx.AsyncClient(follow_redirects=True) as client:

            resp = await client.get(
                url,
                params=params,
                timeout=float(
                    os.environ.get("MCP_HTTP_TIMEOUT", "30.0")
                ),
            )

            resp.raise_for_status()

            data = resp.json()
    except Exception as exc:
        # Network / status-code / timeout failures degrade to a
        # structured error, never a raw crash (same contract as
        # parse errors).
        return json.dumps({
            "status": "error",
            "error": f"USGS water services request failed: {exc}",
            "metadata_verified": False,
            "station_id": station_id,
        }, ensure_ascii=False)

    try:
        time_series = (
            data["value"]
            ["timeSeries"][0]
        )

        values = (
            time_series
            ["values"][0]
            ["value"]
        )

        latest = values[-1]

        # ── Extract coordinates and station name from the response ──
        source_info = time_series.get("sourceInfo", {})
        station_name = source_info.get("siteName")
        geo = (
            source_info
            .get("geoLocation", {})
            .get("geogLocation", {})
        )
        obs_latitude  = geo.get("latitude")
        obs_longitude = geo.get("longitude")

        # Both come from the USGS response body, hence treated as
        # verified.
        location_verified = (
            obs_latitude is not None and obs_longitude is not None
        )

        return json.dumps(
            {
                "status": "ok",
                "query_window": (
                    {"start": start_dt, "end": end_dt}
                    if start_dt and end_dt
                    else None
                ),
                "observation": {
                    "station_id":       station_id,
                    "water_level":      float(latest["value"]),
                    "unit":             os.getenv("USGS_STAGE_UNIT", "ft"),
                    "observation_time": latest["dateTime"],
                    "source":           "USGS",
                    "station_name":     station_name,
                    "latitude":         obs_latitude,
                    "longitude":        obs_longitude,

                    # Data comes straight from the USGS live API, so the
                    # observation itself is verified; flood thresholds
                    # are verified separately via get_nwps_gauge().
                    "metadata_verified":  True,
                    "location_verified":  location_verified,
                    "threshold_verified": False,   # thresholds not fetched here

                    "metadata": {
                        "observation_verified": True,
                        "source_verified":      True,
                        "threshold_source":     "not_fetched",
                        "note": (
                            "threshold_verified refers to this "
                            "observation payload only: it carries no "
                            "flood thresholds. The pipeline verifies "
                            "NWPS flood categories via get_nwps_gauge() "
                            "separately when available, which is the "
                            "authoritative threshold evidence."
                        ),
                    },
                },
            },
            ensure_ascii=False,
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:

        return json.dumps(
            {
                "status": "error",
                "error": (
                    "Unable to parse USGS "
                    f"water observation: {exc}"
                ),
            },
            ensure_ascii=False,
        )

if __name__ == "__main__":
    mcp.run(transport="stdio")