import re
import os
import httpx
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Flood Alert")

# USGS missing-value sentinels: must not win a max() comparison or
# leak into the payload.
MISSING_SENTINELS = {"", "-999999", "-99999", "NaN"}


def select_stage_observation(
    values: list,
    windowed: bool,
) -> tuple | None:
    """Choose the primary stage value and build the hydrograph summary.

    windowed=False: primary = latest instantaneous value (realtime).
    windowed=True: primary = the window PEAK -- an event's severity IS
    its peak stage (NWPS flood categories are defined on instantaneous
    stage), and a flash flood that recedes inside the window would be
    acquitted by its window-end snapshot (Lodi NJ 2026-09-13 case).

    Returns (primary_value_dict, window_summary | None, semantics) or
    None when every value is a missing sentinel.
    """
    clean = [
        v for v in values
        if str(v.get("value", "")).strip() not in MISSING_SENTINELS
    ]
    if not clean:
        return None

    latest = clean[-1]
    if not windowed:
        return latest, None, "latest_instantaneous"

    peak = max(clean, key=lambda v: float(v["value"]))
    window_summary = {
        "value_count": len(clean),
        "peak_stage_ft": float(peak["value"]),
        "peak_time": peak.get("dateTime"),
        "end_stage_ft": float(latest["value"]),
        "end_time": latest.get("dateTime"),
        "min_stage_ft": float(
            min(clean, key=lambda v: float(v["value"]))["value"]
        ),
    }
    return peak, window_summary, "window_peak"

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
    return the PEAK observation within the window (historical replay)
    plus a window_summary describing the full hydrograph (peak / end
    / minimum stage with timestamps).

    Peak semantics rationale: flood severity of a past event IS its
    peak stage -- NWPS flood categories are defined on instantaneous
    stage, and a flash flood that rises and recedes inside one window
    is misrepresented by the window-end value (it reads "in bank"
    after the event has already happened). The end value is preserved
    in window_summary.end_stage_ft so trend analysis stays possible.
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

        selection = select_stage_observation(
            values,
            windowed=bool(start_dt and end_dt),
        )
        if selection is None:
            return json.dumps({
                "status": "error",
                "error": (
                    "USGS returned no usable stage values for this "
                    "window (all missing/sentinel)."
                ),
                "station_id": station_id,
            }, ensure_ascii=False)

        primary, window_summary, observation_semantics = selection

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
                    "water_level":      float(primary["value"]),
                    "unit":             os.getenv("USGS_STAGE_UNIT", "ft"),
                    "observation_time": primary["dateTime"],
                    "observation_semantics": observation_semantics,
                    "window_summary": (
                        {
                            "start": start_dt,
                            "end": end_dt,
                            **window_summary,
                        }
                        if window_summary is not None
                        else None
                    ),
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