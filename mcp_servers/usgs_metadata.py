# usgs_metadata.py

import asyncio
import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("USGS Metadata")


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

MAX_RETRIES = 3
BASE_BACKOFF = 1.0
RETRYABLE_CODES = {502, 503, 504}


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _parse_rdb_site(text: str) -> dict[str, str]:
    """
    Parse a USGS RDB site-service response into a flat dict.

    Returns the first data row as {column_name: value}.
    Raises ValueError if the response is empty or malformed.
    """

    non_comment = [
        line
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]

    if len(non_comment) < 3:
        raise ValueError(
            "RDB response contains insufficient rows "
            f"({len(non_comment)} non-comment lines found)."
        )

    headers = non_comment[0].split("\t")
    data_values = non_comment[2].split("\t")

    if len(data_values) != len(headers):
        raise ValueError(
            "RDB column count mismatch: "
            f"{len(headers)} headers vs {len(data_values)} values."
        )

    return dict(zip(headers, data_values))


def _safe_float_str(value: str | None) -> float | None:
    if not value or not value.strip():
        return None

    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None


async def _http_get_with_retry(
    url: str,
    params: dict | None,
    timeout: float,
) -> httpx.Response:
    
    last_exc: Exception | None = None
    # follow_redirects: waterservices.usgs.gov permanently redirects to
    # nwis.waterservices.usgs.gov; without following, requests fail with 301.
    async with httpx.AsyncClient(follow_redirects=True) as client:

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(
                    url,
                    params=params or {},
                    timeout=timeout,
                )
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if (
                    exc.response.status_code
                    in RETRYABLE_CODES
                    and attempt < MAX_RETRIES - 1
                ):
                    wait = BASE_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue
                raise

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue
                raise

    raise RuntimeError(
        "Retry loop exited unexpectedly"
    ) from last_exc


def _normalize_nwps_flood_categories(
    gauge_data: dict,
) -> dict:
    """
    Extract NWPS flood-category stage thresholds without
    inventing or hardcoding values.

    NWPS stores flood categories under:
        flood.categories

    Expected categories:
        action
        minor
        moderate
        major

    If a category is absent or its stage is unavailable,
    the value remains None.
    """

    flood = gauge_data.get("flood")

    if not isinstance(flood, dict):
        return {}

    categories = flood.get("categories")

    if not isinstance(categories, dict):
        return {}

    normalized: dict[str, dict] = {}

    for category_name in (
        "action",
        "minor",
        "moderate",
        "major",
    ):

        category = categories.get(category_name)

        if not isinstance(category, dict):
            continue

        normalized[category_name] = {
            "stage": category.get("stage"),
            "unit": category.get("unit"),
            "impact": category.get("impact"),
        }

    return normalized


# ----------------------------------------------------------------
# USGS Station Metadata
# ----------------------------------------------------------------

@mcp.tool()
async def get_station_metadata(
    station_id: str,
) -> str:
    """
    Get full base metadata for a USGS monitoring station.

    Data source:
        USGS NWIS Site Service

    Note:
        This tool only returns USGS station metadata.
        Flood stage / flood category are not inferred here.
        Use get_nwps_gauge() for flood categories.
    """

    url = os.environ.get(
        "USGS_SITE_API_URL",
        "https://waterservices.usgs.gov/nwis/site/",
    )

    timeout = float(
        os.environ.get(
            "MCP_HTTP_TIMEOUT",
            15.0,
        )
    )

    station_id = str(station_id).strip()

    if not station_id:

        return json.dumps(
            {
                "status": "error",
                "error": "station_id is empty.",
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    params = {
        "format": "rdb",
        "sites": station_id,
        "siteOutput": "expanded",
    }

    try:

        response = await _http_get_with_retry(
            url=url,
            params=params,
            timeout=timeout,
        )

        text = response.text

    except Exception as exc:

        return json.dumps(
            {
                "status": "error",
                "error": (
                    "USGS site API request failed after "
                    f"{MAX_RETRIES} attempts: {exc}"
                ),
                "station_id": station_id,
                "metadata_verified": False,
                "action_required": (
                    "Verify the USGS station ID and "
                    "USGS Site Service availability."
                ),
            },
            ensure_ascii=False,
        )

    try:

        row = _parse_rdb_site(text)

        latitude = _safe_float_str(
            row.get("dec_lat_va")
        )

        longitude = _safe_float_str(
            row.get("dec_long_va")
        )

        if latitude is None or longitude is None:

            raise ValueError(
                "USGS station metadata does not contain "
                "valid latitude/longitude."
            )

        return json.dumps(
            {
                "status": "ok",
                "metadata": {
                    "station_id": station_id,
                    "station_name": (row.get("station_nm","",).strip()or None),
                    "latitude": latitude,
                    "longitude": longitude,
                    "site_type": (row.get("site_tp_cd","",).strip()or None),
                    "state": (row.get("state_cd","",).strip()or None),
                    "county": (row.get("county_cd","",).strip()or None),
                    "drainage_area_sq_mi": (_safe_float_str(row.get("drain_area_va"))),
                    "contributing_drainage_area_sq_mi": (_safe_float_str(row.get("contrib_drain_area_va"))),
                    "metadata_verified": True,
                    "source": "USGS",
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
                    "Failed to parse USGS RDB metadata: "
                    f"{exc}"
                ),
                "station_id": station_id,
                "metadata_verified": False,
                "action_required": (
                    "Do NOT proceed to LLM inference "
                    "with incomplete station metadata."
                ),
            },
            ensure_ascii=False,
        )


# ----------------------------------------------------------------
# NWPS Gauge Metadata
# ----------------------------------------------------------------

@mcp.tool()
async def get_nwps_gauge(
    station_id: str,
) -> str:
    """
    Get the NWPS gauge metadata corresponding to a USGS station.

    NWPS accepts a USGS station ID as the gauge identifier:

        /v1/gauges/{identifier}

    Returns:
        - NWPS gauge LID
        - USGS station ID
        - reach ID
        - gauge name
        - RFC
        - WFO
        - state
        - coordinates
        - current status (if NWPS provides it)
        - flood category thresholds (if NWPS provides them)

    Flood stage must never be guessed.
    """

    nwps_api_url = os.environ.get(
        "NWPS_API_URL",
        "https://api.water.noaa.gov/nwps/v1",
    ).rstrip("/")

    timeout = float(
        os.environ.get(
            "MCP_HTTP_TIMEOUT",
            15.0,
        )
    )

    station_id = str(station_id).strip()
    if not station_id:

        return json.dumps(
            {
                "status": "error",
                "error": "station_id is empty.",
                "station_id": station_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    url = (
        f"{nwps_api_url}/gauges/"
        f"{station_id}"
    )

    try:
        response = await _http_get_with_retry(
            url=url,
            params={},
            timeout=timeout,
        )
        data = response.json()

    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS gauge metadata request failed "
                    f"after {MAX_RETRIES} attempts: {exc}"
                ),
                "station_id": station_id,
                "metadata_verified": False,
                "action_required": (
                    "Verify that this USGS station has "
                    "a corresponding NWPS gauge."
                ),
            },
            ensure_ascii=False,
        )

    if not isinstance(data, dict):
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS gauge API returned an invalid "
                    "JSON object."
                ),
                "station_id": station_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    usgs_id = str(
        data.get(
            "usgsId",
            "",
        )
    ).strip()

    if usgs_id and usgs_id != station_id:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS gauge identity mismatch: requested "
                    f"USGS station {station_id}, but NWPS returned "
                    f"USGS station {usgs_id}."
                ),
                "station_id": station_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    gauge_id = data.get("lid")

    if not gauge_id:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS gauge response does not contain "
                    "a gauge LID."
                ),
                "station_id": station_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    flood_categories = (
        _normalize_nwps_flood_categories(
            data
        )
    )

    return json.dumps(
        {
            "status": "ok",
            "source": "NOAA/NWS NWPS",
            "source_type": "nwps_gauge_metadata",
            "station_id": station_id,
            "gauge": {
                "lid": gauge_id,
                "usgs_id": data.get("usgsId"),
                "reach_id": data.get("reachId"),
                "name": data.get("name"),
                "description": data.get("description"),
                "rfc": data.get("rfc"),
                "wfo": data.get("wfo"),
                "state": data.get("state"),
                "county": data.get("county"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "time_zone": data.get("timeZone"),
                "datum": data.get("datum"),
                "pedts": data.get("pedts"),
                "status": data.get("status"),
            },

            "flood_categories": flood_categories,
            "metadata_verified": True,
            "threshold_verified": bool(flood_categories),
            "threshold_source": (
                "NOAA/NWS NWPS gauge metadata"
                if flood_categories
                else None
            ),
        },
        ensure_ascii=False,
    )


# ----------------------------------------------------------------
# NWPS Stage / Flow
# ----------------------------------------------------------------

@mcp.tool()
async def get_nwps_stageflow(
    gauge_id: str,
) -> str:
    nwps_api_url = os.environ.get(
        "NWPS_API_URL",
        "https://api.water.noaa.gov/nwps/v1",
    ).rstrip("/")

    timeout = float(
        os.environ.get(
            "MCP_HTTP_TIMEOUT",
            15.0,
        )
    )

    gauge_id = str(gauge_id).strip()

    if not gauge_id:
        return json.dumps(
            {
                "status": "error",
                "error": "gauge_id is empty.",
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    url = (
        f"{nwps_api_url}/gauges/"
        f"{gauge_id}/stageflow"
    )

    try:
        response = await _http_get_with_retry(
            url=url,
            params={},
            timeout=timeout,
        )
        data = response.json()

    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS stageflow request failed "
                    f"after {MAX_RETRIES} attempts: {exc}"
                ),
                "gauge_id": gauge_id,
                "metadata_verified": False,
                "action_required": (
                    "Verify the NWPS gauge LID and "
                    "stageflow service availability."
                ),
            },
            ensure_ascii=False,
        )

    if not isinstance(data, dict):
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS stageflow API returned an invalid "
                    "JSON object."
                ),
                "gauge_id": gauge_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )

    observed = data.get("observed")
    forecast = data.get("forecast")

    if observed is None and forecast is None:
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "NWPS stageflow response contains "
                    "neither observed nor forecast data."
                ),
                "gauge_id": gauge_id,
                "metadata_verified": False,
            },
            ensure_ascii=False,
        )
    
    return json.dumps(
        {
            "status": "ok",
            "source": "NOAA/NWS NWPS",
            "source_type": "nwps_stageflow",
            "gauge_id": gauge_id,

            "observed": observed,
            "forecast": forecast,

            "metadata_verified": True,
            "stageflow_verified": True,
        },
        ensure_ascii=False,
    )
# ----------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )