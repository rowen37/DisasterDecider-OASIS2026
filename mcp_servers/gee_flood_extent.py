import os
import sys
import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import ee
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ============================================================================
# GEE INITIALIZATION
# ============================================================================

_GEE_INITIALIZED = False


def _ensure_gee_initialized() -> None:
    """
    Lazy initialization of Google Earth Engine using service-account credentials.
    Prevents server crash on module import.
    """
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return

    try:
        key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY_PATH")
        service_account = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")

        if not key_path or not service_account:
            raise ValueError(
                "Missing environment variables: GEE_SERVICE_ACCOUNT_KEY_PATH "
                "or GEE_SERVICE_ACCOUNT_EMAIL"
            )

        credentials = ee.ServiceAccountCredentials(
            email=service_account,
            key_file=key_path,
        )

        ee.Initialize(credentials)
        _GEE_INITIALIZED = True

    except Exception as exc:
        print(f"GEE initialization error: {exc}", file=sys.stderr)
        raise RuntimeError(f"Failed to initialize Earth Engine: {exc}") from exc


# ============================================================================
# CONSTANTS / SAFE DEFAULTS
# ============================================================================

DEFAULT_BUFFER_KM = 10.0
DEFAULT_PRE_DAYS = 3
DEFAULT_POST_DAYS = 3

MIN_BUFFER_KM = 1.0
# This product is a city-level assessment.  Larger, regional analyses need a
# separate request with an explicitly supplied administrative boundary.
MAX_BUFFER_KM = 30.0

MIN_PRE_DAYS = 1
MAX_PRE_DAYS = 90

MIN_POST_DAYS = 3
MAX_POST_DAYS = 6

DEFAULT_THRESHOLD_DB = -3.0

SENTINEL_COLLECTION = "COPERNICUS/S1_GRD"


# ============================================================================
# INPUT NORMALIZATION & SAFE NUMERIC HELPERS
# ============================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Ensure returning a JSON-valid finite float."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"", "null", "none", "nil", "n/a", "na"}

    return False


def _parse_float(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if _is_null_like(value):
        raise ValueError(f"{name} must not be null")

    # schema declares a number; reject string input
    if isinstance(value, str):
        raise ValueError(
            f"{name} must be a number (not a string); received {value!r}"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number; received {value!r}")

    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number; received {value!r}")

    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {result}")

    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}; received {result}")

    return result


def _parse_int(
    value: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if _is_null_like(value):
        raise ValueError(f"{name} must not be null")

    # schema declares an integer; reject string input
    if isinstance(value, str):
        raise ValueError(
            f"{name} must be an integer (not a string); received {value!r}"
        )

    # floats must have no fractional part (30.0 ok, 30.5 not)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"{name} must be an integer; received {value!r}"
            )
        return _parse_int(int(value), name, minimum, maximum)

    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer; received {value!r}")

    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}; received {result}")

    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}; received {result}")

    return result


def _parse_bool(value: Any, name: str, default: bool) -> bool:
    if _is_null_like(value):
        return default

    # schema declares a boolean; reject string input
    if isinstance(value, str):
        raise ValueError(
            f"{name} must be a boolean (not a string); received {value!r}"
        )

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    raise ValueError(f"{name} must be boolean; received {value!r}")

def _parse_date(value: Any, name: str = "observation_date") -> str:
    if _is_null_like(value):
        raise ValueError(f"{name} must not be null")

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d")

    text = str(value).strip()

    try:
        parsed = datetime.strptime(text, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        pass

    iso_text = text
    if iso_text.endswith("Z"):
        iso_text = iso_text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"{name} must be a valid date such as YYYY-MM-DD; received {value!r}"
        )


def _normalize_arguments(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        raise ValueError(
            f"Tool arguments must be an object/dict; received {type(arguments).__name__}"
        )

    latitude = _parse_float(
        arguments.get("latitude"), "latitude", minimum=-90.0, maximum=90.0
    )
    longitude = _parse_float(
        arguments.get("longitude"), "longitude", minimum=-180.0, maximum=180.0
    )

    raw_date = arguments.get("observation_date")
    if _is_null_like(raw_date):
        raw_date = arguments.get("event_date")

    observation_date = _parse_date(raw_date, "observation_date")

    buffer_km = (
        DEFAULT_BUFFER_KM
        if _is_null_like(arguments.get("buffer_km"))
        else _parse_float(
            arguments.get("buffer_km"),
            "buffer_km",
            minimum=MIN_BUFFER_KM,
            maximum=MAX_BUFFER_KM,
        )
    )

    pre_days = (
        DEFAULT_PRE_DAYS
        if _is_null_like(arguments.get("pre_days"))
        else _parse_int(
            arguments.get("pre_days"),
            "pre_days",
            minimum=MIN_PRE_DAYS,
            maximum=MAX_PRE_DAYS,
        )
    )

    post_days = (
        DEFAULT_POST_DAYS
        if _is_null_like(arguments.get("post_days"))
        else _parse_int(
            arguments.get("post_days"),
            "post_days",
            minimum=MIN_POST_DAYS,
            maximum=MAX_POST_DAYS,
        )
    )

    return_geometry = _parse_bool(
        arguments.get("return_geometry"), "return_geometry", default=False
    )

    threshold_db = (
        float(
            os.environ.get(
                "GEE_FLOOD_DIFF_THRESHOLD_DB", str(DEFAULT_THRESHOLD_DB)
            )
        )
        if _is_null_like(arguments.get("threshold_db"))
        else _parse_float(
            arguments.get("threshold_db"),
            "threshold_db",
            minimum=-20.0,
            maximum=5.0,
        )
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "observation_date": observation_date,
        "buffer_km": buffer_km,
        "pre_days": pre_days,
        "post_days": post_days,
        "threshold_db": threshold_db,
        "return_geometry": return_geometry,
    }


# ============================================================================
# GEE ANALYSIS (RUNS IN WORKER THREAD)
# ============================================================================

def _extract_flood_extent_sync(request: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_gee_initialized()

    latitude = request["latitude"]
    longitude = request["longitude"]
    observation_date = request["observation_date"]
    buffer_km = request["buffer_km"]
    pre_days = request["pre_days"]
    post_days = request["post_days"]
    threshold_db = request["threshold_db"]
    return_geometry = request["return_geometry"]

    point = ee.Geometry.Point([longitude, latitude])
    aoi = point.buffer(buffer_km * 1000)

    observation_dt = datetime.strptime(observation_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    # --------------------------------------------------------------
    # Time-window computation (with automatic extension)
    # --------------------------------------------------------------
    now_utc = datetime.now(timezone.utc)
    window_adjusted = False

    # Initial observation window
    requested_post_start_dt = observation_dt
    requested_post_end_dt = observation_dt + timedelta(days=max(post_days, 1))

    # If the requested date is today or in the future, fall back to the
    # most recent valid window
    if requested_post_start_dt >= now_utc:
        post_end_dt = now_utc
        post_start_dt = now_utc - timedelta(days=max(post_days, 1))
        window_adjusted = True
    else:
        post_start_dt = requested_post_start_dt
        post_end_dt = min(requested_post_end_dt, now_utc)
        if post_end_dt != requested_post_end_dt:
            window_adjusted = True

    # Baseline window: immediately before the observation window
    pre_end_dt = post_start_dt
    pre_start_dt = pre_end_dt - timedelta(days=max(pre_days, 1))

    # ============= Automatic observation-window extension =============
    # If the observation window has no imagery, extend it backward in
    # time until imagery is found
    MAX_EXTEND_DAYS = 30
    extend_days = 0
    s1 = (
        ee.ImageCollection(SENTINEL_COLLECTION)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    # Try the initial window first
    post_start = post_start_dt.strftime("%Y-%m-%d")
    post_end = post_end_dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------------
    # GEE's filterDate() raises "Empty date ranges not supported" when
    # start == end at day granularity. This can happen even though
    # post_start_dt/post_end_dt differ by minutes - both strftime to
    # the SAME calendar-day string, producing a zero-width range for
    # GEE. This guard must run BEFORE the very first filterDate/
    # .getInfo() call below, since that call raises immediately and
    # skips every later extend-window / dedup safeguard in this
    # function entirely.
    # --------------------------------------------------------------
    if post_start == post_end:
        post_end_dt = post_end_dt + timedelta(days=1)
        post_end = post_end_dt.strftime("%Y-%m-%d")
        window_adjusted = True

    test_collection = s1.filterDate(post_start, post_end)
    post_count = int(test_collection.size().getInfo())

    # If the initial window has no imagery, extend backward: the window
    # end stays anchored (today) while the start grows day by day into
    # the past. An observation_date given as a UTC date can be a day
    # later than local time, so a fixed-width shift would miss the most
    # recent scenes (and get cut short by the pre-window overlap guard).
    while post_count == 0 and extend_days < MAX_EXTEND_DAYS:
        extend_days += 1
        new_post_start_dt = post_end_dt - timedelta(
            days=max(post_days, 1) + extend_days - 1
        )
        new_post_end_dt = post_end_dt
        new_post_start = new_post_start_dt.strftime("%Y-%m-%d")
        new_post_end = new_post_end_dt.strftime("%Y-%m-%d")
        if new_post_start >= new_post_end:
            continue
        test_collection = s1.filterDate(new_post_start, new_post_end)
        test_count = int(test_collection.size().getInfo())
        if test_count > 0:
            post_start_dt = new_post_start_dt
            post_end_dt = new_post_end_dt
            post_start = new_post_start
            post_end = new_post_end
            post_collection = test_collection
            post_count = test_count
            window_adjusted = True
            break

    # If still no imagery after extension, return an error
    if post_count == 0:
        return {
            "status": "error",
            "error_code": "NO_OBSERVATION_IMAGERY",
            "error": f"No Sentinel-1 scenes found in observation window after expanding {extend_days} days.",
            "request": request,
            "window_adjusted": window_adjusted,
        }

    # Recompute the baseline window (the observation window may have moved)
    pre_end_dt = post_start_dt
    pre_start_dt = pre_end_dt - timedelta(days=max(pre_days, 1))

    pre_start = pre_start_dt.strftime("%Y-%m-%d")
    pre_end = pre_end_dt.strftime("%Y-%m-%d")

    # Ensure pre/post window dates do not coincide (avoids empty GEE ranges)
    if pre_start == pre_end:
        pre_start = (pre_start_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    if post_start == post_end:
        post_end = (post_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    # Re-fetch both collections after the window may have moved
    pre_collection = s1.filterDate(pre_start, pre_end)
    post_collection = s1.filterDate(post_start, post_end)

    # Re-check pre_count (it may have changed after the move)
    pre_count = int(pre_collection.size().getInfo())

    # ============= Automatic baseline-window extension =============
    # Symmetric with the observation window: if the baseline window has
    # no imagery, keep its end (pre_end, adjacent to the observation
    # window) fixed and extend further into the past until imagery is
    # found or the extension cap is reached.
    pre_extend_days = 0
    while pre_count == 0 and pre_extend_days < MAX_EXTEND_DAYS:
        pre_extend_days += 1
        new_pre_start_dt = pre_start_dt - timedelta(days=pre_extend_days)
        new_pre_start = new_pre_start_dt.strftime("%Y-%m-%d")
        test_pre_collection = s1.filterDate(new_pre_start, pre_end)
        test_pre_count = int(test_pre_collection.size().getInfo())
        if test_pre_count > 0:
            pre_start_dt = new_pre_start_dt
            pre_start = new_pre_start
            pre_collection = test_pre_collection
            pre_count = test_pre_count
            window_adjusted = True
            break

    if pre_count == 0:
        return {
            "status": "error",
            "error_code": "NO_PRE_EVENT_IMAGERY",
            "error": (
                f"No Sentinel-1 scenes found in baseline window even after "
                f"expanding {pre_extend_days} days back (tried {pre_start} to {pre_end})."
            ),
            "request": request,
            "window_adjusted": window_adjusted,
        }

    pre_times = pre_collection.aggregate_array("system:time_start").getInfo()
    post_times = post_collection.aggregate_array("system:time_start").getInfo()

    def _convert_timestamp(ms: Any) -> Optional[str]:
        try:
            return datetime.fromtimestamp(
                float(ms) / 1000.0, tz=timezone.utc
            ).isoformat()
        except Exception:
            return None

    pre_acquisition_times = [
        x for x in [_convert_timestamp(t) for t in pre_times] if x is not None
    ]
    post_acquisition_times = [
        x for x in [_convert_timestamp(t) for t in post_times] if x is not None
    ]
    latest_post_scene = (
        max(post_acquisition_times) if post_acquisition_times else None
    )

    def _speckle_smoothing(image: ee.Image) -> ee.Image:
        return ee.Image(image).convolve(
            ee.Kernel.circle(radius=50, units="meters")
        )

    pre_mean = pre_collection.map(_speckle_smoothing).mean().clip(aoi)
    post_mean = post_collection.map(_speckle_smoothing).mean().clip(aoi)

    diff = post_mean.subtract(pre_mean)
    flooded_raw = diff.lt(threshold_db)

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
    permanent_water = jrc.gte(10)

    # Sentinel-1 difference identifies changed inundation pixels, not a river
    # model.  Exclude persistent water from the *area* but retain fine-scale
    # morphology (a 2-pixel square mode was producing blocky, regular edges).
    flooded = flooded_raw.where(permanent_water, 0).selfMask()
    flooded = flooded.focal_mode(radius=1, kernelType="circle", units="pixels")

    flooded_mask = flooded.rename("flooded_area")
    area_image = ee.Image.pixelArea().updateMask(flooded_mask).divide(1e6)

    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=10,
        maxPixels=10000000000,
        bestEffort=True,
    )

    stats_info = stats.getInfo() or {}
    raw_area = list(stats_info.values())[0] if stats_info else None
    # reduceRegion can return an empty result when fully masked or
    # degraded - "no data" must stay distinct from "no flooding", so the
    # area is null (downstream treats it as missing and logs a data gap)
    # rather than collapsed to 0.
    area_reduction_available = bool(stats_info) and raw_area is not None
    flooded_area_km2 = _safe_float(raw_area) if area_reduction_available else None

    geojson = None
    used_vector_scale = None
    if return_geometry:
        # Coarser vector scale + feature cap: reduceToVectors at 30 m over
        # a 30 km AOI exceeds the server-side 5000-element getInfo limit
        # and aborts. 300 m polygons (optionally dissolved) are sufficient
        # for city-level clipping / intersection downstream.
        for vector_scale, feature_cap in ((300, 4000), (600, 2000), (1200, 1000)):
            try:
                vectors = flooded_mask.reduceToVectors(
                    geometry=aoi,
                    scale=vector_scale,
                    geometryType="polygon",
                    eightConnected=False,
                    labelProperty="flooded",
                    maxPixels=10000000000,
                    bestEffort=True,
                ).filter(ee.Filter.eq("flooded", 1)).limit(feature_cap)
                geojson = vectors.getInfo()
                if geojson is not None:
                    used_vector_scale = vector_scale
                    break
            except Exception as exc:
                print(f"GEE geometry extraction failed (scale={vector_scale}): {exc}", file=sys.stderr)
                geojson = None

    return {
        "status": "ok",
        "source": "Google Earth Engine / Sentinel-1 SAR",
        "request": request,
        "observation": {
            "requested_observation_date": observation_date,
            "window_adjusted": window_adjusted,
            "latest_post_scene": latest_post_scene,
            "pre_scene_count": pre_count,
            "post_scene_count": post_count,
            "pre_acquisition_times": pre_acquisition_times,
            "post_acquisition_times": post_acquisition_times,
        },
        "spatial_extent": {
            "flooded_area_km2": (
                round(flooded_area_km2, 4)
                if flooded_area_km2 is not None
                else None
            ),
            "geojson": geojson,
        },
        "analysis": {
            "satellite": "Sentinel-1",
            "polarization": "VV",
            "orbit_pass": None,
            "pre_period": {"start": pre_start, "end_exclusive": pre_end},
            "observation_period": {"start": post_start, "end_exclusive": post_end},
            "buffer_km": buffer_km,
            "analysis_scale_m": 10,
            "threshold_method": "fixed",
            "threshold_db": threshold_db,
            "permanent_water_mask": "JRC GSW seasonality >= 10",
            "boundary_semantics": (
                "Detected Sentinel-1 backscatter-change footprint within the "
                "city analysis area; it is not a modeled floodplain or "
                "hydrologic water-network boundary."
            ),
            # Actual vectorization scale (whichever rung of the
            # 300/600/1200 m ladder succeeded); None when geometry was not
            # requested or extraction failed.
            "geometry_scale_m": used_vector_scale,
            # Collection filtered to IW+VV only, with no orbit-direction
            # filter (ascending/descending may mix), so orbit_pass is
            # reported as null instead of asserted DESCENDING.
            "orbit_filter": "none (IW + VV only; ascending/descending may mix)",
        },
        "data_quality": {
            "pre_scene_count": pre_count,
            "post_scene_count": post_count,
            "geometry_returned": geojson is not None,
            "area_reduction_available": area_reduction_available,
        },
    }

# ============================================================================
# MCP HANDLER
# ============================================================================

async def handle_get_flood_extent(
    arguments: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        request = _normalize_arguments(arguments)
    except Exception as exc:
        return {
            "status": "error",
            "error_code": "INVALID_PARAMETERS",
            "error": str(exc),
            "received_arguments": arguments,
        }

    try:
        # Hard timeout: the synchronous extraction can issue ~60 serial
        # getInfo calls (30+30 days of window extension); a hung GEE backend
        # must not stall the whole pipeline. On timeout, return a structured
        # error so downstream degrades as "satellite source unavailable"
        # (never folded to 0).
        gee_timeout = float(os.environ.get("GEE_EXTRACT_TIMEOUT", "150"))
        result = await asyncio.wait_for(
            asyncio.to_thread(_extract_flood_extent_sync, request),
            timeout=gee_timeout,
        )
        return result
    except asyncio.TimeoutError:
        print(
            f"get_flood_extent exceeded {gee_timeout}s wall clock "
            "(GEE backend hung or overloaded)",
            file=sys.stderr,
        )
        return {
            "status": "error",
            "error_code": "GEE_TIMEOUT",
            "error": (
                f"Flood-extent extraction exceeded the "
                f"{gee_timeout}s budget (Earth Engine backend "
                "unresponsive); treat the satellite source as "
                "unavailable for this run."
            ),
            "request": request,
        }
    except Exception as exc:
        print(f"get_flood_extent execution error: {exc}", file=sys.stderr)
        return {
            "status": "error",
            "error_code": "GEE_EXECUTION_ERROR",
            "error": str(exc),
            "request": request,
        }


# ============================================================================
# MCP SERVER
# ============================================================================

app = Server("gee-flood-extent-mcp")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_flood_extent",
            description=(
                "Retrieve flood inundation evidence from Sentinel-1 SAR imagery through Google Earth Engine."
            ),
            inputSchema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "latitude": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                    },
                    "longitude": {
                        "type": "number",
                        "minimum": -180,
                        "maximum": 180,
                    },
                    "observation_date": {
                        "type": "string",
                        "description": "Primary observation date, format: YYYY-MM-DD or ISO 8601.",
                    },
                    # Accepted as an alias by _normalize_arguments' fallback logic
                    "event_date": {
                        "type": "string",
                        "description": (
                            "Alias for observation_date. Used as fallback when "
                            "observation_date is not provided. Format: YYYY-MM-DD or ISO 8601."
                        ),
                    },
                    "buffer_km": {
                        "type": "number",
                        "minimum": MIN_BUFFER_KM,
                        "maximum": MAX_BUFFER_KM,
                        "default": DEFAULT_BUFFER_KM,
                        "description": "Radius of the analysis area in kilometers.",
                    },
                    "pre_days": {
                        "type": "integer",
                        "minimum": MIN_PRE_DAYS,
                        "maximum": MAX_PRE_DAYS,
                        "default": DEFAULT_PRE_DAYS,
                        "description": "Number of days before observation_date for the baseline window.",
                    },
                    # minimum/default are 1, matching the max(post_days, 1) behavior
                    "post_days": {
                        "type": "integer",
                        "minimum": MIN_POST_DAYS,
                        "maximum": MAX_POST_DAYS,
                        "default": DEFAULT_POST_DAYS,
                        "description": "Number of days after observation_date for the observation window. Minimum 1.",
                    },
                    "threshold_db": {
                        "type": "number",
                        "minimum": -20,
                        "maximum": 5,
                        "default": DEFAULT_THRESHOLD_DB,
                        "description": "Backscatter difference threshold (dB) for flood detection.",
                    },
                    "return_geometry": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether to return the flooded area as GeoJSON polygons.",
                    },
                },
                "required": ["latitude", "longitude", "observation_date"],
            },
        )
    ]

@app.call_tool()
async def call_tool_handler(
    name: str,
    arguments: Optional[dict],
):
    if name != "get_flood_extent":
        result = {
            "status": "error",
            "error_code": "UNKNOWN_TOOL",
            "error": f"Unknown tool: {name}",
        }
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    result = await handle_get_flood_extent(arguments)

    # Strict JSON serialization: the payload must not contain NaN/Infinity
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)

    return [
        TextContent(
            type="text",
            text=serialized,
        )
    ]


async def main():
    async with stdio_server() as streams:
        await app.run(
            *streams,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
