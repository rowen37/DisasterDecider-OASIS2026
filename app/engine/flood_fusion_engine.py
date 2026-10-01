"""Fusion Engine — canonical normalization and multi-source evidence fusion.

Engine layer: pure computation, no MCP access, no state mutation.
``fuse_flood_evidence`` accepts an optional ``log`` callback for the
caller's audit trail; without one, rejections are silently tolerated.
"""

from __future__ import annotations

import json
import math
import os
import re

from typing import Any

from ..utils import safe_float as _safe_float

from app.evidence_quality import assess_observation_quality

def normalize_fusion_observation(
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert heterogeneous MCP responses into canonical evidence items.

    Supported MCP response families:

    1. Single measurement:
       measurements = {
           "variable": "...",
           "value": ...,
           "unit": "..."
       }

    2. Multi-measurement:
       measurements = {
           "temperature": {
               "value": ...,
               "unit": "..."
           },
           ...
       }

    3. Alerts:
       alerts / warnings

    4. Forecasts:
       forecasts

    The MCP remains responsible for API-specific parsing.
    This function only normalizes the already-structured MCP output.
    """
    raw = observation["raw"]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Fusion source '{observation['source']}' "
            "did not return valid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Fusion source '{observation['source']}' "
            "returned an invalid structure."
        )

    if data.get("status") != "ok":
        raise RuntimeError(
            f"Fusion source '{observation['source']}' "
            "reported failure: "
            f"{data.get('error', 'unknown error')}"
        )

    source = str(
        observation.get(
            "source",
            data.get("source", "unknown"),
        )
    )

    source_type = str(
        observation.get(
            "source_type",
            data.get("source_type", "unknown"),
        )
    )

    # Prefer the source self-reported in the payload ("NWS", "CENSUS",
    # ...) over the config name ("weather", "river_level"), which is
    # only a display label; otherwise the authority tiers never match
    # and every source degrades to unknown/0.5.
    quality = assess_observation_quality(
        source=(
            data.get("source")
            if isinstance(data.get("source"), str) and data.get("source")
            else source
        ),
        observation=data,
        response_status=data.get("status", "error"),
        timestamp_field="timestamp",
        required_fields=(),
        max_age_minutes=float(
            os.environ.get(
                "FUSION_SOURCE_MAX_AGE_MINUTES",
                "180",
            )
        ),
        metadata_verified=False,
    )

    location_data = data.get("location")

    latitude = None
    longitude = None

    if isinstance(location_data, dict):
        latitude = _safe_float(
            location_data.get("latitude")
        )
        longitude = _safe_float(
            location_data.get("longitude")
        )

    base = {
        "source": source,
        "source_type": source_type,
        "timestamp": data.get("timestamp"),
        "latitude": latitude,
        "longitude": longitude,
        "quality_score": round(quality.quality_score, 6),
        "raw": data,
    }

    evidence_items: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Case 1 / 2: measurement-based MCP output
    # --------------------------------------------------------
    measurements = data.get("measurements")

    if isinstance(measurements, dict):

        # Single canonical measurement.
        if (
            "value" in measurements
            and "variable" in measurements
        ):
            value = _safe_float(
                measurements.get("value")
            )

            variable = measurements.get(
                "variable"
            )

            if value is not None and variable:
                evidence_items.append(
                    {
                        **base,
                        "evidence_type": "measurement",
                        "variable": str(variable),
                        "value": value,
                        "unit": measurements.get("unit"),
                    }
                )

        # Multi-variable measurement payload.
        else:
            for variable, measurement in measurements.items():

                if not isinstance(
                    measurement,
                    dict,
                ):
                    continue

                value = _safe_float(
                    measurement.get("value")
                )

                if value is None:
                    continue

                evidence_items.append(
                    {
                        **base,
                        "evidence_type": "measurement",
                        "variable": str(variable),
                        "value": value,
                        "unit": measurement.get("unit"),
                    }
                )

    # --------------------------------------------------------
    # Case 3: NWS alerts / flood warnings
    # --------------------------------------------------------
    alert_key = None

    if isinstance(
        data.get("alerts"),
        list,
    ):
        alert_key = "alerts"

    elif isinstance(
        data.get("warnings"),
        list,
    ):
        alert_key = "warnings"

    if alert_key is not None:

        alerts = data[alert_key]

        evidence_items.append(
            {
                **base,
                "evidence_type": "alert",
                "variable": None,
                "value": None,
                "unit": None,
                "alert_count": len(alerts),
                "alerts": alerts,
                "radius_km": _safe_float(
                    data.get("radius_km")
                ),
            }
        )

    # --------------------------------------------------------
    # Case 4: forecast
    # --------------------------------------------------------
    forecasts = data.get("forecasts")

    if isinstance(
        forecasts,
        list,
    ):
        evidence_items.append(
            {
                **base,
                "evidence_type": "forecast",
                "variable": None,
                "value": None,
                "unit": None,
                "forecast_count": len(
                    forecasts
                ),
                "forecasts": forecasts,
            }
        )

    # --------------------------------------------------------
    # Case 5: no-precipitation observation.
    #
    # NWS reports null (not 0.0) when nothing was recorded.
    # "No observed rainfall" is operationally meaningful and must
    # not be conflated with "precipitation data unavailable".
    # --------------------------------------------------------
    if data.get("observation_semantics") == "no_precipitation_observed":

        evidence_items.append(
            {
                **base,
                "evidence_type": "status_observation",
                "variable": "precipitation",
                "value": None,
                "unit": None,
                "observation_semantics": "no_precipitation_observed",
                "stations_checked": data.get(
                    "stations_checked"
                ),
                "note": data.get("note"),
            }
        )

    # --------------------------------------------------------
    # Case 6: flat water-level field — top-level or wrapped
    #         inside an "observation" dict.
    #
    # Two response shapes:
    #
    # Shape A (top-level):
    #   {"status": "ok", "water_level": 0.69, "unit": "ft"}
    #
    # Shape B (observation wrapper — flood_alert.py pattern):
    #   {"status": "ok", "observation": {"water_level": 0.69,
    #                                    "unit": "ft", ...}}
    # --------------------------------------------------------
    if not evidence_items:

        _FLAT_LEVEL_FIELDS: tuple[tuple[str, str], ...] = (
            ("water_level",  "water_level"),
            ("river_level",  "river_level"),
            ("stage",        "stage"),
            ("gauge_height", "gauge_height"),
            ("gage_height",  "gage_height"),
            ("level",        "level"),
            ("streamflow",   "streamflow"),
            ("discharge",    "discharge"),
        )

        # Scan the "observation" sub-dict if present, else the
        # top-level response dict.
        observation_wrapper = data.get("observation")
        scan_target: dict[str, Any] = (
            observation_wrapper
            if isinstance(observation_wrapper, dict)
            else data
        )

        # Use the wrapper's timestamp if available.
        flat_timestamp = (
            observation_wrapper.get("observation_time")
            or observation_wrapper.get("timestamp")
            if isinstance(observation_wrapper, dict)
            else data.get("timestamp")
        )

        for field_key, variable_name in _FLAT_LEVEL_FIELDS:
            flat_value = _safe_float(scan_target.get(field_key))
            if flat_value is not None:
                evidence_items.append(
                    {
                        **base,
                        "evidence_type": "measurement",
                        "variable": variable_name,
                        "value": flat_value,
                        "unit": scan_target.get("unit"),
                        # override timestamp with observation-level value
                        "timestamp": flat_timestamp or base.get("timestamp"),
                    }
                )
                break
                break

    # --------------------------------------------------------
    # Case 7: population exposure
    # --------------------------------------------------------
    pop = data.get("population")
    if isinstance(pop, dict) and not evidence_items:
        total = _safe_float(pop.get("total"))
        if total is not None:
            population_role = str(
                data.get("population_role")
                or (
                    "containing_tract_population"
                    if data.get("census_geography")
                    else "population_exposure"
                )
            )
            evidence_items.append({
                **base,
                "evidence_type": "measurement",
                "variable": population_role,
                "value": total,
                "unit": pop.get("unit", "people"),
                "semantics": data.get("note"),
            })

    # --------------------------------------------------------
    # Case 8: GEE flood extent (spatial extent).
    #
    # MCP response shape:
    # {
    #   "status": "ok",
    #   "spatial_extent": {
    #       "flooded_area_km2": 128.5,
    #       "geojson": {...},
    #       "confidence": 0.91
    #   }
    # }
    # --------------------------------------------------------
    spatial_extent = data.get("spatial_extent")

    if isinstance(spatial_extent, dict) and not evidence_items:
        flooded_area = _safe_float(
            spatial_extent.get("flooded_area_km2")
        )

        # The GEE MCP reports scene metadata under "observation"
        # (latest_post_scene / pre_scene_count / post_scene_count),
        # not inside spatial_extent. Propagate the acquisition time
        # so downstream layers can establish WHEN the extent was
        # observed; an extent without a time cannot be validated.
        gee_observation = data.get("observation")
        acquisition_time = None
        pre_scene_count = None
        post_scene_count = None

        if isinstance(gee_observation, dict):
            acquisition_time = gee_observation.get(
                "latest_post_scene"
            )
            pre_scene_count = gee_observation.get(
                "pre_scene_count"
            )
            post_scene_count = gee_observation.get(
                "post_scene_count"
            )

        if flooded_area is not None:
            evidence_items.append(
                {
                    **base,
                    "evidence_type":    "spatial_extent",
                    "variable":         "flooded_area_km2",
                    "value":            flooded_area,
                    "unit":             "km2",
                    "geojson":          spatial_extent.get("geojson"),
                    "confidence":       _safe_float(
                                            spatial_extent.get("confidence")
                                        ),
                    "acquisition_time": acquisition_time,
                    "pre_scene_count":  pre_scene_count,
                    "post_scene_count": post_scene_count,
                    # Override the source-wide timestamp with the
                    # actual satellite acquisition time.
                    "timestamp":        acquisition_time or base.get(
                                            "timestamp"
                                        ),
                }
            )

    return evidence_items

# ============================================================
# Evidence fusion
# ============================================================



def fuse_flood_evidence(
    observations: list[dict[str, Any]],
    log: Any = None,
) -> dict[str, Any]:
    """
    Fuse heterogeneous flood evidence.

    Numeric measurements are quality-score-weighted.
    quality_score is an evidence-quality index, not a
    probability.  It is used solely as a relative fusion
    weight so that higher-quality sources contribute more
    to the fused estimate.

    Alerts and forecasts are preserved as evidence but are
    not mathematically averaged with measurements.
    """

    if log is None:
        def log(*args: Any, **kwargs: Any) -> None:
            pass

    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for observation in observations:

        try:
            items = (
                normalize_fusion_observation(
                    observation
                )
            )

            normalized.extend(items)

        except Exception as exc:

            rejected.append(
                {
                    "source": observation.get(
                        "source"
                    ),
                    "tool": observation.get(
                        "tool"
                    ),
                    "error": str(exc),
                }
            )

            log(
                "fusion_observation_rejected",
                source=observation.get(
                    "source"
                ),
                tool=observation.get(
                    "tool"
                ),
                error=str(exc),
            )

    if not normalized:

        return {
            "status": "insufficient_evidence",
            "observation_count": 0,
            "source_count": 0,
            "observations": [],
            "fused_measurements": {},
            "alerts": [],
            "forecasts": [],
            "rejected_sources": rejected,
        }

    # --------------------------------------------------------
    # Group numeric measurements by variable.
    # --------------------------------------------------------
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    alert_evidence: list[
        dict[str, Any]
    ] = []

    forecast_evidence: list[
        dict[str, Any]
    ] = []

    for item in normalized:

        evidence_type = item.get(
            "evidence_type"
        )

        if evidence_type == "measurement":

            variable = item.get(
                "variable"
            )

            if variable:
                grouped.setdefault(
                    str(variable),
                    [],
                ).append(item)

        elif evidence_type == "alert":

            alert_evidence.append(item)

        elif evidence_type == "forecast":

            forecast_evidence.append(item)

    # --------------------------------------------------------
    # Quality-score-weighted numeric fusion.
    #
    # "total_fusion_weight" is the sum of per-source fusion
    # weights, not an evidence-quality score.
    # --------------------------------------------------------
    fused_measurements: dict[
        str,
        dict[str, Any],
    ] = {}

    for variable, items in grouped.items():

        weighted_values = []
        weights = []
        contributing_sources = []

        for item in items:

            quality_score = _safe_float(
                item.get("quality_score")
            )

            value = _safe_float(
                item.get("value")
            )

            if (
                quality_score is None
                or quality_score <= 0
                or value is None
            ):
                continue

            weighted_values.append(
                value * quality_score
            )

            weights.append(
                quality_score
            )

            contributing_sources.append(
                item["source"]
            )

        if not weights:
            continue

        total_weight = sum(weights)

        if total_weight <= 0:
            continue

        fused_value = (
            sum(weighted_values)
            / total_weight
        )

        units = {
            item.get("unit")
            for item in items
            if item.get("unit")
        }

        # Conflict isolation: when multi-source values for one
        # variable diverge beyond the threshold, keep the fused value
        # but flag it with the conflict ratio and per-source values,
        # so consumers (HITL / report) can see the disagreement.
        # The threshold is a project convention (no external standard).
        conflict_ratio = None
        conflict = False
        if len(contributing_sources) > 1:
            _vals = []
            for item in items:
                _q = _safe_float(item.get("quality_score"))
                _v = _safe_float(item.get("value"))
                if _q is not None and _q > 0 and _v is not None:
                    _vals.append(_v)
            if len(_vals) > 1:
                _spread = max(_vals) - min(_vals)
                _denom = max(max(_vals), 1e-9)
                conflict_ratio = round(_spread / _denom, 4)
                _threshold = float(
                    os.environ.get(
                        "FUSION_CONFLICT_THRESHOLD_RATIO", "0.5"
                    )
                )
                conflict = conflict_ratio > _threshold

        fused_measurements[variable] = {
            "value": round(fused_value, 4),
            "unit": (
                next(iter(units))
                if len(units) == 1
                else None
            ),
            "source_count": len(
                contributing_sources
            ),
            "sources": list(
                dict.fromkeys(
                    contributing_sources
                )
            ),
            "total_fusion_weight": total_weight,
            "conflict": conflict,
            **(
                {"conflict_ratio": conflict_ratio}
                if conflict_ratio is not None
                else {}
            ),
            **(
                {
                    "source_values": {
                        item["source"]: item.get("value")
                        for item in items
                        if _safe_float(item.get("quality_score")) is not None
                        and _safe_float(item.get("value")) is not None
                    }
                }
                if conflict
                else {}
            ),
        }
        if conflict and log is not None:
            log(
                "fusion_conflict_flagged",
                variable=variable,
                conflict_ratio=conflict_ratio,
                sources=list(
                    dict.fromkeys(contributing_sources)
                ),
            )

    # --------------------------------------------------------
    # Precipitation outlook substitute.
    #
    # When the observation network reports NO precipitation, the
    # NWS forecast precipitation probability (PoP) is surfaced as
    # a distinct fused variable so downstream layers can still
    # reason about rainfall risk. It is a forecast probability,
    # never blended into observed-rainfall measurements.
    # --------------------------------------------------------
    no_precip_observed = any(
        item.get("evidence_type") == "status_observation"
        and item.get("observation_semantics")
        == "no_precipitation_observed"
        for item in normalized
    )

    if no_precip_observed:

        pop_values: list[float] = []

        for fc_item in forecast_evidence:
            for period in fc_item.get("forecasts") or []:
                if not isinstance(period, dict):
                    continue
                pop = _safe_float(
                    period.get(
                        "probability_of_precipitation"
                    )
                )
                if pop is not None:
                    pop_values.append(pop)

        if pop_values:
            fused_measurements[
                "precipitation_probability_forecast"
            ] = {
                "value": round(max(pop_values), 2),
                "unit": "%",
                "source_count": 1,
                "sources": [
                    "NWS forecast PoP (substitute: no observed "
                    "precipitation available)"
                ],
                "total_fusion_weight": 1.0,
                "periods_covered": len(pop_values),
                "semantics": (
                    "Max forecast probability of precipitation, "
                    "not an observed rainfall amount."
                ),
            }

    successful_sources = list(
        dict.fromkeys(
            item["source"]
            for item in normalized
        )
    )

    # --------------------------------------------------------
    # Determine evidence sufficiency.
    #
    # At least one valid evidence item is enough to establish
    # that MCP retrieval succeeded. Numeric fusion is separately
    # reported through fused_measurements.
    # --------------------------------------------------------
    status = (
        "fused"
        if (
            fused_measurements
            or alert_evidence
            or forecast_evidence
        )
        else "insufficient_evidence"
    )

    return {
        "status": status,

        # Number of normalized evidence items.
        "observation_count": len(
            normalized
        ),

        # Number of distinct successful MCP sources.
        "source_count": len(
            successful_sources
        ),

        "sources": successful_sources,

        "observations": normalized,

        "fused_measurements": (
            fused_measurements
        ),

        "alerts": alert_evidence,

        "forecasts": forecast_evidence,

        "rejected_sources": rejected,
    }
