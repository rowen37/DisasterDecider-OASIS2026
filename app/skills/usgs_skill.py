"""UsgsSkill — atomic skill for USGS/NWPS station data acquisition.

The skill owns station-ID normalization, station metadata, NWPS gauge
metadata, stage/flow time series and the point observation.  It performs
no fusion, no risk math and no GIS work.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from typing import Any

from ..utils import safe_float as _safe_float
from ..models import Location


class UsgsSkill:

    name = "usgs_station_data"

    def __init__(
        self,
        state,
        verifier,
        mcp,
        logger,
    ):
        self.state = state
        self.verifier = verifier
        self.mcp = mcp
        self.logger = logger


    @staticmethod
    def _extract_station_id(text):
        """Extract an explicit USGS station ID; never accept arbitrary text."""
        if not text or not isinstance(text, str):
            return None
        patterns = [
            r"\bUSGS\s+(?:station\s+)?(?:ID\s*)?[:#]?\s*(\d{5,15})\b",
            r"\bstation\s+(?:ID\s*)?[:#]?\s*(\d{5,15})\b",
            r"\bstation\s*[:#]\s*(\d{5,15})\b",
            r"监测站\s*[:：#]?\s*(\d{5,15})\b",
            r"站点\s*[:：#]?\s*(\d{5,15})\b",
            r"\busing\s+(?:USGS\s+)?station\s+(\d{5,15})\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        stripped = text.strip()
        if re.fullmatch(r"\d{5,15}", stripped):
            return stripped
        return None

    @classmethod
    def _normalize_station_id(cls, station_id=None, raw_task=None):
        """Recover a station ID even if an upstream caller passed the full task."""
        if station_id:
            found = cls._extract_station_id(station_id)
            if found:
                return found
        if raw_task:
            found = cls._extract_station_id(raw_task)
            if found:
                return found
        return None

    @staticmethod
    def _parse_timestamp(value):
        if not value:
            return None
        try:
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    @staticmethod
    def _analyze_stageflow(
        stageflow: dict[str, Any],
        flood_categories: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract peaks, trend and threshold crossings from the NWPS
        time series. All thresholds come from flood_categories
        (authoritative NWPS values); no hardcoded constants.
        """
        observed = stageflow.get("observed") or {}
        forecast = stageflow.get("forecast") or {}

        obs_data = observed.get("data", []) if isinstance(observed, dict) else []
        fct_data = forecast.get("data", []) if isinstance(forecast, dict) else []

        def _extract_stages(data: list) -> list[float]:
            stages = []
            for point in data:
                if isinstance(point, dict):
                    v = _safe_float(point.get("primary") or point.get("stage"))
                    if v is not None:
                        try:
                            val = float(v)
                            if val != -999:   # filter sentinel value
                                stages.append(val)
                        except (TypeError, ValueError):
                            pass
            return stages

        obs_stages = _extract_stages(obs_data)
        fct_stages = _extract_stages(fct_data)

        peak_observed  = max(obs_stages) if obs_stages else None
        peak_forecast  = max(fct_stages) if fct_stages else None

        # Trend from the last 3 observations.
        trend = None
        if len(obs_stages) >= 3:
            recent = obs_stages[-3:]
            if recent[-1] > recent[0]:
                trend = "rising"
            elif recent[-1] < recent[0]:
                trend = "falling"
            else:
                trend = "steady"

        # Threshold crossings come solely from NWPS flood_categories.
        crossings: list[str] = []
        if peak_forecast is not None:
            for category in ("action", "minor", "moderate", "major"):
                cat_data = flood_categories.get(category)
                if not isinstance(cat_data, dict):
                    continue
                threshold = _safe_float(cat_data.get("stage"))
                if threshold is not None and peak_forecast >= threshold:
                    crossings.append(category)

        return {
            "peak_observed_stage":  peak_observed,
            "peak_forecast_stage":  peak_forecast,
            "trend":                trend,
            "forecast_crossings":   crossings,   # flood categories expected to be crossed
            "obs_point_count":      len(obs_stages),
            "fct_point_count":      len(fct_stages),
        }

    async def get_station_metadata(
        self,
        station_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve and validate authoritative USGS station metadata.

        Station coordinates are never inferred from the observation
        response: station identity and location must be explicitly
        verified through the metadata MCP.
        """
        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool": "get_station_metadata",
                "arguments": {
                    "station_id": station_id,
                },
                "purpose": "station_spatial_verification",
            },
        )

        raw = await self.mcp.call(
            "get_station_metadata",
            {
                "station_id": station_id,
            },
        )

        verification = self.verifier.validate_tool_text(raw)

        if not verification.passed:
            raise RuntimeError(
                verification.issues[0].message
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "USGS station metadata MCP did not return valid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "USGS station metadata MCP returned "
                "an invalid response structure."
            )

        if data.get("status") != "ok":
            raise RuntimeError(
                "USGS station metadata query failed: "
                + str(
                    data.get(
                        "error",
                        "unknown error",
                    )
                )
            )

        metadata = data.get("metadata")

        if not isinstance(metadata, dict):
            raise RuntimeError(
                "USGS station metadata response does not contain "
                "a valid metadata object."
            )

        required_fields = (
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "metadata_verified",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in metadata
        ]

        if missing_fields:
            raise RuntimeError(
                "USGS station metadata is missing required fields: "
                f"{missing_fields}"
            )

        latitude = _safe_float(
            metadata.get("latitude")
        )
        longitude = _safe_float(
            metadata.get("longitude")
        )

        if latitude is None or longitude is None:
            raise RuntimeError(
                "USGS station metadata does not contain "
                "valid numeric coordinates."
            )

        station_location = Location(
            name=str(
                metadata["station_name"]
            ),
            latitude=latitude,
            longitude=longitude,
            display_name=str(
                metadata.get("station_name")
            ),
            source="USGS Site Service via USGS Metadata MCP",
        )

        location_check = self.verifier.validate_location(
            station_location
        )

        if not location_check.passed:
            raise RuntimeError(
                location_check.issues[0].message
            )

        if metadata.get("metadata_verified") is not True:
            raise RuntimeError(
                "USGS station metadata is not marked as verified."
            )

        return {
            "station_id": str(
                metadata["station_id"]
            ),
            "station_name": str(
                metadata["station_name"]
            ),
            "latitude": latitude,
            "longitude": longitude,
            "state": metadata.get("state"),
            "county": metadata.get("county"),
            "site_type": metadata.get("site_type"),
            "drainage_area_sq_mi": metadata.get(
                "drainage_area_sq_mi"
            ),
            "contributing_drainage_area_sq_mi": metadata.get(
                "contributing_drainage_area_sq_mi"
            ),
            "metadata_verified": True,
            "source": metadata.get(
                "source",
                "USGS",
            ),
            "location": station_location,
        }

    async def get_nwps_gauge(
        self,
        station_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve authoritative NWPS gauge metadata using
        the USGS station ID.

        NWPS provides:
            - NWPS LID
            - USGS ID
            - reach ID
            - gauge metadata
            - current gauge status
            - flood category thresholds when configured
        """

        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool": "get_nwps_gauge",
                "arguments": {
                    "station_id": station_id,
                },
                "purpose": (
                    "nwps_gauge_metadata_and_flood_categories"
                ),
            },
        )

        raw = await self.mcp.call(
            "get_nwps_gauge",
            {
                "station_id": station_id,
            },
        )

        verification = self.verifier.validate_tool_text(
            raw
        )

        if not verification.passed:

            raise RuntimeError(
                verification.issues[0].message
            )

        try:

            data = json.loads(raw)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "NWPS gauge MCP did not return valid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):

            raise RuntimeError(
                "NWPS gauge MCP returned an invalid response structure."
            )

        if data.get("status") != "ok":

            raise RuntimeError(
                "NWPS gauge query failed: "
                + str(
                    data.get(
                        "error",
                        "unknown error",
                    )
                )
            )

        gauge = data.get(
            "gauge"
        )

        if not isinstance(
            gauge,
            dict,
        ):

            raise RuntimeError(
                "NWPS gauge response does not contain "
                "a valid gauge object."
            )

        gauge_id = gauge.get(
            "lid"
        )

        if not gauge_id:

            raise RuntimeError(
                "NWPS gauge response is missing "
                "the gauge LID."
            )

        usgs_id = str(
            gauge.get(
                "usgs_id",
                "",
            )
        ).strip()

        if usgs_id and usgs_id != station_id:

            raise RuntimeError(
                "NWPS gauge identity mismatch: "
                f"requested {station_id}, "
                f"received {usgs_id}."
            )

        flood_categories = data.get(
            "flood_categories",
            {},
        )

        if not isinstance(
            flood_categories,
            dict,
        ):
            flood_categories = {}

        return {
            "station_id": station_id,
            "gauge_id": str(
                gauge_id
            ),
            "gauge": gauge,
            "flood_categories": flood_categories,
            "threshold_verified": bool(
                data.get(
                    "threshold_verified",
                    False,
                )
            ),
            "threshold_source": data.get(
                "threshold_source"
            ),
            "metadata_verified": bool(
                data.get(
                    "metadata_verified",
                    False,
                )
            ),
            "source": data.get(
                "source",
                "NOAA/NWS NWPS",
            ),
        }

    async def get_nwps_stageflow(
        self,
        gauge_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve observed and forecast stage/flow data
        for an NWPS gauge.
        """

        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool": "get_nwps_stageflow",
                "arguments": {
                    "gauge_id": gauge_id,
                },
                "purpose": (
                    "nwps_observed_and_forecast_stageflow"
                ),
            },
        )

        raw = await self.mcp.call("get_nwps_stageflow",{"gauge_id": gauge_id,},)
        verification = self.verifier.validate_tool_text(raw)

        if not verification.passed:
            raise RuntimeError(
                verification.issues[0].message
            )

        try:
            data = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "NWPS stageflow MCP did not return valid JSON."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "NWPS stageflow MCP returned "
                "an invalid response structure."
            )

        if data.get("status") != "ok":
            raise RuntimeError(
                "NWPS stageflow query failed: "
                + str(
                    data.get(
                        "error",
                        "unknown error",
                    )
                )
            )

        return {
            "gauge_id": gauge_id,
            "observed": data.get("observed"),
            "forecast": data.get("forecast"),
            "stageflow_verified": bool(data.get("stageflow_verified",False,)),
            "metadata_verified": bool(data.get("metadata_verified",False,)),
            "source": data.get("source","NOAA/NWS NWPS",),
        }

    # ============================================================
    # Station observation
    # ============================================================

    async def get_station_observation(
        self,
        station_id: str,
        start_dt: str | None = None,
        end_dt: str | None = None,
    ) -> dict[str, Any]:
        """
        Station gauge observation. No time window = latest instantaneous
        value (real-time mode); with start_dt/end_dt = the PEAK value in
        the event window plus a window_summary hydrograph (historical
        replay -- the event's severity is its peak, not its window-end
        snapshot).
        """
        arguments: dict[str, Any] = {
            "station_id": station_id,
        }
        if start_dt:
            arguments["start_dt"] = start_dt
        if end_dt:
            arguments["end_dt"] = end_dt

        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool": "get_flood_observation",
                "arguments": arguments,
            },
        )

        raw = await self.mcp.call(
            "get_flood_observation",
            arguments,
        )

        verification = self.verifier.validate_tool_text(raw)

        if not verification.passed:
            raise RuntimeError(
                verification.issues[0].message
            )

        def _schema_hint(prefix: str) -> str:
            # A time-windowed query (start_dt/end_dt) sent to an MCP
            # server whose tool signature predates these parameters is
            # rejected by FastMCP validation as non-JSON error text;
            # append an actionable hint.
            if start_dt:
                return (
                    f"{prefix} Note: a time-windowed query (start_dt/"
                    "end_dt) was requested — if the MCP server process "
                    "predates the 2026-08-30 schema update it will "
                    "reject these arguments; restart the web server so "
                    "the MCP subprocesses respawn with the new tool "
                    "schema."
                )
            return prefix

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                _schema_hint(
                    "Flood observation MCP did not return valid JSON."
                )
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                _schema_hint(
                    "Flood observation MCP returned an invalid response structure."
                )
            )

        if data.get("status") != "ok":
            raise RuntimeError(
                _schema_hint(
                    "Flood observation MCP query failed: "
                    + str(
                        data.get(
                            "error",
                            "unknown error",
                        )
                    )
                )
            )

        observation = data.get("observation")

        if not isinstance(observation, dict):
            raise RuntimeError(
                "Flood observation MCP response does not contain "
                "a valid observation object."
            )

        required_fields = [
            "station_id",
            "water_level",
            "unit",
            "observation_time",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in observation
        ]

        if missing_fields:
            raise RuntimeError(
                "Flood observation MCP response is missing "
                f"required canonical fields: {missing_fields}"
            )

        return {
            "station_id": str(
                observation["station_id"]
            ),
            "water_level": float(
                observation["water_level"]
            ),
            "unit": str(
                observation["unit"]
            ),
            "observation_time": observation[
                "observation_time"
            ],
            # "window_peak" (historical replay: event stage = window
            # peak, full hydrograph in window_summary) or
            # "latest_instantaneous" (realtime). Older MCP builds omit
            # both; None means "unspecified snapshot semantics".
            "observation_semantics": observation.get(
                "observation_semantics"
            ),
            "window_summary": observation.get(
                "window_summary"
            ),
            "source": observation.get(
                "source",
                "unknown",
            ),
            "station_name": observation.get(
                "station_name"
            ),
            "latitude": observation.get(
                "latitude"
            ),
            "longitude": observation.get(
                "longitude"
            ),
            "flood_stage": observation.get(
                "flood_stage"
            ),
            "action_stage": observation.get(
                "action_stage"
            ),
            "metadata_verified": bool(
                observation.get(
                    "metadata_verified",
                    False,
                )
            ),
            "raw_metadata": observation.get(
                "metadata",
                {},
            ),
        }
