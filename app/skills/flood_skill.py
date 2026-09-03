"""Deterministic flood skill."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any

from app.evidence_quality import assess_observation_quality

from ..experiment import ExperimentLogger
from ..utils import safe_float as _safe_float, utc_now
from .fusion import (
    _load_json_env,
)

# Engine layer (pure computation) and atomic skills (pluggable data units)
from ..engine.allocation_engine import AllocationEngine
from ..engine.flood_fusion_engine import fuse_flood_evidence
from ..engine.geometry_engine import GeometryEngine
from ..engine.flood_risk_engine import RiskEngine
from .fusion_sources_skill import FusionSourcesSkill
from .geocode_skill import GeocodeSkill
from .resource_skill import ResourceSkill
from .svi_skill import SviSkill
from .usgs_skill import UsgsSkill
from ..hitl import AdaptiveHITL
from ..mcp_client import MCPManager
from ..social_good import (
    SocialGoodError,
    build_demand_records,
    compute_equity_gap,
    compute_equity_gap_or_none,
    compute_vulnerability_weighted_unmet_need,
    demand_impacts_for_plan,
    vulnerability_profile,
)
from .registry import register_skill
from ..models import (
    Evidence,
    Location,
    RunState,
    SkillResult,
    SpatialObject,
    ValidationIssue,
)
from ..verification import Verifier

# ============================================================================
# Time-alignment assessment (pure functions): strict in historical mode,
# relaxed with disclosure in realtime mode
# ============================================================================

# Static-layer sources (census / OSM facilities / road network) do not
# vary with the event time; mark them time_invariant instead of judging
# their offsets.
TIME_INVARIANT_SOURCE_TYPES = {
    "exposure",
    "infrastructure",
    "road",
    "resource_discovery",
    "social_vulnerability",
}


def _to_date(value: Any):
    """Parse an ISO string into a date; return None on failure."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).date()
    except ValueError:
        return None


def assess_time_alignment(
    fused_evidence: dict[str, Any],
    event_date: str | None,
    historical: bool,
    sar_max_lag_days: int = 6,
    sentinel_realtime_max_age_days: int = 6,
) -> dict[str, Any]:
    """
    Assess each source's timestamp offset against the event date.

    Historical mode (strict): hydrology beyond +/-1 day or SAR outside
    [-1, +sar_max_lag] is misaligned. Realtime mode (relaxed): SAR may
    look back up to 30 days (the window auto-extension ceiling), other
    rules as in historical mode; mismatches are disclosed, never dropped
    (current-only sources are skipped earlier at collection time).
    Forecast timestamps are in the future by design and marked
    future_by_design.
    """
    event = _to_date(f"{event_date}T00:00:00Z") if event_date else None
    items: list[dict[str, Any]] = []

    for obs in fused_evidence.get("observations", []):
        if not isinstance(obs, dict):
            continue
        source_type = str(obs.get("source_type", "unknown"))
        ts = obs.get("acquisition_time") or obs.get("timestamp")
        ts_date = _to_date(ts)

        if source_type in TIME_INVARIANT_SOURCE_TYPES:
            status = "time_invariant"
            offset = None
        elif ts_date is None or event is None:
            status = "unknown_timestamp"
            offset = None
        else:
            offset = (ts_date - event).days
            if source_type == "forecast":
                status = "future_by_design" if offset >= 0 else "misaligned"
            elif source_type == "satellite_sar":
                lo, hi = (
                    (-1, sar_max_lag_days)
                    if historical
                    else (-sentinel_realtime_max_age_days, 1)
                )
                status = "aligned" if lo <= offset <= hi else "misaligned"
            else:
                status = "aligned" if -1 <= offset <= 1 else "misaligned"

        items.append(
            {
                "source": obs.get("source"),
                "source_type": source_type,
                "timestamp": ts,
                "offset_days": offset,
                "status": status,
            }
        )

    mismatched = [i for i in items if i["status"] == "misaligned"]
    return {
        "mode": "historical" if historical else "realtime",
        "event_date": event_date,
        "policy": (
            "strict: hydrology within ±1 day, SAR within "
            f"[-1, +{sar_max_lag_days}] days"
            if historical
            else "relaxed: mismatches disclosed, not excluded"
        ),
        "items": items,
        "mismatched_sources": mismatched,
        "skipped_current_only_sources": None,  # filled in by the caller
    }


@register_skill(
    hazard="flood",
    keywords=(
        "flood", "flooding", "flooded", "inundation",
        "storm surge", "water level", "water rising",
        "overflow", "levee",
    ),
    weak_keywords=(
        "river", "stream", "rainfall", "stage", "gauge",
        "discharge", "streamflow",
    ),
    unsupported_hazard_words=(
        "wildfire", "fire", "blaze", "burn scar", "burn area",
        "earthquake", "quake", "seismic", "aftershock",
    ),
)
class FloodSkill:

    # Hazard-specific output rules injected by the synthesis layer
    # (agent.py); flood knowledge lives in this plugin, the shared
    # agent prompt carries none.
    OUTPUT_RULES = """
FLOOD-SPECIFIC RULES

If the evidence contains only:

station + water level + timestamp

you may state:

"The station recorded X ft at time T."

You may NOT state:

"The city is flooding."

You may NOT state:

"Flood risk is high."

You may NOT state:

"The station is above flood stage."

unless a verified threshold or model is present.

A water-level measurement is NOT automatically above flood stage;
flood severity may only be stated when an explicit, verified
threshold or validated model is present in the evidence.
"""

    @staticmethod
    def OUTPUT_VALIDATOR(output: str, packet: dict[str, Any]) -> None:
        """Flood-specific validation of forbidden LLM output claims
        (called by the synthesis layer).

        1) Unverified flood-stage claims;
        2) City-wide flood claims without areal evidence -- forbidden
           city names are derived from the evidence packet (target /
           place attributes) so the check generalizes across cities;
           the static Houston list is only a fallback.
        """

        # 1. Unverified flood-stage claims
        has_flood_stage = any(
            item.get("attributes", {}).get(
                "flood_stage_verified", False
            )
            for item in packet.get("evidence", [])
        )
        if not has_flood_stage:
            forbidden_patterns = [
                r"above flood stage",
                r"exceeded flood stage",
                r"flood stage exceeded",
                r"超过洪水警戒水位",
                r"超过洪水阶段",
            ]
            for pattern in forbidden_patterns:
                if re.search(pattern, output, flags=re.IGNORECASE):
                    raise RuntimeError(
                        "LLM output makes an unsupported "
                        "flood-stage claim."
                    )

        # 2. City-wide flood claims without areal evidence
        has_areal_evidence = any(
            item.get("attributes", {}).get("areal_coverage")
            in {"polygon", "inundation_extent", "citywide"}
            for item in packet.get("evidence", [])
        )
        if not has_areal_evidence:
            candidate_cities: set[str] = {"Houston", "休斯顿"}
            for item in packet.get("evidence", []):
                attributes = item.get("attributes", {})
                for key in ("target", "city", "place", "display_name"):
                    value = (
                        attributes.get(key)
                        if isinstance(attributes, dict)
                        else None
                    )
                    if isinstance(value, str) and value.strip():
                        token = value.strip().split(",")[0].split()[:1]
                        if (
                            token
                            and token[0].isalpha()
                            and len(token[0]) > 2
                        ):
                            candidate_cities.add(token[0])
                            break

            forbidden_city_claims = [
                f"{city} is flooding"
                for city in candidate_cities
            ] + [
                f"{city} is currently flooded"
                for city in candidate_cities
            ] + [
                f"{city} is experiencing flooding"
                for city in candidate_cities
            ]
            for phrase in forbidden_city_claims:
                if phrase.lower() in output.lower():
                    raise RuntimeError(
                        "LLM output makes an unsupported "
                        "city-wide flood claim."
                    )

    """
    Deterministic flood observation workflow.

    The current Flood MCP provides a USGS river-level point observation.
    It does not by itself establish flood severity, inundation extent, or
    city-wide impact.

    Critical parameter-safety rule:
        The Skill NEVER sends the natural-language user request to
        get_flood_observation. A valid numeric USGS station ID must be extracted
        first and is validated again immediately before the MCP call.
    """

    name = "flood_impact_assessment"

    def __init__(
        self,
        state: RunState,
        verifier: Verifier,
        hitl: AdaptiveHITL,
        mcp: MCPManager,
        logger: ExperimentLogger,
    ):
        self.state = state
        self.verifier = verifier
        self.hitl = hitl
        self.mcp = mcp
        self.logger = logger

        # -- Four-layer decoupling ---------------------------------
        # Engine layer: pure computation (geometry / risk indices /
        # Pareto), no I/O
        self.geometry = GeometryEngine()
        self.risk = RiskEngine()
        # Atomic skills: geocoding and USGS station fetch as pluggable units
        self.geocoder = GeocodeSkill(state, verifier, mcp, logger)
        self.usgs = UsgsSkill(state, verifier, mcp, logger)

        self.fusion_sources = _load_json_env(
            "FLOOD_FUSION_SOURCES_JSON",
            required=False,
        ) or []

        self.resource_tool = os.getenv(
            "RESOURCE_DISCOVERY_TOOL"
        )

        self.resource_tool_arguments = _load_json_env(
            "RESOURCE_DISCOVERY_ARGUMENTS_JSON",
            required=False,
        ) or {}

        self.allocation_objectives = _load_json_env(
            "RESOURCE_ALLOCATION_OBJECTIVES_JSON",
            required=False,
        ) or []

        self.allocation_weights = _load_json_env(
            "RESOURCE_OBJECTIVE_WEIGHTS_JSON",
            required=False,
        ) or {}

        self.max_plan_combinations = int(
            os.getenv(
                "RESOURCE_MAX_PLAN_COMBINATIONS",
                "10000",
            )
        )

        station_distance_policy = os.getenv(
            "FLOOD_STATION_MAX_DISTANCE_KM"
        )

        if station_distance_policy is None:
            raise RuntimeError(
                "FLOOD_STATION_MAX_DISTANCE_KM is not configured."
            )

        # Satellite flood analysis radius (km): GEE Sentinel-1 AOI buffer
        # size; the analysis area is later clipped to the city boundary.
        # Default buffer of 350 m = 300 m (coarsest GEE vectorization
        # step, geometry_scale_m, dominant edge-quantization term)
        # + 50 m (SAR speckle smoothing). The inherent 10 m grid sampling
        # of Sentinel-1 IW GRD is already covered by the quantization term.
        self.gis_flood_buffer_m = float(
            os.getenv("GIS_FLOOD_BUFFER_METERS", "350")
        )
        self.gis_search_radius_poi_m = float(
            os.getenv("GIS_SEARCH_RADIUS_POI_KM", "5")
        ) * 1000.0
        # Near-field building survey radius (distinct from the POI radius; separately configurable)
        self.gis_search_radius_buildings_m = float(
            os.getenv("GIS_SEARCH_RADIUS_BUILDINGS_M", "3000")
        )
        # Road-network search radius for detour routes
        self.gis_route_search_radius_km = float(
            os.getenv("GIS_ROUTE_SEARCH_RADIUS_KM", "10")
        )
        self.gis_route_avoid_flood = (
            os.getenv("GIS_ROUTE_AVOID_FLOOD", "true").lower() == "true"
        )
        # Optional event date (YYYY-MM-DD): when unset, fusion sources
        # fetch "today UTC"; when set, historical replay is enabled.
        self.event_date: str | None = None
        # Historical mode = event_date earlier than today (UTC).
        # Historical replay enforces strict time alignment: current-only
        # sources (active alerts / latest forecasts / latest observations)
        # are not collected, the gauge is queried in the event-day window,
        # and SAR acquisitions must fall near the event window. Realtime
        # mode is relaxed (mismatches disclosed, not blocking).
        self.historical_mode = False
        self.event_start_iso: str | None = None
        self.event_end_iso: str | None = None
        # Allowed SAR acquisition lag (days) relative to event_date:
        # strict [-1, +lag] in historical mode; realtime defaults to
        # [-6, +1] because water level is the primary daily evidence and
        # month-old scenes have no nowcast value. Sentinel-1A revisits
        # Texas about every 12 days, so some dates in a 6-day window
        # legitimately have no scene (spatial products degrade to
        # "only with fresh imagery" and say so).
        self.sar_max_lag_days = int(
            os.getenv("SAR_ACQUISITION_MAX_LAG_DAYS", "6")
        )
        self.sentinel_realtime_max_age_days = int(
            os.getenv("SENTINEL_REALTIME_MAX_AGE_DAYS", "6")
        )
        # Cross-dimension contradiction threshold: warn when the water
        # level is in bank (severity 0) while the satellite footprint
        # covers >= this share of the analysis area (no score change;
        # prompts manual imagery review).
        self.sar_stage_contradiction_share = float(
            os.getenv("SAR_STAGE_CONTRADICTION_SHARE", "0.05")
        )
        # Coverage-radius sensitivity sweep (km, comma-separated): makes
        # the equity metrics' dependence on R explicit; VWUN/equity gap
        # are recomputed per radius for the recommended plan.
        self.coverage_radius_sweep = [
            float(v)
            for v in os.getenv(
                "VULNERABILITY_COVERAGE_RADIUS_SWEEP", "2,5,10"
            ).split(",")
            if v.strip()
        ]

        self.flood_analysis_radius_km = float(
            os.getenv("GEE_FLOOD_ANALYSIS_RADIUS_KM", "10")
        )

        try:
            self.station_max_distance_km = float(
                station_distance_policy
            )
        except ValueError as exc:
            raise RuntimeError(
                "FLOOD_STATION_MAX_DISTANCE_KM must be numeric."
            ) from exc

        if (
            not math.isfinite(self.station_max_distance_km)
            or self.station_max_distance_km <= 0
        ):
            raise RuntimeError(
                "FLOOD_STATION_MAX_DISTANCE_KM must be "
                "a finite positive number."
            )

        warning_radius_policy = os.getenv(
            "NWS_WARNING_RADIUS_KM"
        )

        if warning_radius_policy is None:
            raise RuntimeError(
                "NWS_WARNING_RADIUS_KM is not configured."
            )

        try:
            self.nws_warning_radius_km = float(
                warning_radius_policy
            )
        except ValueError as exc:
            raise RuntimeError(
                "NWS_WARNING_RADIUS_KM must be numeric."
            ) from exc

        if (
            not math.isfinite(
                self.nws_warning_radius_km
            )
            or self.nws_warning_radius_km <= 0
        ):
            raise RuntimeError(
                "NWS_WARNING_RADIUS_KM must be "
                "a finite positive number."
            )
        
        self.svi_tool = os.getenv(
            "SVI_TOOL",
            "get_social_vulnerability",
        )

        self.svi_radius_km = float(
            os.environ["SVI_RADIUS_KM"]
        )

        self.svi_max_features = int(
            os.environ["SVI_MAX_FEATURES"]
        )

        self.vulnerability_weight = float(
            os.environ["VULNERABILITY_WEIGHT"]
        )

        self.equity_threshold = float(
            os.environ["EQUITY_HIGH_VULNERABILITY_THRESHOLD"]
        )

        # -- Four-layer decoupling assembly -------------------------
        # Engine: pure resource-allocation computation (objectives /
        # weights / caps injected from configuration)
        self.allocation = AllocationEngine(
            objectives=self.allocation_objectives,
            weights=self.allocation_weights,
            max_plan_combinations=self.max_plan_combinations,
            vulnerability_coverage_radius_km=float(
                os.getenv("VULNERABILITY_COVERAGE_RADIUS_KM", "10")
            ),
        )
        # Atomic skills: multi-source collection / SVI / resource discovery
        self.sources = FusionSourcesSkill(
            state, verifier, mcp, logger,
            warning_radius_km=self.nws_warning_radius_km,
            flood_analysis_radius_km=self.flood_analysis_radius_km,
            source_configs=self.fusion_sources,
        )
        self.svi = SviSkill(
            state, verifier, mcp, logger,
            svi_tool=self.svi_tool,
            svi_radius_km=self.svi_radius_km,
            svi_max_features=self.svi_max_features,
        )
        self.resources = ResourceSkill(
            state, verifier, mcp, logger,
            resource_tool=self.resource_tool,
            resource_tool_arguments=self.resource_tool_arguments,
        )

    # ============================================================
    # Geocoding
    # ============================================================

    # ============================================================
    # USGS / NWPS data acquisition (see usgs_skill.py)
    # ============================================================

    def _nwps_stageflow_fallback_observation(
        self,
        nwps_stageflow: dict[str, Any],
        nwps_gauge: dict[str, Any],
        station_id: str,
        usgs_error: Exception | None,
    ) -> dict[str, Any] | None:
        """Build a substitute observation from NWPS realtime stageflow
        when the USGS IV service fails.

        Only verified realtime stageflow is accepted; historical replay
        never collects stageflow, so this returns None there. The
        returned dict mirrors the field structure of
        UsgsSkill.get_station_observation exactly, so downstream code is
        unchanged; the substituted source must be disclosed via
        validation_issues.
        """
        if not nwps_stageflow.get("stageflow_verified"):
            return None
        observed = nwps_stageflow.get("observed")
        points: list = []
        units = "ft"
        if isinstance(observed, dict):
            points = observed.get("data") or []
            units = observed.get("primaryUnits") or "ft"
        elif isinstance(observed, list):
            points = observed
        stage_ft = None
        valid_time = None
        for point in reversed(points):
            if not isinstance(point, dict):
                continue
            value = _safe_float(point.get("primary") or point.get("stage"))
            if value is None or value == -999:   # -999 = NWPS missing-value sentinel
                continue
            stage_ft = value
            valid_time = point.get("validTime") or point.get("time")
            break
        if stage_ft is None:
            return None
        gauge = nwps_gauge.get("gauge") or {}
        return {
            "station_id": str(station_id),
            "water_level": float(stage_ft),
            "unit": str(units or "ft"),
            "observation_time": valid_time or utc_now().isoformat(),
            "source": "NOAA/NWS NWPS stageflow (USGS IV unavailable)",
            "station_name": gauge.get("name"),
            "latitude": gauge.get("latitude"),
            "longitude": gauge.get("longitude"),
            "flood_stage": None,
            "action_stage": None,
            "metadata_verified": bool(
                nwps_stageflow.get("metadata_verified", False)
            ),
            "raw_metadata": {
                "observation_verified": False,
                "source_verified": True,
                "fallback_from": "usgs_iv",
                "usgs_error": str(usgs_error) if usgs_error else None,
                "note": (
                    "Cross-source substitution: USGS IV failed; stage "
                    "taken from the NWPS observed stageflow feed."
                ),
            },
        }


    # ============================================================
    # Multi-source data acquisition
    # ============================================================

    def _build_allocation_evidence(
        self,
        optimization: dict[str, Any],
    ) -> Evidence:

        recommended = (
            optimization.get(
                "recommended_plan"
            )
        )

        frontier = (
            optimization.get(
                "pareto_frontier",
                [],
            )
        )

        if recommended is None:

            observation = (
                "No feasible resource allocation "
                "plan was identified from the "
                "available resource candidates."
            )

        else:

            observation = (
                f"Pareto-based resource optimization "
                f"identified "
                f"{len(frontier)} non-dominated "
                f"allocation plan(s). The selected "
                f"recommendation is plan "
                f"{recommended['plan_id']}."
            )

        return Evidence(
            evidence_id="resource_optimization",
            source=(
                "Configured resource discovery MCP "
                "and Pareto optimization"
            ),
            observation=observation,
            timestamp=utc_now().isoformat(),
            attributes={
                "optimization_status": (
                    optimization.get(
                        "status"
                    )
                ),
                "candidate_plan_count": (
                    optimization.get(
                        "candidate_plan_count",
                        0,
                    )
                ),
                "pareto_plan_count": (
                    optimization.get(
                        "pareto_plan_count",
                        0,
                    )
                ),
                "recommended_plan": recommended,
                "pareto_frontier": frontier,
                "svi_weight_applied": optimization.get(
                    "svi_weight_applied"
                ),
                "equity_ledger": optimization.get(
                    "equity_ledger"
                ),
                "objectives": (
                    self.allocation_objectives
                ),
            },
        )

    # ============================================================
    # Missing station result
    # ============================================================

    async def run(
        self,
        target: str,
        station_id: str | None = None,
        raw_task: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> SkillResult:
        if overrides:
            if 'station_max_distance_km' in overrides:
                self.station_max_distance_km = float(overrides['station_max_distance_km'])
            if 'nws_warning_radius_km' in overrides:
                self.nws_warning_radius_km = float(overrides['nws_warning_radius_km'])
            if 'svi_radius_km' in overrides:
                self.svi_radius_km = float(overrides['svi_radius_km'])
            if 'vulnerability_weight' in overrides:
                self.vulnerability_weight = float(overrides['vulnerability_weight'])
            if 'equity_threshold' in overrides:
                self.equity_threshold = float(overrides['equity_threshold'])
            if 'event_date' in overrides and overrides['event_date']:
                _ed = str(overrides['event_date']).strip()
                # Accept only YYYY-MM-DD; keeps arbitrary strings out of MCP arguments
                if len(_ed) == 10 and _ed[4] == "-" and _ed[7] == "-":
                    self.event_date = _ed
                    _today = utc_now().strftime("%Y-%m-%d")
                    self.historical_mode = _ed < _today
                    _start = datetime.fromisoformat(
                        f"{_ed}T00:00:00+00:00"
                    )
                    self.event_start_iso = _start.strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.event_end_iso = (
                        _start + timedelta(days=1)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    self.state.log(
                        "event_mode_selected",
                        event_date=_ed,
                        mode="historical" if self.historical_mode else "realtime",
                        time_alignment=(
                            "strict (current-only sources excluded, "
                            "windowed gauge query, SAR lag gate)"
                            if self.historical_mode
                            else "relaxed (mismatches disclosed, not blocking)"
                        ),
                    )

        self.state.hazard_type = "flood"
        self.state.log(
            "skill_started", skill=self.name, target=target,
            station_id=station_id, raw_task=raw_task,
        )

        # Non-fatal problems (GIS tool failures, unverified flood
        # extents, ...) recorded here and reported in the SkillResult
        # instead of being silently swallowed.
        validation_issues: list[ValidationIssue] = []

        try:
            verified_station_id = UsgsSkill._normalize_station_id(
                station_id=station_id,
                raw_task=raw_task,
            )

            _station_attempts = 0
            _max_station_attempts = 3
            while not verified_station_id:
                _station_attempts += 1
                if _station_attempts > _max_station_attempts:
                    # Bounded retry: in web mode each ask waits out the
                    # full HITL timeout budget, so an unbounded loop
                    # would hang the whole assessment.
                    raise RuntimeError(
                        "A USGS station ID is required but was not "
                        f"provided after {_max_station_attempts} "
                        "human-input attempts (or each attempt timed "
                        "out). Re-run with an explicit station ID, e.g. "
                        "'assess flooding near Friendswood station "
                        "08077600'."
                    )
                self.state.log("input_required", reason="missing_station_id", target=target)

                user_input = await self.hitl.ask_async(
                    reason=f"Missing USGS station ID for flood assessment of '{target}'.",
                    question="Please enter the USGS station ID (e.g., 01184000):",
                    proposed_value=None,
                    context={"target": target}
                )

                if user_input and isinstance(user_input, str) and user_input.strip():
                    verified_station_id = UsgsSkill._normalize_station_id(
                        station_id=user_input.strip(),
                        raw_task=None,
                    )
                    if verified_station_id:
                        print(f"✅ Station ID verified: {verified_station_id}")
                    else:
                        print(f"⚠️  Invalid station ID format. Please enter a valid numeric ID.")
                else:
                    print("⚠️  Input cannot be empty. Please try again.")

            self.state.log(
                "station_id_verified",
                skill=self.name,
                target=target,
                station_id=verified_station_id,
            )

            # -- Station metadata first (disambiguation anchor) --------
            # Station IDs are unique keys; place names are ambiguous
            # labels (Manhattan NY vs KS). Fetch the verified station
            # coordinates first and pass them as the geocoding bias, so
            # "Manhattan" resolves to the station's state. A metadata
            # failure is not fatal: degrade to "location not
            # independently verified" (fall back to observation
            # coordinates) and continue.
            station_metadata = None
            try:
                station_metadata = await self.usgs.get_station_metadata(
                    verified_station_id
                )
            except Exception as meta_err:
                self.state.log(
                    "station_metadata_unavailable",
                    station_id=verified_station_id,
                    error=str(meta_err),
                )
                validation_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="STATION_METADATA_UNAVAILABLE",
                        message=(
                            "USGS station metadata service failed "
                            f"({meta_err}); station location falls back "
                            "to the observation payload and is marked "
                            "as not independently verified."
                        ),
                        field="station_metadata",
                    )
                )

            geocode_bias = None
            if station_metadata is not None:
                geocode_bias = station_metadata.get("location")

            location = await self.geocoder.run(target, bias=geocode_bias)

            # -- City administrative boundary (best-effort) ------------
            # Used to clip the flood extent to the target city: the
            # satellite analysis area is a rectangular AOI buffer around
            # the target, so without clipping the extent is rectangular
            # and far exceeds the city. On failure, skip clipping.
            # Boundary disambiguation uses the same anchor as the
            # coordinates so both point at the same place.
            city_boundary_geojson: dict[str, Any] | None = None
            city_boundary_area_km2: float | None = None
            try:
                _boundary_args: dict[str, Any] = {"place_name": target}
                if geocode_bias is not None:
                    _boundary_args["bias_lat"] = geocode_bias.latitude
                    _boundary_args["bias_lon"] = geocode_bias.longitude
                boundary_raw = await self.mcp.call(
                    "geocode_boundary",
                    _boundary_args,
                )
                boundary_data = json.loads(boundary_raw)
                if (
                    isinstance(boundary_data, dict)
                    and boundary_data.get("status") == "ok"
                    and isinstance(boundary_data.get("geojson"), dict)
                ):
                    city_boundary_geojson = boundary_data["geojson"]
                    self.state.log(
                        "city_boundary_loaded",
                        target=target,
                        osm_type=boundary_data.get("osm_type"),
                    )
            except Exception as boundary_err:
                self.state.log(
                    "city_boundary_unavailable",
                    error=str(boundary_err),
                )

            # (Station metadata was fetched before geocoding; see the disambiguation block above.)

            # NWPS flood-category profile: many valid USGS stations have
            # no NWPS record (small basins / western sites), so a missing
            # record must not fail the whole assessment -- degrade to
            # "thresholds unverifiable" and continue.
            try:
                nwps_gauge = await self.usgs.get_nwps_gauge(
                    verified_station_id
                )
            except Exception as gauge_err:
                self.state.log(
                    "nwps_gauge_unavailable",
                    station_id=verified_station_id,
                    error=str(gauge_err),
                )
                validation_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="NWPS_GAUGE_UNAVAILABLE",
                        message=(
                            "No NWPS flood-category profile for this "
                            f"station ({gauge_err}); severity cannot be "
                            "interpreted from stage alone and forecast "
                            "threshold crossings were not evaluated."
                        ),
                        field="nwps_gauge",
                    )
                )
                nwps_gauge = {
                    "station_id": verified_station_id,
                    "gauge_id": None,
                    "gauge": {},
                    "flood_categories": {},
                    "threshold_verified": False,
                    "threshold_source": None,
                    "metadata_verified": False,
                    "source": "NOAA/NWS NWPS",
                }

            if nwps_gauge.get("gauge_id"):
                if self.historical_mode:
                    # NWPS stageflow provides only recent observations /
                    # forecasts and has no archive endpoint, so historical
                    # replay skips it (forecasts are meaningless for past
                    # dates) and records the skip explicitly.
                    self.state.log(
                        "nwps_stageflow_skipped_historical",
                        event_date=self.event_date,
                        reason="no_historical_archive_endpoint",
                    )
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="NWPS_STAGEFLOW_SKIPPED_HISTORICAL",
                            message=(
                                "Historical replay: NWPS observed/forecast "
                                "stageflow has no archive endpoint; "
                                "forecast threshold crossings were not "
                                "evaluated for the event window."
                            ),
                            field="nwps_stageflow",
                        )
                    )
                    nwps_stageflow = {
                        "observed": [],
                        "forecast": [],
                        "stageflow_verified": False,
                        "metadata_verified": False,
                    }
                else:
                    try:
                        nwps_stageflow = await self.usgs.get_nwps_stageflow(
                            nwps_gauge["gauge_id"]
                        )
                    except Exception as stageflow_err:
                        self.state.log(
                            "nwps_stageflow_unavailable",
                            gauge_id=nwps_gauge["gauge_id"],
                            error=str(stageflow_err),
                        )
                        validation_issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="NWPS_STAGEFLOW_UNAVAILABLE",
                                message=(
                                    "NWPS observed/forecast stageflow is "
                                    f"unavailable ({stageflow_err})."
                                ),
                                field="nwps_stageflow",
                            )
                        )
                        nwps_stageflow = {
                            "observed": [],
                            "forecast": [],
                            "stageflow_verified": False,
                            "metadata_verified": False,
                        }
            else:
                nwps_stageflow = {
                    "observed": [],
                    "forecast": [],
                    "stageflow_verified": False,
                    "metadata_verified": False,
                }

            # Primary observation source (USGS IV): retry once on
            # transient 503 / network errors; if it still fails and a
            # verified NWPS stageflow observation exists, fall back to it
            # (realtime mode only -- stageflow has no archive). The
            # fallback must be disclosed as a warning, never silent.
            observation = None
            obs_err: Exception | None = None
            for attempt in range(2):
                try:
                    observation = await self.usgs.get_station_observation(
                        verified_station_id,
                        # Historical replay: query the USGS IV archive in
                        # the event-day window so the stage aligns with
                        # satellite/census data; realtime mode returns the
                        # latest instantaneous value.
                        start_dt=(
                            self.event_start_iso if self.historical_mode else None
                        ),
                        end_dt=(self.event_end_iso if self.historical_mode else None),
                    )
                    break
                except Exception as exc:
                    obs_err = exc
                    if attempt == 0:
                        await asyncio.sleep(2.0)

            if observation is None:
                fallback = self._nwps_stageflow_fallback_observation(
                    nwps_stageflow, nwps_gauge, verified_station_id, obs_err
                )
                if fallback is not None:
                    observation = fallback
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="USGS_OBSERVATION_NWPS_FALLBACK",
                            message=(
                                "USGS water services produced no usable "
                                f"observation ({obs_err}); the stage reading "
                                "was substituted from the verified NWPS "
                                "stageflow feed. Re-verify against USGS "
                                "once the service recovers."
                            ),
                            field="usgs_observation",
                        )
                    )
                    self.state.log(
                        "usgs_observation_nwps_fallback",
                        station_id=verified_station_id,
                        usgs_error=str(obs_err),
                        stage_ft=fallback["water_level"],
                        observation_time=fallback["observation_time"],
                    )
                else:
                    # A missing core observation is truly unrecoverable,
                    # but the error message must be actionable
                    raise RuntimeError(
                        f"USGS station {verified_station_id} produced no usable "
                        f"river-stage observation ({obs_err}). The station may "
                        "not measure parameter 00065 (gage height), may be "
                        "seasonally dry, or the ID may be invalid — verify at "
                        "waterdata.usgs.gov before retrying."
                    ) from obs_err

            # Station location/name: prefer metadata; fall back to the observation's own coordinates
            if station_metadata is not None:
                station_location = station_metadata["location"]
                station_name = station_metadata["station_name"]
                station_location_verified = True
            else:
                _obs_lat = _safe_float(observation.get("latitude"))
                _obs_lon = _safe_float(observation.get("longitude"))
                station_name = (
                    observation.get("station_name") or verified_station_id
                )
                if _obs_lat is not None and _obs_lon is not None:
                    station_location = Location(
                        name=verified_station_id,
                        latitude=_obs_lat,
                        longitude=_obs_lon,
                        source=(
                            "USGS observation payload "
                            "(metadata service unavailable)"
                        ),
                    )
                    station_location_verified = bool(
                        observation.get("location_verified", False)
                    )
                else:
                    station_location = None
                    station_location_verified = False
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="STATION_LOCATION_UNVERIFIED",
                            message=(
                                "Station coordinates unavailable from "
                                "both metadata and observation; target-"
                                "station distance was not verified."
                            ),
                            field="station_location",
                        )
                    )

            station_spatial_verified = False

            if station_location is not None:
                spatial_verification = (
                    self.verifier.verify_station_target_distance(
                        target=location,
                        station=station_location,
                        max_distance_km=self.station_max_distance_km,
                    )
                )
                station_spatial_verified = bool(spatial_verification.passed)

                station_distance_km = self.verifier.haversine_km(
                    location.latitude,
                    location.longitude,
                    station_location.latitude,
                    station_location.longitude,
                )

                self.state.log(
                    "station_spatial_verification",
                    target=target,
                    station_id=verified_station_id,
                    station_name=station_name,
                    station_latitude=station_location.latitude,
                    station_longitude=station_location.longitude,
                    target_latitude=location.latitude,
                    target_longitude=location.longitude,
                    distance_km=station_distance_km,
                    max_allowed_distance_km=(
                        self.station_max_distance_km
                    ),
                    passed=spatial_verification.passed,
                    location_verified=station_location_verified,
                )

                # Distance exceeded: the station was explicitly chosen,
                # so continue with a warning instead of failing outright
                # (residual geocoding ambiguity or a multi-station
                # network); the audit trail records it and the resource
                # gate reads the real value.
                if not spatial_verification.passed:
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="STATION_TARGET_DISTANCE_EXCEEDED",
                            message=(
                                f"Station is {station_distance_km:.1f} km "
                                f"from the geocoded target (max "
                                f"{self.station_max_distance_km} km); "
                                "the pairing may be wrong — verify the "
                                "station serves this location."
                            ),
                            field="station_target_distance",
                        )
                    )
            else:
                station_distance_km = None

            observed_at = UsgsSkill._parse_timestamp(observation["observation_time"])
            retrieved_at = utc_now()
            freshness_seconds = None
            if observed_at is not None:
                freshness_seconds = max(
                    0.0,
                    (retrieved_at - observed_at.astimezone(timezone.utc)).total_seconds(),
                )

            evidence_attributes = {
                "station_id": observation["station_id"],
                "station_name": station_name,
                "water_level": observation["water_level"],
                "unit": observation["unit"],
                "observation_time": observation["observation_time"],
                "retrieved_at": retrieved_at.isoformat(),

                # Provenance
                "source_verified": True,
                "observation_verified": True,
                "metadata_verified": station_metadata is not None,

                # Station spatial provenance
                "station_latitude": (
                    station_location.latitude
                    if station_location is not None
                    else observation.get("latitude")
                ),
                "station_longitude": (
                    station_location.longitude
                    if station_location is not None
                    else observation.get("longitude")
                ),
                "target_latitude": location.latitude,
                "target_longitude": location.longitude,
                "station_target_distance_km": station_distance_km,
                "station_location_verified": station_location_verified,
                "target_station_spatial_relationship_verified": (
                    station_location is not None
                ),

                # Coverage semantics
                "areal_coverage": "point_observation",

                # These remain false until an authoritative threshold
                # source has actually been verified.
                "flood_stage_verified": (nwps_gauge["threshold_verified"]),
                "severity_interpretable": (nwps_gauge["threshold_verified"]),
                "nwps_gauge_id": (nwps_gauge["gauge_id"]),
                "nwps_gauge_verified": (nwps_gauge["metadata_verified"]),
                "nwps_stageflow_verified": (nwps_stageflow["stageflow_verified"]),
                "nwps_flood_categories": (nwps_gauge["flood_categories"]),
                "nwps_stageflow": {
                    "observed": nwps_stageflow["observed"],
                    "forecast": nwps_stageflow["forecast"],
                },
            }

            if freshness_seconds is not None:
                evidence_attributes["freshness_seconds"] = freshness_seconds

            # Historical replay: measure freshness against the end of the
            # event window so past-event observations are not judged
            # stale against the current wall clock.
            _quality_reference_now = (
                datetime.fromisoformat(self.event_end_iso.replace("Z", "+00:00"))
                if self.historical_mode and self.event_end_iso
                else None
            )
            obs_quality = assess_observation_quality(
                source="USGS Water Services via Flood Alert MCP",
                observation=observation,          # return value of _get_station_observation()
                response_status="ok",             # reaching this point implies status=ok
                timestamp_field="observation_time",
                required_fields=(
                    "station_id",
                    "water_level",
                    "unit",
                    "observation_time",
                ),
                max_age_minutes=float(
                    os.environ.get("USGS_REALTIME_MAX_AGE_MINUTES", "60")
                ),
                metadata_verified=(station_metadata is not None),
                reference_now=_quality_reference_now,
            )

            evidence = Evidence(
                evidence_id=f"usgs_station_{observation['station_id']}",
                source="USGS Water Services via Flood Alert MCP",
                observation=(
                    f"USGS monitoring station {observation['station_id']} reported "
                    f"{observation['water_level']} ft at "
                    f"{observation['observation_time']}."
                ),
                quality_score=obs_quality.quality_score,
                timestamp=observation["observation_time"],
                attributes=evidence_attributes,
            )

            def _should_escalate_to_hitl(
                self,
                fused_evidence: dict,
                nwps_gauge: dict,
                observation: dict,
            ) -> tuple[bool, str]:

                # Rule 1: multi-source value conflict detection
                measurements = fused_evidence.get("fused_measurements", {})
                for variable, fused in measurements.items():
                    items = [
                        item for item in fused_evidence.get("observations", [])
                        if item.get("variable") == variable
                    ]
                    if len(items) >= 2:
                        values = [i["value"] for i in items if i.get("value") is not None]
                        if values and (max(values) - min(values)) / max(max(values), 1e-9) > float(
                            os.environ.get("FUSION_CONFLICT_THRESHOLD_RATIO", "0.5")
                        ):
                            return True, (
                                f"Multi-source conflict on '{variable}': "
                                f"range [{min(values):.2f}, {max(values):.2f}]. "
                                "Human verification required before resource dispatch."
                            )

                # Rule 2: ask before continuing when flood thresholds are unverified
                if not nwps_gauge.get("threshold_verified"):
                    return True, (
                        "Flood category thresholds are not verified. "
                        "Severity cannot be interpreted from stage alone. "
                        "Proceed with resource optimization?"
                    )

                return False, ""

            # Initialize the HITL veto flag outside the fusion block so
            # the resource gate below (the human_denied_mobilization
            # read) still runs safely when fusion is not configured or
            # collect() raises.
            human_denied_mobilization = False

            fused_evidence = {"status": "not_configured"}
            if self.fusion_sources:
                try:
                    fusion_observations = await self.sources.collect(
                        target=target,
                        station_id=verified_station_id,
                        location=location,
                        event_date=self.event_date,
                        mode="historical" if self.historical_mode else "realtime",
                    )
                    fused_evidence = fuse_flood_evidence(fusion_observations, log=self.state.log)

                    # -- Time-alignment ledger: per-source offsets and
                    # mismatch disclosure. Historical mode already skips
                    # current-only sources at collection; re-check the
                    # timestamps of whatever was actually returned.
                    time_alignment = assess_time_alignment(
                        fused_evidence,
                        event_date=self.event_date
                        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        sentinel_realtime_max_age_days=(
                            self.sentinel_realtime_max_age_days
                        ),
                        historical=self.historical_mode,
                        sar_max_lag_days=self.sar_max_lag_days,
                    )
                    time_alignment["skipped_current_only_sources"] = (
                        list(self.sources.skipped_historical_sources)
                    )
                    evidence.attributes["time_alignment"] = time_alignment
                    if time_alignment["mismatched_sources"]:
                        self.state.log(
                            "time_alignment_mismatch",
                            mode=time_alignment["mode"],
                            mismatched=[
                                f"{i['source']}({i['offset_days']}d)"
                                for i in time_alignment["mismatched_sources"]
                            ],
                        )
                        validation_issues.append(
                            ValidationIssue(
                                severity=(
                                    "error" if self.historical_mode else "warning"
                                ),
                                code="TIME_ALIGNMENT_MISMATCH",
                                message=(
                                    "Evidence timestamps deviate from the "
                                    "event date: "
                                    + "; ".join(
                                        f"{i['source']} offset {i['offset_days']}d"
                                        for i in time_alignment[
                                            "mismatched_sources"
                                        ]
                                    )
                                    + (
                                        " (historical mode: strict policy)"
                                        if self.historical_mode
                                        else " (realtime mode: disclosed)"
                                    )
                                ),
                                field="time_alignment",
                            )
                        )

                    evidence.attributes["multi_source_fusion"] = {
                        "status": fused_evidence.get("status"),
                        "source_count": fused_evidence.get("source_count",0,),
                        "evidence_count": fused_evidence.get("observation_count",0,),
                        "fused_variables": list(fused_evidence.get("fused_measurements",{},).keys()),
                        "fused_values": fused_evidence.get("fused_measurements", {}),
                        "alert_evidence_count": len(fused_evidence.get("alerts",[],)),
                        "forecast_evidence_count": len(fused_evidence.get("forecasts",[],)),
                        "rejected_source_count": len(fused_evidence.get("rejected_sources",[],)),
                        "rejected_sources": fused_evidence.get("rejected_sources", []),
                        "observations": fused_evidence.get("observations", []),
                        "analysis_radius_km": self.flood_analysis_radius_km,
                    }

                    stageflow_analysis = UsgsSkill._analyze_stageflow(
                        stageflow=nwps_stageflow,
                        flood_categories=nwps_gauge["flood_categories"],
                    )
                    evidence_attributes["stageflow_analysis"] = stageflow_analysis

                    # A major crossing escalates straight to HITL and the
                    # answer must affect the flow: an explicit denial
                    # blocks resource optimization. timeout_value="no"
                    # makes "do not authorize mobilization" the safe
                    # default -- no answer never means go.
                    if "major" in stageflow_analysis.get("forecast_crossings", []):
                        major_answer = await self.hitl.ask_async(
                            reason="Forecast predicts major flood threshold crossing.",
                            question="Major flood forecast confirmed. Authorize full resource mobilization?",
                            proposed_value="yes",
                            context=stageflow_analysis,
                            timeout_value="no",
                        )
                        _major_denied = isinstance(major_answer, str) and (
                            major_answer.strip().lower() in ("no", "deny", "denied")
                        )
                        self.state.log(
                            "hitl_major_crossing_answered",
                            answer=(str(major_answer)[:80] if major_answer is not None else None),
                            mobilization_authorized=not _major_denied,
                        )
                        if _major_denied:
                            human_denied_mobilization = True

                    # Multi-source conflicts and unverified thresholds
                    # must escalate to a human before any downstream
                    # resource decision is made.
                    escalate, escalate_reason = _should_escalate_to_hitl(
                        self,
                        fused_evidence=fused_evidence,
                        nwps_gauge=nwps_gauge,
                        observation=observation,
                    )

                    if escalate:
                        self.state.log(
                            "hitl_escalation",
                            reason=escalate_reason,
                            target=target,
                            station_id=verified_station_id,
                        )
                        escalation_answer = await self.hitl.ask_async(
                            reason=escalate_reason,
                            question=(
                                "Human confirmation required. "
                                "Proceed with resource optimization?"
                            ),
                            proposed_value="proceed",
                            context={
                                "target": target,
                                "station_id": verified_station_id,
                                "fused_variables": list(
                                    fused_evidence.get(
                                        "fused_measurements",
                                        {},
                                    ).keys()
                                ),
                            },
                            # Safe default = do not proceed: on timeout or
                            # an absent operator, resource decisions must
                            # not auto-approve.
                            timeout_value="no",
                        )
                        # A non-empty answer other than proceed = human refusal
                        _escalation_denied = (
                            isinstance(escalation_answer, str)
                            and escalation_answer.strip()
                            and escalation_answer.strip().lower() not in ("proceed", "yes", "ok", "y")
                        )
                        self.state.log(
                            "hitl_escalation_answered",
                            answer=(str(escalation_answer)[:80] if escalation_answer is not None else None),
                            proceed_authorized=not _escalation_denied,
                        )
                        if _escalation_denied:
                            human_denied_mobilization = True

                    self.state.log(
                        "multi_source_fusion",
                        target=target,
                        source_count=fused_evidence.get(
                            "source_count",
                            0,
                        ),
                        evidence_count=fused_evidence.get(
                            "observation_count",
                            0,
                        ),
                        fused_measurements=fused_evidence.get(
                            "fused_measurements",
                            {},
                        ),
                        alert_evidence_count=len(
                            fused_evidence.get(
                                "alerts",
                                [],
                            )
                        ),
                        forecast_evidence_count=len(
                            fused_evidence.get(
                                "forecasts",
                                [],
                            )
                        ),
                    )

                except Exception as fusion_err:
                    self.state.log("fusion_warning", error=str(fusion_err))
                    fused_evidence = {"status": "fusion_failed", "error": str(fusion_err)}
                    evidence.attributes["multi_source_fusion"] = fused_evidence

            self.state.log(
                "flood_observation",
                target=location.name,
                station_id=observation["station_id"],
                water_level_ft=observation["water_level"],
                observation_time=observation["observation_time"],
                freshness_seconds=freshness_seconds,
                target_location_verified=True,
                station_location_verified=station_location_verified,
            )

            optimization = {"status": "not_configured"}
            evidence_list = [evidence]

            # =============================================================
            # Fused evidence enters the reporting layer: forecasts /
            # alerts / no-precipitation observations become standalone
            # Evidence items (not just counts).
            # =============================================================

            # ---- Forecasts -> standalone Evidence ----
            for fc_idx, fc_item in enumerate(
                fused_evidence.get("forecasts", [])
            ):
                if not isinstance(fc_item, dict):
                    continue

                fc_periods = fc_item.get("forecasts") or []
                if not isinstance(fc_periods, list):
                    fc_periods = []

                fc_first_ts = next(
                    (
                        p.get("timestamp")
                        for p in fc_periods
                        if isinstance(p, dict) and p.get("timestamp")
                    ),
                    None,
                )

                fc_summary = "; ".join(
                    f"{p.get('period')}: {p.get('short_forecast')}"
                    + (
                        f" (PoP {p.get('probability_of_precipitation')}%)"
                        if p.get("probability_of_precipitation") is not None
                        else ""
                    )
                    for p in fc_periods[:4]
                    if isinstance(p, dict)
                )

                evidence_list.append(
                    Evidence(
                        evidence_id=f"forecast_{fc_idx}",
                        source=str(fc_item.get("source", "NWS")),
                        observation=(
                            f"NWS forecast periods: {fc_summary}."
                            if fc_summary
                            else "NWS forecast data is available."
                        ),
                        timestamp=fc_first_ts,
                        quality_score=_safe_float(
                            fc_item.get("quality_score")
                        ),
                        attributes={
                            "source_type": "forecast",
                            "forecast_count": len(fc_periods),
                            "forecasts": fc_periods,
                        },
                    )
                )

            # ---- Alerts / warnings -> standalone Evidence ----
            for al_idx, al_item in enumerate(
                fused_evidence.get("alerts", [])
            ):
                if not isinstance(al_item, dict):
                    continue

                al_list = al_item.get("alerts") or []
                if not isinstance(al_list, list):
                    al_list = []

                # Compact summaries keep the LLM evidence packet small;
                # the raw NWS features can be very large.
                al_compact = [
                    {
                        "event": props.get("event"),
                        "severity": props.get("severity"),
                        "area": props.get("areaDesc"),
                        "headline": (props.get("headline") or "")[:200],
                        "expires": props.get("expires"),
                        "distance_km": a.get("_distance_to_center_km"),
                    }
                    for a in al_list[:10]
                    if isinstance(a, dict)
                    for props in (
                        a.get("properties") if isinstance(
                            a.get("properties"), dict
                        ) else a,
                    )
                ]

                evidence_list.append(
                    Evidence(
                        evidence_id=f"alert_{al_idx}",
                        source=str(al_item.get("source", "NWS")),
                        observation=(
                            f"{len(al_list)} active NWS alert(s) "
                            "within the warning radius."
                        ),
                        timestamp=al_item.get("timestamp"),
                        quality_score=_safe_float(
                            al_item.get("quality_score")
                        ),
                        attributes={
                            "source_type": "warning",
                            "alert_count": len(al_list),
                            "alerts": al_compact,
                        },
                    )
                )

            # ---- No-precipitation observations -> standalone Evidence ----
            for st_item in fused_evidence.get("observations", []):
                if not isinstance(st_item, dict):
                    continue
                if st_item.get("evidence_type") != "status_observation":
                    continue

                evidence_list.append(
                    Evidence(
                        evidence_id="precipitation_status",
                        source=str(st_item.get("source", "NWS")),
                        observation=(
                            "No measurable precipitation observed: "
                            + str(
                                st_item.get("note")
                                or "all nearby NWS stations reported "
                                "null precipitation values."
                            )
                            + (
                                f" ({st_item.get('stations_checked')} "
                                "stations checked)"
                                if st_item.get("stations_checked")
                                is not None
                                else ""
                            )
                        ),
                        timestamp=st_item.get("timestamp"),
                        quality_score=_safe_float(
                            st_item.get("quality_score")
                        ),
                        attributes={
                            "source_type": "precipitation",
                            "precipitation_status":
                                "no_precipitation_observed",
                            "stations_checked": st_item.get(
                                "stations_checked"
                            ),
                        },
                    )
                )

            for obs_item in fused_evidence.get("observations", []):

                if not isinstance(obs_item, dict):
                    continue

                if obs_item.get("evidence_type") != "spatial_extent":
                    continue

                geojson: dict[str, Any] | None = obs_item.get("geojson")  # type: ignore[assignment]

                if not isinstance(geojson, dict):
                    continue

                flooded_area    = _safe_float(obs_item.get("value"))
                gee_confidence  = _safe_float(obs_item.get("confidence"))
                gee_source: str = str(obs_item.get("source", "GEE Sentinel-1"))
                gee_ts:     str = str(obs_item.get("timestamp", ""))

                self.state.add_object(
                    SpatialObject(
                        object_id="flood_inundation_extent",
                        object_type="flood_extent",
                        geometry=geojson,
                        crs="EPSG:4326",
                        source=gee_source,
                        timestamp=gee_ts or utc_now().isoformat(),
                        attributes={
                            "flooded_area_km2": flooded_area,
                            "confidence":       gee_confidence,
                            "unit":             "km2",
                        },
                    )
                )
                break

            # =============================================================
            # GIS spatial analysis -- extracted directly from the GEE
            # response
            # =============================================================
            gis_results = {
                "flood_boundary_path": None,
                "flood_buffer_500m_path": None,
                "affected_buildings_path": None,
                "blocked_roads_path": None,
                "rescue_route_path": None,
                "poi_path": None,
                "map_path": None,
                "stats": {}
            }

            # ---- 1. Extract GEE results from fused_evidence ----
            gee_flooded_area_km2 = None
            gee_geojson = None
            gee_acquisition_time = None
            gee_post_scene_count = None

            for obs in fused_evidence.get("observations", []):
                if not isinstance(obs, dict):
                    continue
                if obs.get("evidence_type") == "spatial_extent":
                    gee_flooded_area_km2 = obs.get("value")
                    gee_geojson = obs.get("geojson")
                    gee_acquisition_time = obs.get(
                        "acquisition_time"
                    ) or obs.get("timestamp")
                    gee_post_scene_count = obs.get("post_scene_count")
                    break

            # ---- 1a. City clipping: restrict the satellite flood
            # polygons to the target city. The GEE analysis area is a
            # rectangular AOI, so an unclipped extent hugs the AOI edges
            # and far exceeds the city; after clipping, the extent
            # follows the city boundary and the area is recomputed.
            # ----
            city_area_km2: float | None = None
            flood_union_geom = None
            # City boundary geometry (available when clipping succeeds):
            # used downstream to align SVI/population denominators -- the
            # exposure numerator (flood intersected with census tracts,
            # clipped to the city) and the denominator (tract set) must
            # cover the same spatial extent.
            analysis_boundary_geom = None
            try:
                if (
                    city_boundary_geojson
                    and gee_geojson
                    and isinstance(gee_geojson, dict)
                    and gee_geojson.get("features")
                ):
                    clip = self.geometry.clip_flood_extent_to_city(
                        city_boundary_geojson,
                        gee_geojson,
                    )
                    if clip is not None:
                        gee_geojson = clip.clipped_geojson
                        gee_flooded_area_km2 = clip.flooded_area_km2
                        city_area_km2 = clip.city_area_km2
                        flood_union_geom = clip.flood_union_geom
                        analysis_boundary_geom = clip.city_boundary_geom


                    # -- Plausibility guard (extent cap) ----------------
                    # Stage in bank (<= action) with a footprint covering
                    # >= 50% of the city is physically contradictory for
                    # riverine flooding (false detection or stale water):
                    # reject the whole extent (skip all GIS products) and
                    # log a critical issue. Threshold is configurable.
                        _plausible_cap = _safe_float(
                            os.getenv("SAR_MAX_PLAUSIBLE_SHARE", "0.5")
                        )
                        _stage_now = _safe_float(
                            observation.get("water_level")
                        )
                        _action_now = _safe_float(
                            (nwps_gauge.get("flood_categories", {}).get(
                                "action", {}) or {}).get("stage")
                        )
                        if (
                            gee_flooded_area_km2 is not None
                            and city_area_km2
                            and city_area_km2 > 0
                            and _plausible_cap > 0
                            and (_stage_now is None or _stage_now <= _action_now)
                            and gee_flooded_area_km2 / city_area_km2
                            >= _plausible_cap
                        ):
                            _implausible_share = round(
                                gee_flooded_area_km2 / city_area_km2 * 100.0, 1
                            )
                            gee_geojson = {
                                "type": "FeatureCollection",
                                "features": [],
                            }
                            gee_flooded_area_km2 = None
                            flood_union_geom = None
                            validation_issues.append(
                                ValidationIssue(
                                    severity="critical",
                                    code="SAR_EXTENT_IMPLAUSIBLE",
                                    message=(
                                        f"Satellite footprint covers "
                                        f"{_implausible_share}% of the "
                                        f"analysis area while the gauge reads "
                                        f"{_stage_now} ft (action stage "
                                        f"{_action_now} ft) — physically "
                                        "contradictory for riverine flooding; "
                                        "extent rejected as implausible "
                                        "(change-detection false positives or "
                                        "stale water). Verify imagery manually."
                                    ),
                                    field="flood_inundation_extent",
                                )
                            )
                            self.state.log(
                                "sar_extent_implausible",
                                share=_implausible_share,
                                stage_ft=_stage_now,
                                action_ft=_action_now,
                                cap=_plausible_cap,
                            )

                        # Keep the stored flood-extent SpatialObject in
                        # sync with the clipped geometry used downstream.
                        existing_extent = self.state.spatial_objects.get(
                            "flood_inundation_extent"
                        )
                        if existing_extent is not None:
                            existing_extent.geometry = gee_geojson
                            existing_extent.attributes[
                                "flooded_area_km2"
                            ] = gee_flooded_area_km2
                            existing_extent.attributes[
                                "clipped_to_city_boundary"
                            ] = True
                            existing_extent.attributes[
                                "city_area_km2"
                            ] = round(city_area_km2, 3)

                        self.state.log(
                            "gis_extent_clipped_to_city",
                            original_area_km2=(
                                _safe_float(gee_flooded_area_km2)
                            ),
                            clipped_area_km2=gee_flooded_area_km2,
                            city_area_km2=round(city_area_km2, 3),
                            polygon_count=clip.polygon_count,
                        )
                    else:
                        # Zero in-city detection: water found in the 30 km
                        # buffer but no overlap with the city boundary --
                        # a known urban false-negative of single-pair SAR
                        # change detection. Do not map it as flooding;
                        # keep it as a context layer plus an explicit
                        # warning.
                        gee_geojson = {
                            "type": "FeatureCollection",
                            "features": [],
                        }
                        gee_flooded_area_km2 = 0.0
                        raw_gj = fused_evidence.get("observations", [])
                        for obs_item in raw_gj:
                            if (
                                isinstance(obs_item, dict)
                                and obs_item.get("evidence_type") == "spatial_extent"
                                and obs_item.get("geojson")
                                and isinstance(obs_item.get("geojson"), dict)
                                and (obs_item["geojson"].get("features") or [])
                            ):
                                import tempfile as _tf
                                _ctx = _tf.NamedTemporaryFile(
                                    mode="w", suffix=".geojson",
                                    delete=False,
                                )
                                json.dump(
                                    obs_item["geojson"], _ctx,
                                    ensure_ascii=False,
                                )
                                _ctx.close()
                                gis_results["context_extent_path"] = (
                                    _ctx.name
                                )
                                break
                        validation_issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="SAR_EXTENT_NO_INCITY_OVERLAP",
                                message=(
                                    "Satellite change detection found water "
                                    "in the 30 km buffer but ZERO overlap "
                                    "with the city boundary for this scene "
                                    "pair — known urban false-negative of "
                                    "single-pair SAR change detection. No "
                                    "in-city flood map can be produced; "
                                    "context layer shown for reference only."
                                ),
                                field="flood_inundation_extent",
                            )
                        )
                        self.state.log(
                            "gis_extent_clip_empty",
                            note="No flooded pixels inside city boundary",
                        )
            except Exception as clip_err:
                self.state.log(
                    "gis_extent_clip_failed",
                    error=str(clip_err),
                )

            # ---- 1b. Gate: the flood extent must carry a verifiable
            # satellite acquisition time to drive downstream GIS analysis
            # (buffer / flooded buildings / rescue routes); an undated
            # layer cannot be proven contemporaneous with the
            # observation.
            # ---- 1b'. Staleness gate: the acquisition must also fall
            # near the event window -- historical mode [-1, +sar_max_lag]
            # days (strict); realtime mode [-30, +1] days (the GEE window
            # auto-extension ceiling). A month-old scene must not drive a
            # "current" rescue route.
            # ----
            _sar_event_date = self.event_date or datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d")
            gee_lag_days = None
            gee_lag_ok = True
            _acq_date = _to_date(gee_acquisition_time)
            _event_d = _to_date(f"{_sar_event_date}T00:00:00Z")
            if _acq_date is not None and _event_d is not None:
                gee_lag_days = (_acq_date - _event_d).days
                _lo, _hi = (
                    (-1, self.sar_max_lag_days)
                    if self.historical_mode
                    else (-self.sentinel_realtime_max_age_days, 1)
                )
                gee_lag_ok = _lo <= gee_lag_days <= _hi
            if os.getenv("SAR_GATE_DEBUG"):
                print(f"[sar-gate] features={bool(gee_geojson and isinstance(gee_geojson, dict) and gee_geojson.get('features'))} "
                      f"acq={gee_acquisition_time!r} post_count={gee_post_scene_count!r} "
                      f"lag_ok={gee_lag_ok} lag={gee_lag_days!r} "
                      f"flooded={gee_flooded_area_km2!r} city={city_area_km2!r} "
                      f"union={'set' if flood_union_geom is not None else 'None'}",
                      file=sys.stderr)
            gee_extent_usable = bool(
                gee_geojson
                and isinstance(gee_geojson, dict)
                and gee_geojson.get("features")
                and gee_acquisition_time
                and str(gee_acquisition_time).strip().lower()
                not in ("", "none", "null")
                and (
                    gee_post_scene_count is None
                    or gee_post_scene_count > 0
                )
                and gee_lag_ok
            )

            if gee_lag_days is not None and not gee_lag_ok:
                validation_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="SAR_EXTENT_STALE",
                        message=(
                            f"SAR acquisition ({gee_acquisition_time}) is "
                            f"{gee_lag_days:+d} days from the event date "
                            f"({_sar_event_date}); allowed window is "
                            + (
                                f"[-1, +{self.sar_max_lag_days}]"
                                if self.historical_mode
                                else "[-30, +1]"
                            )
                            + " days. Downstream GIS analysis was skipped "
                            "(time-alignment gate)."
                        ),
                        field="flood_inundation_extent",
                    )
                )
                self.state.log(
                    "gis_extent_stale",
                    acquisition_time=gee_acquisition_time,
                    event_date=_sar_event_date,
                    lag_days=gee_lag_days,
                    mode="historical" if self.historical_mode else "realtime",
                    action="gis_analysis_skipped",
                )
                # A stale area must not enter the CDRI either (the hazard
                # component would still use a mismatched-date extent):
                # set it to None so hazard degrades explicitly to the
                # water-level ratio only, with the data gap recorded.
                # Invalidate flood_union_geom / gee_geojson as well so the
                # mismatched geometry cannot reach exposure population via
                # the union_features fallback.
                gee_flooded_area_km2 = None
                flood_union_geom = None
                gee_geojson = {"type": "FeatureCollection", "features": []}

            if (
                gee_geojson
                and isinstance(gee_geojson, dict)
                and not gee_extent_usable
            ):
                validation_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="UNVERIFIED_FLOOD_EXTENT",
                        message=(
                            "GEE flood extent has no verifiable "
                            "satellite acquisition time "
                            f"(acquisition_time={gee_acquisition_time!r}, "
                            f"post_scene_count={gee_post_scene_count!r}); "
                            "downstream GIS analysis was skipped."
                        ),
                        field="flood_inundation_extent",
                    )
                )
                self.state.log(
                    "gis_extent_unverified",
                    acquisition_time=gee_acquisition_time,
                    post_scene_count=gee_post_scene_count,
                    action="gis_analysis_skipped",
                )

            # Record GIS tool {"error": ...} payloads as validation
            # issues instead of skipping silently.
            def _record_gis_tool_error(
                payload: Any,
                tool: str,
            ) -> bool:
                if isinstance(payload, dict) and payload.get("error"):
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="GIS_TOOL_ERROR",
                            message=(
                                f"{tool} failed: "
                                f"{payload['error']}"
                            ),
                            field=tool,
                        )
                    )
                    return True
                return False

            # ---- 2. If GeoJSON is available, use it directly ----
            if gee_extent_usable:
                # Save via geopandas and force the CRS
                import geopandas as gpd
                gdf = gpd.GeoDataFrame.from_features(gee_geojson["features"])
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
                temp_geojson = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.geojson', delete=False
                )
                gdf.to_file(temp_geojson.name, driver="GeoJSON")
                temp_geojson.close()
                flood_boundary_path = temp_geojson.name

                gis_results["flood_boundary_path"] = flood_boundary_path
                gis_results["stats"]["flood_area_km2"] = gee_flooded_area_km2 or 0
                
                self.state.log("gis_gee_geojson", path=flood_boundary_path, area=gee_flooded_area_km2)
                
                # ---- 3. Run the remaining GIS analysis (buffer, intersects, routes) ----
                try:
                    # vec_buffer: flood buffer (500m)
                    buffer_result = await self.mcp.call("vec_buffer", {
                        "input_path": flood_boundary_path,
                        "buffer_meters": self.gis_flood_buffer_m,
                        "dissolve": True,
                    })
                    buffer_data = json.loads(buffer_result)
                    if (
                        not _record_gis_tool_error(buffer_data, "vec_buffer")
                        and isinstance(buffer_data, dict)
                    ):
                        flood_buffer_path = buffer_data.get("output_path")
                        if flood_buffer_path:
                            gis_results["flood_buffer_500m_path"] = flood_buffer_path

                    # vec_intersect: flooded buildings. poi_search_osm
                    # cannot query by polygon, so buildings are fetched by
                    # center point + radius, then intersected with the
                    # flood boundary. A large-radius building query can be
                    # very heavy downtown (tens of thousands of ways);
                    # budget 90 s because public Overpass mirrors commonly
                    # return slow 504s at peak (~30 s per failed attempt),
                    # leaving room to rotate to a healthy mirror instead
                    # of losing the whole building layer. The tool caps
                    # itself at an 85 s wall clock.
                    buildings_result = await self.mcp.call("poi_search_osm", {
                        "center_lat": location.latitude,
                        "center_lon": location.longitude,
                        "radius_m": self.gis_search_radius_buildings_m,
                        "amenity_types": "building",
                        "max_results": 500,
                    }, timeout=90.0, max_retries=1)
                    buildings_data = json.loads(buildings_result)
                    if (
                        not _record_gis_tool_error(
                            buildings_data, "poi_search_osm"
                        )
                        and isinstance(buildings_data, dict)
                    ):
                        buildings_path = buildings_data.get("output_path")
                        if buildings_path:
                            # Intersect the flood extent with the buildings
                            intersect_result = await self.mcp.call("vec_intersect", {
                                "layer1_path": flood_boundary_path,
                                "layer2_path": buildings_path,
                                "keep_fields": "name,amenity",
                            })
                            intersect_data = json.loads(intersect_result)
                            if (
                                not _record_gis_tool_error(
                                    intersect_data, "vec_intersect"
                                )
                                and isinstance(intersect_data, dict)
                            ):
                                affected_buildings_path = intersect_data.get("output_path")
                                if affected_buildings_path:
                                    gis_results["affected_buildings_path"] = affected_buildings_path
                                    gis_results["stats"]["affected_buildings"] = intersect_data.get("intersected_count", 0)

                    # vec_shortest_path: rescue route (the POI layer is
                    # allowed to fail fast; a small hospital,shelter query
                    # usually takes seconds, 45 s leaves rotation room).
                    # POI search follows the flood footprint when present
                    # (centroid + bounding radius, capped at 30 km)
                    # instead of a small fixed radius around the target --
                    # otherwise POIs cluster at the center of a city-wide
                    # footprint.
                    poi_center_lat = location.latitude
                    poi_center_lon = location.longitude
                    poi_radius_m = self.gis_search_radius_poi_m
                    if flood_union_geom is not None:
                        _linked = self.geometry.flood_center_radius(
                            flood_union_geom
                        )
                        if _linked:
                            poi_center_lat = _linked[0]
                            poi_center_lon = _linked[1]
                            poi_radius_m = min(
                                max(_linked[2], 1000.0), 30000.0
                            )
                            gis_results["stats"]["poi_search_scope"] = (
                                "flood_footprint"
                            )
                            gis_results["stats"]["poi_search_center"] = [
                                round(poi_center_lat, 5),
                                round(poi_center_lon, 5),
                            ]
                            gis_results["stats"]["poi_search_radius_km"] = (
                                round(poi_radius_m / 1000.0, 2)
                            )
                    poi_result = await self.mcp.call("poi_search_osm", {
                        "center_lat": poi_center_lat,
                        "center_lon": poi_center_lon,
                        "radius_m": poi_radius_m,
                        "amenity_types": "hospital,shelter",
                        "max_results": 10,
                    }, timeout=45.0, max_retries=1)
                    poi_data = json.loads(poi_result)
                    if (
                        not _record_gis_tool_error(
                            poi_data, "poi_search_osm"
                        )
                        and isinstance(poi_data, dict)
                    ):
                        poi_path = poi_data.get("output_path")
                        if poi_path:
                            gis_results["poi_path"] = poi_path
                            affected_poi_result = await self.mcp.call("vec_intersect", {
                                "layer1_path": flood_boundary_path,
                                "layer2_path": poi_path,
                                "keep_fields": "name,amenity,osm_id",
                            })
                            affected_poi_data = json.loads(affected_poi_result)
                            if (
                                not _record_gis_tool_error(affected_poi_data, "vec_intersect_facilities")
                                and isinstance(affected_poi_data, dict)
                            ):
                                gis_results["stats"]["affected_facilities"] = affected_poi_data.get("intersected_count", 0)
                            features = poi_data.get("feature_list", [])
                            if features and isinstance(features, list) and len(features) > 0:
                                dest = features[0]
                                if isinstance(dest, dict):
                                    route_args = {
                                        "origin_lat": location.latitude,
                                        "origin_lon": location.longitude,
                                        "dest_lat": dest.get("lat"),
                                        "dest_lon": dest.get("lon"),
                                        "travel_mode": "drive",
                                        "search_radius_km": self.gis_route_search_radius_km,
                                    }
                                    if self.gis_route_avoid_flood:
                                        route_args["avoid_polygon_path"] = flood_boundary_path
                                    route_result = await self.mcp.call("vec_shortest_path", route_args)
                                    route_data = json.loads(route_result)
                                    # If flooding cuts the path, fall back
                                    # to a non-avoiding route as an
                                    # accessibility reference (explicitly
                                    # marked as not avoiding flood)
                                    route_err = (
                                        route_data.get("error")
                                        if isinstance(route_data, dict)
                                        else None
                                    )
                                    if route_err and "no_path" in str(route_err).lower():
                                        route_result = await self.mcp.call("vec_shortest_path", {
                                            "origin_lat": location.latitude,
                                            "origin_lon": location.longitude,
                                            "dest_lat": dest.get("lat"),
                                            "dest_lon": dest.get("lon"),
                                            "travel_mode": "drive",
                                            "search_radius_km": self.gis_route_search_radius_km,
                                        })
                                        route_data = json.loads(route_result)
                                        if isinstance(route_data, dict) and not route_data.get("error"):
                                            gis_results["stats"]["route_avoids_flood"] = False
                                    if (
                                        not _record_gis_tool_error(
                                            route_data, "vec_shortest_path"
                                        )
                                        and isinstance(route_data, dict)
                                    ):
                                        route_path = route_data.get("output_path")
                                        if route_path:
                                            gis_results["rescue_route_path"] = route_path
                                            gis_results["stats"]["route_length_km"] = route_data.get("path_length_km", 0)
                                            gis_results["stats"]["travel_time_min"] = route_data.get("travel_time_min", 0)
                                            # Flooded road segments = edges removed from the routing graph
                                            removed = _safe_float(
                                                route_data.get("flooded_edges_removed")
                                            )
                                            if removed is not None:
                                                gis_results["stats"]["affected_roads"] = int(removed)

                    if not gis_results.get("poi_path"):
                        # Disclose search failure / empty results -- a
                        # silently missing layer reads as "no facilities
                        # available".
                        validation_issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="POI_SEARCH_FAILED",
                                message=(
                                    "Critical-facility (POI) search failed "
                                    "or returned nothing — the facilities "
                                    "layer is unavailable this run."
                                ),
                                field="gis_poi",
                            )
                        )
                        self.state.log("poi_search_failed")

                    # vis_flood_map: render the map. Path arguments must
                    # be empty strings, not None -- FastMCP rejects null
                    # values for str parameters, which would abort the
                    # whole GIS section with a json parse error.
                    map_result = await self.mcp.call("vis_flood_map", {
                        "center_lat": location.latitude,
                        "center_lon": location.longitude,
                        "flood_boundary_path": gis_results.get("flood_boundary_path") or "",
                        "affected_buildings_path": gis_results.get("affected_buildings_path") or "",
                        "rescue_route_path": gis_results.get("rescue_route_path") or "",
                        "poi_path": gis_results.get("poi_path") or "",
                        "station_lat": (
                            station_location.latitude
                            if station_location is not None
                            else 0.0
                        ),
                        "station_lon": (
                            station_location.longitude
                            if station_location is not None
                            else 0.0
                        ),
                        "station_name": station_name,
                        "target_lat": location.latitude,
                        "target_lon": location.longitude,
                        "target_name": location.name,
                        "zoom_start": 14,
                    })
                    try:
                        map_data = json.loads(map_result)
                    except (json.JSONDecodeError, TypeError):
                        map_data = {
                            "error": (
                                "vis_flood_map returned non-JSON "
                                f"output: {str(map_result)[:200]}"
                            )
                        }
                    if (
                        not _record_gis_tool_error(map_data, "vis_flood_map")
                        and isinstance(map_data, dict)
                    ):
                        gis_results["map_path"] = map_data.get("output_path")

                except Exception as gis_err:
                    self.state.log("gis_analysis_error", error=str(gis_err))
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="GIS_ANALYSIS_ERROR",
                            message=(
                                "GIS spatial analysis aborted: "
                                f"{gis_err}"
                            ),
                            field="gis_analysis",
                        )
                    )

            # ---- 4. Extent failed the gate (or no GeoJSON): record the area only ----
            elif gee_flooded_area_km2 is not None:
                gis_results["stats"]["flood_area_km2"] = gee_flooded_area_km2
                self.state.log("gis_gee_area_only", area=gee_flooded_area_km2,
                               note="GEE extent missing, incomplete, or unverified; "
                                    "GIS analysis limited to the area value")

            # Store GIS results in state
            self.state.gis_results = gis_results
            evidence.attributes["gis_stats"] = gis_results["stats"]

            resource_gate_passed, resource_gate_reasons = (
                AllocationEngine.optimization_allowed(
                    fused_evidence=fused_evidence,
                    station_spatial_verified=station_spatial_verified,
                )
            )
            if human_denied_mobilization:
                resource_gate_passed = False
                resource_gate_reasons = list(resource_gate_reasons) + [
                    "Human operator declined resource mobilization at HITL checkpoint."
                ]

            social_vulnerability = None

            try:
                social_vulnerability = await self.svi.collect(
                    location
                )

                # -- Align the denominator scope ------------------------
                # The SVI search is a 25 km circle around the city center
                # and includes out-of-city tracts, while the flood extent
                # is clipped to the city boundary. An in-city numerator
                # with an out-of-city denominator systematically dilutes
                # population_factor and vulnerability_coverage: when city
                # boundary geometry exists, filter tracts to the analysis
                # extent.
                if analysis_boundary_geom is not None:
                    _tracts_raw = social_vulnerability.get("tracts") or []
                    _tracts_kept, _tracts_excluded = (
                        self.geometry.filter_tracts_to_boundary(
                            _tracts_raw, analysis_boundary_geom
                        )
                    )
                    if _tracts_excluded:
                        try:
                            social_vulnerability["profile"] = (
                                vulnerability_profile(_tracts_kept)
                            )
                            social_vulnerability["tracts"] = _tracts_kept
                            social_vulnerability["tract_scope"] = (
                                "city_boundary"
                            )
                            self.state.log(
                                "svi_tracts_filtered_to_city",
                                raw_tract_count=len(_tracts_raw),
                                kept_tract_count=len(_tracts_kept),
                                excluded_tract_count=_tracts_excluded,
                            )
                        except SocialGoodError:
                            # No valid tracts after filtering (boundary and
                            # search circle disjoint, etc.): keep the
                            # original set and disclose it.
                            social_vulnerability["tract_scope"] = (
                                "radius_25km_fallback_after_empty_filter"
                            )
                    else:
                        social_vulnerability["tract_scope"] = "city_boundary"

                vulnerability_evidence = (
                    self.svi.build_evidence(
                        social_vulnerability
                    )
                )

                evidence_list.append(
                    vulnerability_evidence
                )

                self.state.log(
                    "social_vulnerability",
                    target=target,
                    tract_count=(
                        social_vulnerability["profile"]["tract_count"]
                    ),
                    population_weighted_svi=(
                        social_vulnerability["profile"][
                            "population_weighted_svi"
                        ]
                    ),
                    coverage_completeness=(
                        social_vulnerability["profile"][
                            "coverage_completeness"
                        ]
                    ),
                )

            except Exception as svi_err:
                self.state.log(
                    "social_vulnerability_warning",
                    error=str(svi_err),
                )

                social_vulnerability = {
                    "status": "unavailable",
                    "error": str(svi_err),
                }

            # -- Affected population -----------------------------------
            # Flood polygons intersected with SVI census tracts: a tract
            # counts as affected when its centroid falls inside the flood
            # polygon, and its whole population is included. Documented
            # approximation (tract population is not uniformly
            # distributed).
            try:
                # Without a city boundary (clip not run), build the union
                # geometry from the raw GEE geojson so the affected
                # population estimate still works.
                if flood_union_geom is None:
                    flood_union_geom = self.geometry.union_features(gee_geojson)

                if (
                    flood_union_geom is not None
                    and isinstance(social_vulnerability, dict)
                    and social_vulnerability.get("status") == "ok"
                ):
                    affected_pop, affected_tracts, pop_method = (
                        self.geometry.estimate_affected_population(
                            flood_union_geom,
                            social_vulnerability,
                        )
                    )
                    gis_results["stats"]["affected_population"] = int(affected_pop)
                    gis_results["stats"]["affected_population_tracts"] = affected_tracts
                    gis_results["stats"]["affected_population_method"] = pop_method
                    # Per-tract exposure (area-weighted) feeds the equity ledger (VWUN/equity gap)
                    gis_results["tract_exposure"] = self.geometry.tract_exposure(
                        flood_union_geom,
                        social_vulnerability,
                    )
                    # Method bracket: the area-weighted and centroid
                    # whole-tract estimates bound the point estimate,
                    # making the uniform-distribution uncertainty
                    # explicit.
                    _bracket = self.geometry.affected_population_bracket(
                        gis_results["tract_exposure"]
                    )
                    if _bracket is not None:
                        gis_results["stats"][
                            "affected_population_interval"
                        ] = _bracket["interval"]
                    self.state.log(
                        "affected_population_estimated",
                        affected_population=int(affected_pop),
                        affected_tracts=affected_tracts,
                        method=pop_method,
                    )
            except Exception as pop_err:
                self.state.log(
                    "affected_population_estimation_failed",
                    error=str(pop_err),
                )

            # -- HITL parameter confirmation checkpoint -----------------
            # Before decision-critical steps (resource optimization),
            # confirm key parameters with the operator: empty input /
            # Use Default continues with defaults; a JSON answer may
            # override vulnerability_weight / equity_threshold /
            # station_max_distance_km. In web mode the checkpoint
            # auto-accepts defaults (equivalent to "leave empty") to
            # avoid blocking the browser for up to 300 s; CLI mode keeps
            # manual confirmation.
            _checkpoint_enabled = (
                os.getenv("HITL_PARAMETER_CHECKPOINT", "true").lower() == "true"
                and not getattr(self.hitl, "_web_mode", False)
            )
            if not _checkpoint_enabled and os.getenv("HITL_PARAMETER_CHECKPOINT", "true").lower() == "true":
                self.state.log(
                    "hitl_parameter_checkpoint_auto_accepted",
                    mode="web",
                    params={
                        "vulnerability_weight": self.vulnerability_weight,
                        "equity_threshold": self.equity_threshold,
                        "station_max_distance_km": self.station_max_distance_km,
                    },
                )
            # Adaptive HITL: when primary evidence quality is at or above
            # AUTO_APPROVE_CONFIDENCE, auto-approve the default
            # parameters and only interrupt a human when evidence is weak.
            _checkpoint_confidence = getattr(evidence, "quality_score", None)
            _checkpoint_ask = _checkpoint_enabled and self.hitl.should_intervene(
                confidence=_checkpoint_confidence
            )
            if _checkpoint_enabled and not _checkpoint_ask:
                self.state.log(
                    "hitl_parameter_checkpoint_auto_approved",
                    confidence=_checkpoint_confidence,
                    auto_approve_confidence=self.hitl.auto_approve_confidence,
                )
            if _checkpoint_ask:
                checkpoint_params = {
                    "vulnerability_weight": self.vulnerability_weight,
                    "equity_threshold": self.equity_threshold,
                    "station_max_distance_km": self.station_max_distance_km,
                }
                checkpoint_answer = await self.hitl.ask_async(
                    reason=(
                        "Parameter confirmation before resource "
                        "optimization and final reporting."
                    ),
                    question=(
                        "Continue with default analysis parameters? "
                        "Leave empty to accept, or enter JSON overrides "
                        '(e.g. {"vulnerability_weight": 1.5}).'
                    ),
                    proposed_value=json.dumps(checkpoint_params),
                    context={
                        "target": target,
                        "water_level_ft": observation.get("water_level"),
                        "flooded_area_km2": gee_flooded_area_km2,
                        # The frontend renders sliders from this (instead
                        # of bare text inputs) to explore VWUN vs lambda;
                        # submission still goes through the JSON channel
                        "adjustable_parameters": [
                            {
                                "key": "vulnerability_weight",
                                "label": "λ vulnerability weight (VWUN)",
                                "min": 0.0,
                                "max": 3.0,
                                "step": 0.1,
                                "default": self.vulnerability_weight,
                            },
                            {
                                "key": "equity_threshold",
                                "label": "High-SVI grouping threshold",
                                "min": 0.5,
                                "max": 0.99,
                                "step": 0.01,
                                "default": self.equity_threshold,
                            },
                        ],
                    },
                )
                self.state.log(
                    "hitl_parameter_checkpoint",
                    accepted_default=(
                        checkpoint_answer is None
                        or str(checkpoint_answer).strip() == ""
                        or str(checkpoint_answer) == json.dumps(checkpoint_params)
                    ),
                    raw_answer=str(checkpoint_answer)[:200],
                )
                if checkpoint_answer and isinstance(checkpoint_answer, str):
                    try:
                        overrides_from_human = json.loads(checkpoint_answer)
                        if isinstance(overrides_from_human, dict):
                            for key in (
                                "vulnerability_weight",
                                "equity_threshold",
                                "station_max_distance_km",
                            ):
                                if key in overrides_from_human:
                                    setattr(
                                        self,
                                        key,
                                        float(overrides_from_human[key]),
                                    )
                            self.state.log(
                                "hitl_parameters_overridden",
                                applied=overrides_from_human,
                            )
                    except json.JSONDecodeError:
                        validation_issues.append(
                            ValidationIssue(
                                severity="warning",
                                code="HITL_OVERRIDE_UNPARSEABLE",
                                message=(
                                    "HITL checkpoint answer was not valid "
                                    "JSON; default parameters were kept."
                                ),
                                field="hitl_checkpoint",
                            )
                        )

            # Transparent, bounded decision-support indices.  They are
            # normalized scores (not probabilities or forecasts) and retain
            # their component values for audit in the primary USGS evidence.
            # Computation lives in the Risk Engine (Engine layer); the Skill
            # only supplies inputs and attaches the result.
            evidence.attributes["decision_indices"] = self.risk.compute_decision_indices(
                water_level=observation.get("water_level"),
                action_stage=_safe_float(
                    (nwps_gauge.get("flood_categories", {}).get("action", {}) or {}).get("stage")
                ),
                major_stage=_safe_float(
                    (nwps_gauge.get("flood_categories", {}).get("major", {}) or {}).get("stage")
                ),
                flooded_area_km2=gee_flooded_area_km2,
                city_area_km2=city_area_km2,
                fallback_analysis_radius_km=self.flood_analysis_radius_km,
                social_vulnerability=social_vulnerability,
                gis_stats=gis_results["stats"],
                fused_measurements=fused_evidence.get("fused_measurements", {}),
                fusion_sources=self.fusion_sources,
                observations=fused_evidence.get("observations", []),
            )

            # -- Cross-dimension contradiction warning ------------------
            # Stage in bank (severity 0) while the satellite footprint
            # covers >= 5% of the analysis area: either change-detection
            # false positives (wet soil / vegetation / stale water) or
            # real flooding in a gauge blind spot. The indices do not
            # score up (water level stays primary), but the contradiction
            # must be disclosed for manual imagery review.
            _di = evidence.attributes.get("decision_indices") or {}
            _di_inputs = _di.get("inputs") or {}
            _water_sev = _safe_float(_di_inputs.get("water_severity")) or 0.0
            _extent_share = _safe_float(_di_inputs.get("extent_severity")) or 0.0
            if (
                _water_sev <= 0.0
                and _extent_share >= self.sar_stage_contradiction_share
            ):
                validation_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="SAR_EXTENT_STAGE_CONTRADICTION",
                        message=(
                            f"Satellite change detection reports "
                            f"{_extent_share:.0%} of the analysis area as "
                            "open water while the gauge reads at/below "
                            "action stage — possible false positives, "
                            "standing water from earlier rain, or "
                            "gauge-blind flooding; verify the imagery "
                            "layer before acting."
                        ),
                        field="flood_inundation_extent",
                    )
                )
                self.state.log(
                    "sar_stage_contradiction",
                    extent_share=round(_extent_share, 4),
                    water_severity=round(_water_sev, 4),
                )
                # -- Second false-positive guard: building density in the
                # footprint. Real urban flooding typically shows at least
                # a few buildings/km2; very low density with stage in bank
                # flags a suspected false positive.
                _flood_km2 = _safe_float(
                    gis_results.get("stats", {}).get("flood_area_km2")
                )
                _aff_bld = _safe_float(
                    gis_results.get("stats", {}).get("affected_buildings")
                )
                _suspect_density = _safe_float(os.getenv(
                    "SAR_SUSPECT_DENSITY_PER_KM2", "5"
                ))
                if (
                    _flood_km2 and _flood_km2 > 0
                    and _aff_bld is not None
                    and (_aff_bld / _flood_km2) < _suspect_density
                ):
                    _density = round(_aff_bld / _flood_km2, 2)
                    validation_issues.append(
                        ValidationIssue(
                            severity="warning",
                            code="SAR_EXTENT_SUSPECTED_FALSE_POSITIVE",
                            message=(
                                f"Detected footprint shows extremely low "
                                f"building density ({_aff_bld:.0f} buildings "
                                f"in {_flood_km2:.1f} km² = {_density}/km²) "
                                "while the gauge is in bank — change "
                                "detection likely false positives (wet "
                                "soil/vegetation/agriculture); verify "
                                "imagery before acting."
                            ),
                            field="flood_inundation_extent",
                        )
                    )
                    self.state.log(
                        "sar_extent_suspected_false_positive",
                        density_per_km2=_density,
                        affected_buildings=int(_aff_bld),
                        flood_km2=round(_flood_km2, 2),
                        suspect_threshold_per_km2=_suspect_density,
                    )



            self.state.log(
                "resource_optimization_gate",
                target=target,
                passed=resource_gate_passed,
                reasons=resource_gate_reasons,
            )

            if not resource_gate_passed:
                optimization = {
                    "status":  "blocked_by_evidence_gate",
                    "reasons": resource_gate_reasons,
                }

            elif self.resource_tool:
                try:
                    _checkpoint_action = _safe_float(
                        (nwps_gauge.get("flood_categories", {}).get("action", {}) or {}).get("stage")
                    )
                    _water_level_val = _safe_float(observation.get("water_level"))
                    _water_ratio_for_resources = (
                        min(1.0, _water_level_val / _checkpoint_action)
                        if _checkpoint_action and _water_level_val is not None
                        else None
                    )
                    resources = await self.resources.discover(
                        target=target,
                        location=location,
                        fused_evidence=fused_evidence,
                        social_vulnerability=social_vulnerability,
                        affected_population=_safe_float(
                            gis_results["stats"].get("affected_population")
                        ),
                        water_level_ratio=_water_ratio_for_resources,
                    )

                    # Spatial input for the equity objective: SVI census
                    # tracts (population / SVI / geometry). When
                    # unavailable, the Engine skips vulnerability_coverage
                    # rather than pretending the weight injection took
                    # effect.
                    _svi_tracts = (
                        social_vulnerability.get("tracts")
                        if isinstance(social_vulnerability, dict)
                        and social_vulnerability.get("status") == "ok"
                        else None
                    )
                    plans = self.allocation.generate_plans(
                        resources=resources,
                        fused_evidence=fused_evidence,
                        svi_tracts=_svi_tracts,
                    )

                    # -- SVI influences Pareto weights: when SVI is
                    # available, add vulnerability_weight dynamically on
                    # top of the RESOURCE_OBJECTIVE_WEIGHTS_JSON base
                    # weights.
                    effective_weights = dict(self.allocation_weights)

                    _raw_profile = (
                        social_vulnerability.get("profile", {})
                        if (
                            social_vulnerability is not None
                            and isinstance(social_vulnerability, dict)
                            and social_vulnerability.get("status") == "ok"
                        )
                        else {}
                    )
                    # Narrow the type explicitly so svi_profile is a dict (avoids Pylance Any->str inference)
                    svi_profile: dict[str, Any] = (
                        _raw_profile if isinstance(_raw_profile, dict) else {}
                    )

                    if svi_profile:
                        pop_weighted_svi = svi_profile.get(
                            "population_weighted_svi"
                        )
                        if pop_weighted_svi is not None:
                            base_vuln_weight = float(
                                effective_weights.get(
                                    "vulnerability_coverage", 0.0
                                )
                            )
                            effective_weights["vulnerability_coverage"] = round(
                                base_vuln_weight
                                + self.vulnerability_weight * float(pop_weighted_svi),
                                6,
                            )
                            self.state.log(
                                "svi_weight_adjustment",
                                target=target,
                                population_weighted_svi=pop_weighted_svi,
                                base_vulnerability_weight=base_vuln_weight,
                                adjusted_vulnerability_weight=effective_weights[
                                    "vulnerability_coverage"
                                ],
                            )

                    # Proportional normalization (or clip) of weights
                    # before they are used by the optimizer

                    def _normalize_weights(
                        weights: dict[str, float],
                        objectives: list[dict[str, Any]],
                    ) -> dict[str, float]:
                        """
                        Normalize weights to sum to 1.0 so
                        recommendation_score stays within [0, 1] and is
                        comparable across runs.
                        """
                        total = sum(
                            float(weights.get(obj["name"], 0.0))
                            for obj in objectives
                        )
                        if total <= 0:
                            raise RuntimeError("All objective weights are zero.")
                        return {
                            obj["name"]: round(
                                float(weights.get(obj["name"], 0.0)) / total,
                                6,
                            )
                            for obj in objectives
                        }

                    # Normalize over the same effective objective set that
                    # optimize() uses, otherwise the injected
                    # vulnerability_coverage weight is silently dropped.
                    _active_objectives = self.allocation.effective_objectives(plans)
                    effective_weights = _normalize_weights(
                        effective_weights,
                        _active_objectives,
                    )

                    optimization = self.allocation.optimize(
                        plans,
                        weights_override=effective_weights,
                    )

                    self.state.log(
                        "pareto_optimization",
                        target=target,
                        status=optimization.get("status"),
                        candidate_plan_count=optimization.get(
                            "candidate_plan_count", 0
                        ),
                        pareto_plan_count=optimization.get(
                            "pareto_plan_count", 0
                        ),
                        recommended_plan_id=(
                            optimization.get("recommended_plan") or {}
                        ).get("plan_id"),
                        # Honesty flag: the equity objective genuinely participated in this Pareto run
                        svi_integrated=(
                            bool(svi_profile)
                            and any(
                                o["name"] == "vulnerability_coverage"
                                for o in _active_objectives
                            )
                        ),
                        effective_weights=effective_weights,  # for audit
                    )

                    if (
                        isinstance(optimization, dict)
                        and optimization.get("status") == "optimized"
                    ):
                        # -- Equity ledger --------------------------------
                        # Per-tract demand_impacts: exposed_population and
                        # hazard_exposure come from area-weighted flood
                        # intersections with census tracts; coverage comes
                        # from each plan's facility coverage radius. VWUN /
                        # equity_gap / covered-community lists are computed
                        # per plan over the whole Pareto frontier (the
                        # efficiency-equity trade-off curve), using the
                        # same social_good pure functions as the frontend
                        # /api/equity/sensitivity endpoint.
                        recommended = (
                            optimization.get("recommended_plan") or {}
                        )

                        _cover_radius_km = (
                            self.allocation.vulnerability_coverage_radius_km
                        )
                        _svi_tracts_for_lookup = (
                            social_vulnerability.get("tracts") or []
                            if isinstance(social_vulnerability, dict)
                            else []
                        )

                        # Demand records (with centroids) are built once
                        # and reused for any plan x coverage radius --
                        # the basis of the radius sensitivity sweep.
                        _demand_records = build_demand_records(
                            gis_results.get("tract_exposure", []),
                            _svi_tracts_for_lookup,
                            centroid_of=self.allocation._tract_centroid,
                        )

                        def _plan_alloc_points(plan: dict[str, Any]):
                            pts = [
                                (
                                    _safe_float(a.get("lat")),
                                    _safe_float(a.get("lon")),
                                )
                                for a in (plan.get("allocations") or [])
                            ]
                            return [
                                pt for pt in pts
                                if pt[0] is not None and pt[1] is not None
                            ]

                        def _plan_demand_impacts(
                            plan: dict[str, Any],
                            radius_km: float | None = None,
                        ):
                            return demand_impacts_for_plan(
                                _demand_records,
                                _plan_alloc_points(plan),
                                radius_km
                                if radius_km is not None
                                else _cover_radius_km,
                            )

                        # Equity curve over the full frontier: cost / VWUN
                        # / equity gap / prioritized communities per
                        # non-dominated plan
                        equity_curve = []
                        for plan in optimization.get(
                            "pareto_frontier", []
                        ):
                            impacts = _plan_demand_impacts(plan)
                            if not impacts:
                                continue
                            try:
                                vwun_p = (
                                    compute_vulnerability_weighted_unmet_need(
                                        impacts, self.vulnerability_weight
                                    )
                                )
                                gap_p, _ = compute_equity_gap_or_none(
                                    impacts, self.equity_threshold
                                )
                            except SocialGoodError:
                                continue
                            covered_ids = [
                                d["tract_id"]
                                for d in impacts if d["coverage"] >= 1.0
                            ]
                            equity_curve.append(
                                {
                                    "plan_id": plan.get("plan_id"),
                                    "cost": (plan.get("objectives") or {}).get(
                                        "cost"
                                    ),
                                    "risk_reduction": (
                                        plan.get("objectives") or {}
                                    ).get("risk_reduction"),
                                    "vulnerability_coverage": (
                                        plan.get("objectives") or {}
                                    ).get("vulnerability_coverage"),
                                    "vulnerability_weighted_unmet_need": round(
                                        vwun_p, 2
                                    ),
                                    "equity_gap": (
                                        round(gap_p, 6)
                                        if gap_p is not None
                                        else None
                                    ),
                                    "covered_tract_ids": covered_ids,
                                    "covered_high_svi_tract_ids": [
                                        d["tract_id"]
                                        for d in impacts
                                        if d["coverage"] >= 1.0
                                        and d["svi"] >= self.equity_threshold
                                    ],
                                }
                            )
                        optimization["frontier_equity_curve"] = equity_curve

                        # Main ledger for the recommended plan (same computation)
                        demand_impacts = _plan_demand_impacts(recommended)
                        if demand_impacts:
                            try:
                                vwun = (
                                    compute_vulnerability_weighted_unmet_need(
                                        demand_impacts,
                                        self.vulnerability_weight,
                                    )
                                )
                                equity_gap, equity_gap_note = (
                                    compute_equity_gap_or_none(
                                        demand_impacts,
                                        self.equity_threshold,
                                    )
                                )

                                # -- Coverage-radius sensitivity ---------
                                # Coverage is a binary "centroid within R km
                                # of a facility" test, and R is the most
                                # fragile assumption of the equity metrics:
                                # recompute VWUN / equity gap per sweep
                                # radius for the recommended plan. This is
                                # an interval, not a new recommendation.
                                radius_sensitivity = []
                                for _r in self.coverage_radius_sweep:
                                    _impacts_r = _plan_demand_impacts(
                                        recommended, radius_km=_r
                                    )
                                    if not _impacts_r:
                                        continue
                                    try:
                                        radius_sensitivity.append(
                                            {
                                                "radius_km": _r,
                                                "vulnerability_weighted_unmet_need": round(
                                                    compute_vulnerability_weighted_unmet_need(
                                                        _impacts_r,
                                                        self.vulnerability_weight,
                                                    ),
                                                    2,
                                                ),
                                                "equity_gap": (
                                                    lambda _g: round(_g, 6)
                                                    if _g is not None
                                                    else None
                                                )(
                                                    compute_equity_gap_or_none(
                                                        _impacts_r,
                                                        self.equity_threshold,
                                                    )[0]
                                                ),
                                                "covered_tract_count": sum(
                                                    1
                                                    for d in _impacts_r
                                                    if d["coverage"] >= 1.0
                                                ),
                                                "is_operating_radius": (
                                                    _r == _cover_radius_km
                                                ),
                                            }
                                        )
                                    except SocialGoodError:
                                        continue

                                self.state.log(
                                    "social_good_metrics_computed",
                                    vulnerability_weighted_unmet_need=round(
                                        vwun, 2
                                    ),
                                    equity_gap=(
                                        round(equity_gap, 6)
                                        if equity_gap is not None
                                        else None
                                    ),
                                    equity_gap_note=equity_gap_note,
                                    equity_threshold=self.equity_threshold,
                                    vulnerability_weight=(
                                        self.vulnerability_weight
                                    ),
                                    demand_tract_count=len(demand_impacts),
                                    covered_tract_count=sum(
                                        1
                                        for d in demand_impacts
                                        if d["coverage"] >= 1.0
                                    ),
                                    frontier_plan_count=len(equity_curve),
                                    radius_sensitivity=radius_sensitivity,
                                    interpretation=(
                                        "VWUN: vulnerability-weighted unmet "
                                        "need of the RECOMMENDED plan (lower "
                                        "is better). equity_gap: mean coverage "
                                        "of high-SVI tracts minus low-SVI "
                                        "tracts (negative = the most "
                                        "vulnerable are underserved)."
                                    ),
                                )
                                optimization["equity_ledger"] = {
                                    "vulnerability_weighted_unmet_need": round(
                                        vwun, 2
                                    ),
                                    "equity_gap": (
                                        round(equity_gap, 6)
                                        if equity_gap is not None
                                        else None
                                    ),
                                    "equity_gap_note": equity_gap_note,
                                    "equity_threshold": self.equity_threshold,
                                    "vulnerability_weight": (
                                        self.vulnerability_weight
                                    ),
                                    "coverage_radius_km": _cover_radius_km,
                                    "coverage_radius_sensitivity": (
                                        radius_sensitivity
                                    ),
                                    "demand_tract_count": len(demand_impacts),
                                    "demand_impacts": demand_impacts,
                                    "frontier_equity_curve": equity_curve,
                                    # Full context for frontend lambda /
                                    # radius recomputation: demand records,
                                    # recommended-plan allocation points,
                                    # frontier plans with objectives, and
                                    # the objective directions / base
                                    # weights / SVI used for scoring --
                                    # enough to preview whether the
                                    # recommendation flips under a different
                                    # lambda or radius, without rerunning
                                    # the pipeline.
                                    "sensitivity_context": {
                                        "demand_records": _demand_records,
                                        "coverage_radius_km": _cover_radius_km,
                                        "recommended_allocation_points": (
                                            _plan_alloc_points(recommended)
                                        ),
                                        "recommended_plan_id": recommended.get(
                                            "plan_id"
                                        ),
                                        "frontier_plans": [
                                            {
                                                "plan_id": plan.get("plan_id"),
                                                "allocation_points": (
                                                    _plan_alloc_points(plan)
                                                ),
                                                "objectives": dict(
                                                    plan.get("objectives") or {}
                                                ),
                                            }
                                            for plan in optimization.get(
                                                "pareto_frontier", []
                                            )
                                        ],
                                        "optimization_objectives": [
                                            {
                                                "name": o["name"],
                                                "direction": o["direction"],
                                            }
                                            for o in _active_objectives
                                        ],
                                        "weights_used": dict(effective_weights),
                                        "vulnerability_weight": (
                                            self.vulnerability_weight
                                        ),
                                        "population_weighted_svi": (
                                            svi_profile.get(
                                                "population_weighted_svi"
                                            )
                                        ),
                                    },
                                }
                            except SocialGoodError as sg_err:
                                self.state.log(
                                    "social_good_metrics_failed",
                                    error=str(sg_err),
                                )
                        else:
                            self.state.log(
                                "social_good_metrics_skipped",
                                reason=(
                                    "no per-tract exposure records: "
                                    "flood extent or SVI geometry unavailable"
                                ),
                            )

                        allocation_evidence = self._build_allocation_evidence(
                            optimization
                        )
                        evidence_list.append(allocation_evidence)

                except Exception as resource_err:
                    self.state.log(
                        "resource_optimization_warning",
                        error=str(resource_err),
                    )
                    optimization = {
                        "status": "resource_optimization_failed",
                        "error":  str(resource_err),
                    }

            else:
                optimization = {"status": "not_configured"}
            

            summary_parts = [
                f"USGS station {observation['station_id']} reported "
                f"{observation['water_level']} ft at {observation['observation_time']}."
            ]

            if fused_evidence.get("status") == "fused":
                summary_parts.append(
                    f"Fused {fused_evidence.get('observation_count', 0)} additional source(s)."
                )

            if isinstance(optimization, dict) and optimization.get("status") == "optimized":
                rec = optimization.get("recommended_plan") if isinstance(optimization, dict) else None
                if rec and isinstance(rec, dict):
                    score = rec.get("recommendation_score")
                    plan_id = rec.get("plan_id", "unknown")
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    summary_parts.append(f"Pareto recommendation: Plan {plan_id} (score: {score_str}).")
                summary_parts.append(
                    f"{optimization.get('pareto_plan_count', 0)} non-dominated plans on Pareto frontier."
                )
            else:
                summary_parts.append("No resource allocation optimization was performed.")

            summary = " ".join(summary_parts)

            next_actions = []

            if self.historical_mode:
                _skipped_names = [
                    s.get("source")
                    for s in self.sources.skipped_historical_sources
                ]
                next_actions.append(
                    f"Historical replay for {self.event_date} (strict time "
                    "alignment): gauge queried in the event window"
                    + (
                        "; current-only sources ("
                        + ", ".join(filter(None, _skipped_names))
                        + ") were excluded — no historical archive endpoint."
                        if _skipped_names
                        else "; no current-only sources were configured."
                    )
                )

            if not nwps_gauge["threshold_verified"]:
                next_actions.append(
                    "NWPS did not provide verified flood-category thresholds "
                    "for this gauge; do not infer flood severity from stage alone."
                )

            if not nwps_stageflow["stageflow_verified"]:
                next_actions.append(
                    "NWPS observed/forecast stageflow evidence is unavailable."
                )

            next_actions.append("Add observed precipitation and precipitation forecast evidence.")

            # State explicitly when there is no flood extent (so a missing
            # map does not read as a failure): distinguish "no inundation
            # detected" from "data unavailable" and say what was skipped.
            if gee_flooded_area_km2 is None:
                _rejected_sat = [
                    r.get("source", "satellite")
                    for r in (fused_evidence.get("rejected_sources", [])
                              if isinstance(fused_evidence, dict) else [])
                    if r.get("source_type") == "satellite_sar"
                ]
                next_actions.append(
                    "No satellite flood extent was "
                    + (
                        "retrieved (SAR source unavailable)"
                        if _rejected_sat
                        else "detected for this event"
                    )
                    + "; inundation-dependent analysis (flood map, areal "
                    "population exposure, route impact) was skipped. "
                    "CDRI hazard is based on water-level ratio only."
                )
            else:
                next_actions.append("Add inundation/extent data for city-wide impact assessment.")

            if isinstance(optimization, dict) and optimization.get("status") == "optimized":
                next_actions.append(
                    "Inspect the Pareto frontier to compare risk reduction, coverage, response time, cost, and unmet demand trade-offs."
                )
                next_actions.append(
                    "Validate the recommended allocation against operational constraints before dispatch."
                )
            elif self.resource_tool:
                next_actions.append(
                    "Resource discovery succeeded but no feasible allocation plan was found. Check resource availability."
                )
            else:
                next_actions.append(
                    "Configure RESOURCE_DISCOVERY_TOOL and objectives to enable Pareto optimization."
                )

            if freshness_seconds is not None:
                next_actions.append(
                    f"Check observation freshness ({freshness_seconds:.0f}s) against your operational policy."
                )
            else:
                next_actions.append(
                    "Obtain an observation timestamp from the MCP before evaluating data freshness."
                )

            return SkillResult(
                status="completed",
                summary=summary,
                evidence=evidence_list,
                spatial_objects=list(self.state.spatial_objects.values()),
                validation_issues=validation_issues,
                next_actions=next_actions,
            )

        except Exception as exc:
            import traceback
            self.state.log(
                "skill_error",
                skill=self.name,
                error=str(exc),
                traceback=traceback.format_exc(limit=12),
            )
            return SkillResult(
                status="error",
                summary=(
                    "Flood assessment failed during verified execution: "
                    f"{str(exc)}"
                ),
                evidence=[],
                spatial_objects=list(self.state.spatial_objects.values()),
                validation_issues=validation_issues,
                next_actions=[
                    "Inspect the raw Flood MCP response in the terminal.",
                    "Verify that the requested USGS station ID exists and is reachable.",
                    "If the MCP response is valid but uses an unexpected textual format, update UsgsSkill._parse_observation.",
                ],
            )
