# app/main.py

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
import warnings
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_settings.exceptions import IncompleteFieldDefinitionWarning

# Project-internal imports
from .agent import FinalDecisionAgent, StrategySelector
from .utils import safe_float as _safe_float
from .experiment import ExperimentLogger
from .historical_validation import (
    flood_category,
    load_default_manifest,
    validate_manifest,
)
from .hitl import AdaptiveHITL
from .ops import OpsExecutor
from .models import HazardType, RunState, SkillResult
from .skills import MasterRouter
from .skills import registry as skill_registry
from .skills.registry import create_skill
from .verification import Verifier

# Suppress the PydanticSettings lifespan warning
warnings.filterwarnings(
    "ignore",
    category=IncompleteFieldDefinitionWarning,
    message=".*lifespan.*",
)

# ---------------------------------------------------------------------
# Hazard type validation
# ---------------------------------------------------------------------
HAZARD_TYPE_ADAPTER = TypeAdapter(HazardType)

def validate_hazard_type(value: str) -> HazardType:
    try:
        return HAZARD_TYPE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"Unsupported hazard type: {value!r}") from exc

# ---------------------------------------------------------------------
# Request models (module-level so FastAPI can resolve them)
# ---------------------------------------------------------------------
class AssessRequest(BaseModel):
    query: str
    station_id: str | None = None
    # Optional flood plugin parameters (event replay / equity tuning)
    event_date: str | None = None          # YYYY-MM-DD
    vulnerability_weight: float | None = None
    equity_threshold: float | None = None
    # Demo mode: FakeMCP offline pipeline (zero network dependency);
    # yields full indices + equity ledger for any date.
    demo: bool = False


class PlanSelectionRequest(BaseModel):
    run_id: str
    scenario_id: str
    plan_id: str
    selected_by: str = "operator"
    note: str | None = None

# ---------------------------------------------------------------------
# 1. CLI single-run
# ---------------------------------------------------------------------
async def run_once(user_task: str) -> None:
    """Single CLI run (event replay / tuning is web-mode only)."""
    load_dotenv()
    run_id = str(uuid.uuid4())
    state = RunState(run_id=run_id)
    logger = ExperimentLogger()
    verifier = Verifier()
    hitl = AdaptiveHITL(state)
    ops = OpsExecutor("config/mcp.json")

    print(f"\nRun ID: {run_id}")
    print("Loading MCP servers...")
    try:
        await ops.connect()
        ops.print_catalog()

        router = MasterRouter()
        raw_hazard = router.classify(user_task)
        hazard = validate_hazard_type(raw_hazard)
        target = router.extract_target(user_task, hazard)
        state.hazard_type = hazard

        logger.log(run_id, "master_router", "task_classified", {
            "user_task": user_task,
            "hazard": hazard,
            "target": target,
        })

        print("\n" + "=" * 72)
        print("MASTER AGENT")
        print("=" * 72)
        print(f"Hazard: {hazard}")
        print(f"Target: {target}")

        # Three-part input "<hazard> <place> <station>": the station ID
        # is extracted here via deterministic regex and verified against
        # the USGS metadata service.
        station_id = router.extract_station_id(user_task)
        if station_id:
            print(f"Station: {station_id}")

        # Plugin factory: adding a new hazard requires no change here.
        skill = create_skill(hazard, state, verifier, hitl, ops, logger)

        print("\n" + "=" * 72)
        print(f"RUNNING SKILL: {skill.name}")
        print("=" * 72)

        result = await skill.run(
            target, station_id=station_id, raw_task=user_task
        )
        logger.log(run_id, skill.name, "skill_completed", result.model_dump())

        # Agent-layer strategy selection: short-circuit and skip the LLM
        # when evidence is clearly quiet. A calm gauge does not mean a
        # safe city: never short-circuit when flood extent or active
        # alerts exist.
        structured = _extract_structured_data(result, state)
        strategy = StrategySelector().select(
            water_level=structured.get("water_level"),
            action_stage=structured.get("action_stage"),
            skill_name=skill.name,
            flooded_area_km2=structured.get("flooded_area_km2"),
            alert_count=structured.get("alert_count"),
        )
        state.log(
            "agent_strategy_selected",
            strategy=strategy.strategy,
            skip_llm=strategy.skip_llm,
            reason=strategy.reason,
        )

        if strategy.skip_llm:
            final_text = strategy.message
        else:
            spec = skill_registry.get_spec(hazard)
            reporter = FinalDecisionAgent()
            final_text = await reporter.synthesize(
                user_task=user_task,
                skill_name=skill.name,
                result=result,
                hazard_rules=(
                    spec.prompt_rules if spec else ""
                ),
                output_validator=(
                    spec.output_validator if spec else None
                ),
            )

        print("\n" + "=" * 72)
        print("FINAL DECISION SUPPORT")
        print("=" * 72)
        print(final_text)

        print("\n" + "=" * 72)
        print("RUN SUMMARY")
        print("=" * 72)
        print(f"Run ID: {run_id}")
        print(f"Status: {result.status}")
        print(f"Events logged: {len(state.events)}")
        print(f"Spatial objects: {len(state.spatial_objects)}")
        print(f"HITL requests: {len(state.hitl_requests)}")
        print("Trajectory log: runs/trajectories.jsonl")

    finally:
        await ops.close()

# ---------------------------------------------------------------------
# 2. Structured data extraction
# ---------------------------------------------------------------------
def _extract_structured_data(result: SkillResult, state: RunState) -> dict:
    structured = {
        "water_level": None,
        "action_stage": None,
        "minor_stage": None,
        "moderate_stage": None,
        "major_stage": None,
        "station_id": None,
        "station_name": None,
        "observation_time": None,
        "observation_semantics": None,
        "window_summary": None,
        "assessment_mode": "realtime",
        "time_alignment": None,
        "station_lat": None,
        "station_lon": None,
        "target_lat": None,
        "target_lon": None,
        "target_name": None,
        "svi": None,
        "svi_population": None,
        "svi_tracts": None,
        "svi_scope": None,
        "svi_radius_km": None,
        "population_context": None,
        "population_total": None,
        "population_affected": None,
        "population_affected_source": None,
        "temperature": None,
        "infrastructure_counts": {},
        "osm_facilities": [],
        "decision_indices": {},
        "analysis_radius_km": None,
        # Alert, inundation, and forecast fields
        "alerts": None,
        "alert_count": 0,
        "flooded_area_km2": None,
        "flood_extent": None,
        "extent_provenance": None,
        "extent_model": None,
        "extent_confidence": None,
        "extent_timestamp": None,
        "precipitation": None,
        "forecast": None,
        # Per-period forecast details (temperature, precipitation
        # probability, etc.) for the sidebar
        "forecast_periods": [],
        "hydrologic_category": None,
        "next_flood_threshold": None,
        "next_actions": list(result.next_actions),
    }

    # Target info from state.spatial_objects
    for obj_id, obj in state.spatial_objects.items():
        if obj.object_type == "place" and obj.attributes.get("name"):
            structured["target_name"] = obj.attributes["name"]
            if structured["target_lat"] is None and obj.geometry:
                coords = obj.geometry.get("coordinates")
                if coords and len(coords) == 2:
                    structured["target_lon"] = coords[0]
                    structured["target_lat"] = coords[1]
            break

    # Walk the evidence items
    for ev in result.evidence:
        attrs = ev.attributes

        # Station info and water level
        if "station_id" in attrs:
            structured["station_id"] = attrs["station_id"]
            structured["station_name"] = attrs.get("station_name")
            structured["observation_time"] = attrs.get("observation_time")
            structured["observation_semantics"] = attrs.get(
                "observation_semantics"
            )
            structured["window_summary"] = attrs.get("window_summary")
            if attrs.get("observation_semantics") == "window_peak":
                structured["assessment_mode"] = "historical"
            if isinstance(attrs.get("time_alignment"), dict):
                structured["time_alignment"] = attrs["time_alignment"]
                structured["assessment_mode"] = attrs["time_alignment"].get(
                    "mode", structured["assessment_mode"]
                )
            structured["station_lat"] = attrs.get("station_latitude")
            structured["station_lon"] = attrs.get("station_longitude")
            if structured["target_lat"] is None:
                structured["target_lat"] = attrs.get("target_latitude")
                structured["target_lon"] = attrs.get("target_longitude")
            structured["water_level"] = attrs.get("water_level")
            if "nwps_flood_categories" in attrs:
                cats = attrs["nwps_flood_categories"]
                structured["action_stage"] = cats.get("action", {}).get("stage")
                structured["minor_stage"] = cats.get("minor", {}).get("stage")
                structured["moderate_stage"] = cats.get("moderate", {}).get("stage")
                structured["major_stage"] = cats.get("major", {}).get("stage")

        # SVI
        if "SVI" in ev.source:
            structured["svi"] = attrs.get("population_weighted_svi")
            structured["svi_population"] = attrs.get("total_population")
            structured["population_total"] = attrs.get("total_population")
            structured["svi_tracts"] = attrs.get("tract_count")
            structured["svi_scope"] = attrs.get("tract_scope")
            structured["svi_radius_km"] = attrs.get("radius_km")

        # Population context and temperature come only from structured
        # fields (the containing-tract population in the fusion branch
        # below; temperature is backfilled after the loop), never guessed
        # from free-text prose.  Population context is not exposure.

        # Alerts: only fused evidence with source_type == "warning"
        # counts. Do not substring-match "alert" in ev.source;
        # water-level sources contain it without carrying alert data.
        if ev.attributes.get("source_type") == "warning":
            alert_count = ev.attributes.get("alert_count")
            if alert_count is None:
                alert_count = ev.attributes.get("count")
            if alert_count is not None:
                structured["alerts"] = f"{alert_count} active"
                try:
                    structured["alert_count"] = max(
                        structured["alert_count"], int(alert_count)
                    )
                except (TypeError, ValueError):
                    pass
            else:
                structured["alerts"] = "Active"

        # Forecast: only trust source_type == "forecast"; no source
        # substring guessing (same false-match risk as alerts).
        if ev.attributes.get("source_type") == "forecast":
            forecast_count = ev.attributes.get("forecast_count") or ev.attributes.get("count")
            if forecast_count is not None:
                structured["forecast"] = f"{forecast_count} periods"
            else:
                structured["forecast"] = "Available"

            periods = ev.attributes.get("forecasts")
            if isinstance(periods, list) and not structured["forecast_periods"]:
                structured["forecast_periods"] = [
                    {
                        "period": p.get("period"),
                        "temperature": p.get("temperature"),
                        "temperature_unit": p.get("temperature_unit"),
                        "probability_of_precipitation": p.get("probability_of_precipitation"),
                        "wind_speed": p.get("wind_speed"),
                        "short_forecast": p.get("short_forecast"),
                        "timestamp": p.get("timestamp"),
                    }
                    for p in periods
                    if isinstance(p, dict)
                ][:8]

        # Extract flood_extent (spatial_extent)
        if ev.attributes.get("source_type") == "inundation" or "spatial_extent" in ev.observation.lower():
            # Look for flooded_area_km2
            flooded_area = ev.attributes.get("flooded_area_km2")
            if flooded_area is not None:
                structured["flood_extent"] = f"{flooded_area} km²"
            else:
                structured["flood_extent"] = "Detected"
            structured["flood_extent_status"] = "detected"

        # FloodSkill persists normalized fusion observations on the primary
        # evidence item.  Read OSM directly instead of scraping its prose
        # summary so the side panel always reflects fetched infrastructure.
        fusion = attrs.get("multi_source_fusion")
        if isinstance(fusion, dict):
            structured["analysis_radius_km"] = fusion.get("analysis_radius_km") or structured["analysis_radius_km"]
            # Raw-data panel: per-source fused values plus conflict
            # flags, so contested numbers stay visible in the UI.
            # Evidence attribute key is fused_values (value/unit/
            # conflict/conflict_ratio/source_values).
            structured["fused_measurements"] = fusion.get(
                "fused_values", {}
            )
            structured["rejected_sources"] = [
                {
                    "source": r.get("source"),
                    "reason": str(r.get("reason", ""))[:120],
                }
                for r in fusion.get("rejected_sources", [])
                if isinstance(r, dict)
            ]
            for item in fusion.get("observations", []):
                if not isinstance(item, dict):
                    continue
                raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
                if raw.get("source_type") in {
                    "population_context",
                    "population_exposure",  # compatibility with older MCP output
                }:
                    population = raw.get("population", {})
                    if isinstance(population, dict) and population.get("total") is not None:
                        # This MCP returns the containing tract's population,
                        # not flood exposure.  Keep it as contextual evidence
                        # and never use it as affected_population.
                        structured["population_context"] = population["total"]
                if raw.get("source_type") == "infrastructure":
                    measurements = raw.get("measurements", {})
                    if isinstance(measurements, dict):
                        for key, value in measurements.items():
                            if isinstance(value, dict) and value.get("value") is not None:
                                structured["infrastructure_counts"][f"nearby_{key}"] = value["value"]
                    facilities = raw.get("metadata", {}).get("facilities", [])
                    if isinstance(facilities, list):
                        structured["osm_facilities"] = facilities[:100]
                if raw.get("source_type") == "road_network":
                    measurements = raw.get("measurements", {})
                    if isinstance(measurements, dict):
                        for key, value in measurements.items():
                            if isinstance(value, dict) and value.get("value") is not None:
                                structured["infrastructure_counts"][f"nearby_{key}"] = value["value"]

        # Resource optimization and equity objectives
        # (auditable: whether the SVI weight took effect)
        if attrs.get("optimization_status") is not None:
            rec = attrs.get("recommended_plan") or {}
            supply_demand = rec.get("supply_demand") or {}
            structured["resource_optimization"] = {
                "status": attrs.get("optimization_status"),
                "svi_weight_applied": attrs.get("svi_weight_applied"),
                "recommended_plan_id": rec.get("plan_id"),
                "vulnerability_coverage": (rec.get("objectives") or {}).get(
                    "vulnerability_coverage"
                ),
                "served_people": supply_demand.get("served_people"),
                "unmet_people": supply_demand.get("unmet_people"),
                "assignment_method": supply_demand.get("method"),
                "travel_distance_p95_km": supply_demand.get(
                    "travel_distance_p95_km"
                ),
                "transfer_time_estimate_range_minutes": supply_demand.get(
                    "transfer_time_estimate_range_minutes"
                ),
                "time_estimate_basis": supply_demand.get("time_estimate_basis"),
                "planning_gaps": supply_demand.get("planning_gaps", []),
                "excluded_facility_types": supply_demand.get(
                    "excluded_facility_types", []
                ),
                "facility_utilization": supply_demand.get(
                    "facility_utilization", []
                ),
                "equity_ledger": attrs.get("equity_ledger"),
                "plan_scenarios": attrs.get("plan_scenarios", []),
            }

        if isinstance(attrs.get("decision_indices"), dict):
            structured["decision_indices"] = attrs["decision_indices"]
        if isinstance(attrs.get("gis_stats"), dict):
            stats = attrs["gis_stats"]
            if "affected_buildings" in stats:
                structured["infrastructure_counts"]["affected_buildings"] = stats["affected_buildings"]
            if "affected_facilities" in stats:
                structured["infrastructure_counts"]["affected_facilities"] = stats["affected_facilities"]
            if "affected_roads" in stats:
                structured["infrastructure_counts"]["affected_roads"] = stats["affected_roads"]
            if "unverified_surface_water_area_km2" in stats:
                structured["unverified_surface_water_area_km2"] = stats[
                    "unverified_surface_water_area_km2"
                ]
            if "affected_bridges" in stats:
                structured["infrastructure_counts"]["affected_bridges"] = stats["affected_bridges"]
            if stats.get("extent_provenance") is not None:
                structured["extent_provenance"] = stats["extent_provenance"]
            if stats.get("extent_model") is not None:
                structured["extent_model"] = stats["extent_model"]
            # Flood area (recomputed after clipping to the city boundary)
            flood_area = stats.get("flood_area_km2")
            if flood_area is not None and structured["flood_extent"] is None:
                structured["flood_extent"] = f"{flood_area} km²"
                structured.setdefault("flood_extent_status", "detected")
            if flood_area is not None:
                try:
                    structured["flooded_area_km2"] = float(flood_area)
                except (TypeError, ValueError):
                    pass
            # Affected population (census tracts intersected with the
            # flood extent), for the sidebar and indices
            if stats.get("affected_population") is not None:
                structured["population_affected"] = stats["affected_population"]
                # Primary key is affected_population_method (e.g.
                # "areal_weighted"); affected_population_source is only
                # a legacy fallback name.
                structured["population_affected_source"] = stats.get(
                    "affected_population_method",
                    stats.get(
                        "affected_population_source",
                        "census_tract_areal_weighted",
                    ),
                )
                if stats.get("affected_population_tracts") is not None:
                    structured["infrastructure_counts"]["affected_population_tracts"] = stats[
                        "affected_population_tracts"
                    ]

        # Precipitation: prefer the structured no-rain status; avoid
        # regex-matching unrelated numbers in free text.
        if ev.attributes.get("precipitation_status") == "no_precipitation_observed":
            structured["precipitation"] = "None observed"
        elif "precipitation" in ev.observation.lower():
            nums = re.findall(r'(\d+\.?\d*)\s*(?:mm|毫米)', ev.observation)
            if nums:
                structured["precipitation"] = f"{nums[0]} mm"
            else:
                structured["precipitation"] = "Available"

    # Additionally extract infrastructure counts from result.summary
    if result.summary:
        facilities = re.search(r'Facilities:\s*(\d+)', result.summary)
        if facilities:
            structured["infrastructure_counts"]["facilities"] = int(facilities.group(1))
        roads = re.search(r'Roads:\s*(\d+)', result.summary)
        if roads:
            structured["infrastructure_counts"]["roads"] = int(roads.group(1))
        bridges = re.search(r'Bridges:\s*(\d+)', result.summary)
        if bridges:
            structured["infrastructure_counts"]["bridges"] = int(bridges.group(1))

    # Fallback: label flood extent status explicitly so the frontend
    # never has to guess why no map is shown.
    structured.setdefault(
        "flood_extent_status",
        "detected" if structured.get("flood_extent") else "not_detected",
    )

    # Temperature backfill: from fused structured measurements (weather
    # source measurements.temperature.value), not free text.
    if structured["temperature"] is None:
        _temp = (structured.get("fused_measurements") or {}).get("temperature")
        if isinstance(_temp, dict) and _temp.get("value") is not None:
            structured["temperature"] = _temp["value"]

    # Population context backfill: the containing-tract Census population is
    # useful provenance/context but is not an affected-population estimate.
    if structured["population_context"] is None:
        _pop = (structured.get("fused_measurements") or {}).get(
            "containing_tract_population"
        )
        if isinstance(_pop, dict) and _pop.get("value") is not None:
            structured["population_context"] = _pop["value"]

    # Precipitation backfill: same source as temperature (fused
    # structured measurements). Skip probability fields (a percentage,
    # not an amount).
    if structured["precipitation"] is None:
        _fm = structured.get("fused_measurements") or {}
        for _key, _m in _fm.items():
            if "precipitation" not in _key or "probability" in _key:
                continue
            if isinstance(_m, dict) and _m.get("value") is not None:
                structured["precipitation"] = (
                    f"{_m['value']} {_m.get('unit') or 'mm'}".strip()
                )
                break

    # The accepted spatial object is the operational extent. A raw SAR
    # observation or area-only statistic cannot promote itself to one.
    accepted_extent = False
    for obj in result.spatial_objects:
        if obj.object_type != "flood_extent":
            continue
        geometry = obj.geometry or {}
        if not isinstance(geometry, dict):
            continue
        if geometry.get("type") == "FeatureCollection":
            if not geometry.get("features"):
                continue
        elif not geometry.get("coordinates"):
            continue
        accepted_extent = True
        obj_attrs = obj.attributes or {}
        obj_area = _safe_float(obj_attrs.get("flooded_area_km2"))
        if obj_area is not None:
            structured["flooded_area_km2"] = obj_area
            structured["flood_extent"] = f"{obj_area} km²"
        elif obj.geometry:
            structured["flood_extent"] = "Detected"
        structured["flood_extent_status"] = "detected"
        structured["extent_provenance"] = (
            obj_attrs.get("extent_provenance")
            or structured.get("extent_provenance")
        )
        structured["extent_model"] = (
            obj_attrs.get("model") or structured.get("extent_model")
        )
        structured["extent_confidence"] = (
            obj_attrs.get("confidence")
            if obj_attrs.get("confidence") is not None
            else obj.confidence
        )
        structured["extent_timestamp"] = obj.timestamp
        break
    if not accepted_extent:
        structured["flood_extent"] = None
        structured["flooded_area_km2"] = None
        structured["flood_extent_status"] = "not_detected"
        structured["extent_provenance"] = None
        structured["extent_model"] = None
        structured["extent_confidence"] = None
        structured["extent_timestamp"] = None

    # Authoritative hydrologic classification is computed once in the
    # backend from the station-specific NWPS thresholds.  The frontend only
    # renders this result and never re-implements threshold logic.
    stage = _safe_float(structured.get("water_level"))
    categories = {
        name: value
        for name in ("action", "minor", "moderate", "major")
        if (
            value := _safe_float(structured.get(f"{name}_stage"))
        ) is not None
    }
    if stage is not None and "action" in categories:
        category = flood_category(stage, categories)
        structured["hydrologic_category"] = category
        ordered = ("action", "minor", "moderate", "major")
        next_name = None
        if category == "below_action":
            next_name = "action"
        elif category in ordered:
            current_index = ordered.index(category)
            for candidate in ordered[current_index + 1:]:
                if candidate in categories:
                    next_name = candidate
                    break
        if next_name is not None:
            next_stage = categories[next_name]
            structured["next_flood_threshold"] = {
                "category": next_name,
                "stage_ft": next_stage,
                "margin_ft": round(next_stage - stage, 3),
            }

    return structured

# ---------------------------------------------------------------------
# 3. Web mode (FastAPI)
# ---------------------------------------------------------------------
# Per-run state keyed by run_id, so concurrent /api/assess requests
# never share HITL instances or equity contexts.
_active_hitl: dict[str, "AdaptiveHITL"] = {}
_equity_contexts: dict[str, dict] = {}
_plan_contexts: dict[str, dict[str, str]] = {}
_EQUITY_CONTEXT_LIMIT = 8   # the lambda slider only needs recent run contexts


def _remember_equity_context(run_id: str, context: dict) -> None:
    """Store equity-ledger context for the lambda/radius sliders, capped to bound growth."""
    _equity_contexts[run_id] = context
    while len(_equity_contexts) > _EQUITY_CONTEXT_LIMIT:
        oldest = next(iter(_equity_contexts))
        _equity_contexts.pop(oldest)
        _plan_contexts.pop(oldest, None)


def _latest_equity_context() -> dict | None:
    """Most recent equity context; fallback for requests without a run_id."""
    if not _equity_contexts:
        return None
    return _equity_contexts[next(reversed(_equity_contexts))]


def _prune_static_maps(static_map_dir: str, max_age_hours: float = 24.0) -> None:
    """Delete artifacts in static/maps older than max_age_hours.

    Each assessment copies map HTML and layer GeoJSON files there, so
    the directory grows without bound unless pruned. A failed delete
    never blocks the remaining cleanup.
    """
    try:
        cutoff = time.time() - max_age_hours * 3600.0
        for name in os.listdir(static_map_dir):
            path = os.path.join(static_map_dir, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.unlink(path)
            except OSError:
                pass
    except OSError:
        pass


def start_web_server():
    """Start the FastAPI web server."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    import os

    # Global MCP manager
    mcp_manager = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal mcp_manager
        load_dotenv()
        mcp_manager = OpsExecutor("config/mcp.json")
        await mcp_manager.connect()
        yield
        if mcp_manager:
            await mcp_manager.close()

    

    app = FastAPI(lifespan=lifespan)
    # CORS allows only localhost origins: pages are served same-origin
    # from /static, and open CORS would let any webpage answer
    # /api/hitl/submit (the human-approval endpoint).
    _port = int(os.environ.get("PORT", "8000"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{_port}",
            f"http://127.0.0.1:{_port}",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files directory
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if not os.path.exists(static_dir):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/hitl/status")
    async def hitl_status():
        """Check for a pending HITL request (earliest pending run first)."""
        for run_id, hitl in _active_hitl.items():
            pending = hitl.get_pending_request()
            if pending:
                return {
                    "has_pending": True,
                    "run_id": run_id,
                    "request": pending,
                }
        return {"has_pending": False}

    @app.post("/api/hitl/submit")
    async def hitl_submit(request_data: dict):
        """Submit an HITL response; with a run_id, route to that exact run."""
        value = request_data.get("value")
        run_id = request_data.get("run_id")
        if run_id is not None:
            hitl = _active_hitl.get(run_id)
            if hitl is not None:
                hitl.submit_response(value)
            return {"status": "ok"}
        # Without a run_id (legacy clients): submit only to an instance
        # that actually has a pending request.
        routed = False
        for hitl in _active_hitl.values():
            if hitl.get_pending_request() is not None:
                hitl.submit_response(value)
                routed = True
                break
        return {"status": "ok", "routed": routed}
    
    @app.get("/api/validation")
    async def historical_validation_status():
        """Held-out historical validation declared in the manifest.

        Surfaces the offline CLI result (scripts/validate_historical_events.py)
        through the product so the claim is visible without running pytest.
        """
        try:
            result = validate_manifest(load_default_manifest())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"status": "error", "reason": str(exc)}
        return result

    @app.post("/api/assess")
    async def assess(req: AssessRequest):
        if mcp_manager is None and not req.demo:
            raise HTTPException(status_code=503, detail="MCP Manager not ready")

        # Build overrides (keys recognized by the flood plugin)
        overrides = {}
        if req.event_date:
            overrides['event_date'] = req.event_date
        if req.vulnerability_weight is not None:
            overrides['vulnerability_weight'] = req.vulnerability_weight
        if req.equity_threshold is not None:
            overrides['equity_threshold'] = req.equity_threshold

        if req.demo:
            # Demo mode: FakeMCP offline pipeline, zero network
            # dependency; always produces full decision indices plus an
            # equity ledger. No HITL instance is registered (the
            # FakeMCP pipeline is self-contained).
            from .demo_fixtures import run_demo_assessment

            result, state = await run_demo_assessment(
                event_date=req.event_date,
                vulnerability_weight=req.vulnerability_weight,
                equity_threshold=req.equity_threshold,
            )
            hazard = "flood"
            run_id = state.run_id
        else:
            router = MasterRouter()
            try:
                hazard = router.classify(req.query)
                target = router.extract_target(req.query, hazard)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            # Three-part "<hazard> <place> <station>": a station ID in
            # the query takes precedence over the form field.
            station_id = req.station_id or router.extract_station_id(req.query)

            run_id = str(uuid.uuid4())
            state = RunState(run_id=run_id)

            hitl = AdaptiveHITL(state)
            hitl.enable_web_mode()
            # Registered by run_id so concurrent requests never
            # overwrite each other; unregistered when the run ends.
            _active_hitl[run_id] = hitl

            verifier = Verifier()
            logger = ExperimentLogger()

            # Plugin factory: no change needed here when new hazards
            # register (only the flood skill consumes station_id).
            try:
                skill = create_skill(
                    hazard, state, verifier, hitl, mcp_manager, logger
                )
                result = await skill.run(
                    target=target,
                    station_id=station_id,
                    raw_task=req.query,
                    overrides=overrides
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            finally:
                _active_hitl.pop(run_id, None)

        # Extract structured data (also feeds agent strategy selection)
        structured = _extract_structured_data(result, state)

        # Agent-layer strategy selection: short-circuit and skip the LLM
        # when evidence is clearly quiet.
        strategy = StrategySelector().select(
            water_level=structured.get("water_level"),
            action_stage=structured.get("action_stage"),
            skill_name=hazard,
            flooded_area_km2=structured.get("flooded_area_km2"),
            alert_count=structured.get("alert_count"),
        )
        state.log(
            "agent_strategy_selected",
            strategy=strategy.strategy,
            skip_llm=strategy.skip_llm,
            reason=strategy.reason,
        )

        if strategy.skip_llm:
            report = strategy.message
        else:
            spec = skill_registry.get_spec(hazard)
            agent = FinalDecisionAgent()
            report = await agent.synthesize(
                req.query, hazard, result,
                hazard_rules=(spec.prompt_rules if spec else ""),
                output_validator=(spec.output_validator if spec else None),
            )

        gis_map_path = state.gis_results.get("map_path") if hasattr(state, 'gis_results') else None
        gis_map_url = None
        static_map_dir = os.path.join("static", "maps")
        os.makedirs(static_map_dir, exist_ok=True)
        _prune_static_maps(static_map_dir)
        if gis_map_path and os.path.exists(gis_map_path):
            import shutil
            map_filename = os.path.basename(gis_map_path)
            static_map_path = os.path.join(static_map_dir, map_filename)
            shutil.copy(gis_map_path, static_map_path)
            gis_map_url = f"/static/maps/{map_filename}"

        # Copy spatial layer GeoJSON into the static directory for the
        # main Leaflet map to overlay (flood extent, affected buildings,
        # rescue routes, POIs).
        gis_layers = {}
        if hasattr(state, 'gis_results'):
            layer_keys = (
                ("flood_extent_url", "flood_boundary_path"),
                ("affected_buildings_url", "affected_buildings_path"),
                ("rescue_route_url", "rescue_route_path"),
                ("poi_url", "poi_path"),
                ("context_extent_url", "context_extent_path"),
            )
            for url_key, path_key in layer_keys:
                layer_path = state.gis_results.get(path_key)
                if layer_path and os.path.exists(layer_path):
                    import shutil as _shutil
                    # run_id filename prefix keeps concurrent runs from
                    # overwriting each other's layer files.
                    layer_filename = f"{run_id[:8]}_{path_key}.geojson"
                    layer_static = os.path.join(static_map_dir, layer_filename)
                    try:
                        _shutil.copy(layer_path, layer_static)
                        gis_layers[url_key] = f"/static/maps/{layer_filename}"
                    except OSError:
                        pass

        # Active-hazard / cross-dimension contradiction flags (frontend
        # labeling and panel suppression)
        _di_final = {}
        for _ev in result.evidence:
            if "decision_indices" in (_ev.attributes or {}):
                _di_final = _ev.attributes["decision_indices"]
        _ws = _safe_float((_di_final.get("inputs") or {}).get("water_severity"))
        _issue_codes = {i.code for i in result.validation_issues}
        # A quiet gauge is not sufficient to dismiss a corroborated pluvial
        # flood or an active official warning.  Use the same evidence set as
        # the strategy gate rather than deriving UI state from CDRI's
        # water-only hazard component.
        active_hazard = bool(
            (_ws is not None and _ws > 0)
            or ((_safe_float(structured.get("flooded_area_km2")) or 0) > 0)
            or ((_safe_float(structured.get("alert_count")) or 0) > 0)
        )
        sar_contradiction = "SAR_EXTENT_STAGE_CONTRADICTION" in _issue_codes
        sar_implausible = "SAR_EXTENT_IMPLAUSIBLE" in _issue_codes
        sar_uncorroborated = "SAR_EXTENT_UNCORROBORATED" in _issue_codes

        # Save the equity ledger context so the lambda slider can
        # recompute VWUN without rerunning the pipeline (keyed by
        # run_id so concurrent runs stay isolated).
        opt_ledger = None
        for _ev in result.evidence:
            _ledger = (_ev.attributes or {}).get("equity_ledger")
            if isinstance(_ledger, dict):
                opt_ledger = _ledger
                break
        if opt_ledger and isinstance(
            opt_ledger.get("demand_impacts"), list
        ):
            _remember_equity_context(run_id, {
                "run_id": run_id,
                "demand_impacts": opt_ledger["demand_impacts"],
                "equity_threshold": opt_ledger.get("equity_threshold", 0.9),
                "frontier_equity_curve": opt_ledger.get(
                    "frontier_equity_curve", []
                ),
                "sensitivity_context": opt_ledger.get(
                    "sensitivity_context"
                ),
                "covered_tracts_by_plan": {
                    entry.get("plan_id"): entry.get(
                        "covered_high_svi_tract_ids", []
                    )
                    for entry in opt_ledger.get(
                        "frontier_equity_curve", []
                    )
                    if isinstance(entry, dict)
                },
            })
        scenarios = (
            (structured.get("resource_optimization") or {}).get("plan_scenarios")
            or []
        )
        _plan_contexts[run_id] = {
            str(item["scenario_id"]): str(item["plan_id"])
            for item in scenarios
            if isinstance(item, dict)
            and item.get("scenario_id")
            and item.get("plan_id")
        }
        while len(_plan_contexts) > _EQUITY_CONTEXT_LIMIT:
            _plan_contexts.pop(next(iter(_plan_contexts)))

        return {
            "run_id": run_id,
            "report": report,
            "skill_used": hazard,
            "status": result.status,
            # When status == "error", give the frontend an actionable
            # reason; empty "--" panels alone cannot distinguish "no
            # hazard" from "failed".
            "error_detail": (
                (result.summary or "")[:500]
                if result.status == "error"
                else None
            ),
            "structured": structured,
            "gis_map_url": gis_map_url,
            "gis_layers": gis_layers,
            "active_hazard": active_hazard,
            "sar_stage_contradiction": sar_contradiction,
            "sar_implausible": sar_implausible,
            "sar_uncorroborated": sar_uncorroborated,
        }

    @app.post("/api/plan-selection")
    async def select_plan(payload: PlanSelectionRequest):
        """Record an explicit human choice among generated plan profiles."""
        allowed = _plan_contexts.get(payload.run_id)
        if allowed is None:
            raise HTTPException(status_code=404, detail="assessment run not found")
        if allowed.get(payload.scenario_id) != payload.plan_id:
            raise HTTPException(
                status_code=400,
                detail="scenario and plan do not match this assessment",
            )
        ExperimentLogger().log(
            payload.run_id,
            "plan_selection",
            "human_plan_selected",
            {
                "scenario_id": payload.scenario_id,
                "plan_id": payload.plan_id,
                "selected_by": payload.selected_by[:80],
                "note": payload.note[:500] if payload.note else None,
            },
        )
        return {"status": "recorded", "plan_id": payload.plan_id}

    @app.post("/api/equity/sensitivity")
    async def equity_sensitivity(payload: dict):
        """Recompute equity metrics for a different vulnerability weight."""
        run_id = payload.get("run_id")
        equity = (
            _equity_contexts.get(run_id)
            if run_id is not None
            else _latest_equity_context()
        )
        if run_id is not None and equity is None:
            raise HTTPException(
                status_code=404,
                detail=f"No equity context for run {run_id}.",
            )
        if not equity or not equity.get("demand_impacts"):
            raise HTTPException(
                status_code=404,
                detail=(
                    "No equity context available yet — run a flood "
                    "assessment with flood extent + SVI data first."
                ),
            )
        from .social_good import (
            SocialGoodError,
            compute_coverage_concentration_index_or_none,
            compute_equity_gap_or_none,
            compute_normalized_vulnerability_weighted_unmet_need,
            compute_vulnerability_weighted_unmet_need,
        )

        try:
            lam = float(payload.get("vulnerability_weight", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid lambda")
        if lam < 0 or lam > 10:
            raise HTTPException(
                status_code=400, detail="lambda must be within [0, 10]"
            )
        threshold = float(
            payload.get(
                "equity_threshold",
                equity["equity_threshold"],
            )
        )

        context = equity.get("sensitivity_context") or {}
        impacts = equity["demand_impacts"]

        try:
            vwun = compute_vulnerability_weighted_unmet_need(impacts, lam)
            normalized_vwun = (
                compute_normalized_vulnerability_weighted_unmet_need(
                    impacts, lam
                )
            )
            gap, gap_note = compute_equity_gap_or_none(impacts, threshold)
            concentration, concentration_note = (
                compute_coverage_concentration_index_or_none(impacts)
            )
        except SocialGoodError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        # Preview which already-computed frontier plan would win under this
        # vulnerability weight. Assignment results are not recomputed.
        frontier_preview = None
        frontier_plans = context.get("frontier_plans") or []
        obj_defs = context.get("optimization_objectives") or []
        base_weights = context.get("base_objective_weights") or {}
        pop_svi = context.get("population_weighted_svi")
        frontier_impacts = context.get("frontier_plan_impacts") or {}
        if (
            frontier_plans
            and obj_defs
            and base_weights
            and pop_svi is not None
            and frontier_impacts
        ):
            import copy

            per_plan = {}
            normalized_per_plan = {}
            for fp in frontier_plans:
                _imp = frontier_impacts.get(str(fp.get("plan_id")))
                if not _imp:
                    continue
                try:
                    per_plan[fp["plan_id"]] = round(
                        compute_vulnerability_weighted_unmet_need(_imp, lam),
                        2,
                    )
                    normalized_per_plan[fp["plan_id"]] = round(
                        compute_normalized_vulnerability_weighted_unmet_need(
                            _imp, lam
                        ),
                        6,
                    )
                except (SocialGoodError, KeyError, ValueError):
                    continue
            # Rebuild from the unnormalized configured weights. Subtracting
            # λ from already-normalized weights changes their ratios.
            _new_weights = dict(base_weights)
            if any(o["name"] == "vulnerability_coverage" for o in obj_defs):
                _new_weights["vulnerability_coverage"] = (
                    float(base_weights.get("vulnerability_coverage", 0.0))
                    + lam * float(pop_svi)
                )
            _total = sum(
                _new_weights.get(o["name"], 0.0) for o in obj_defs
            )
            would_recommend = None
            if _total > 0:
                _new_weights = {
                    o["name"]: _new_weights.get(o["name"], 0.0) / _total
                    for o in obj_defs
                }
                from app.engine.pareto_engine import (
                    select_best_pareto_plan,
                )

                _plans_copy = copy.deepcopy(frontier_plans)
                _renamed = [
                    {**p, "objectives": p.get("objectives", {})}
                    for p in _plans_copy
                ]
                try:
                    _best = select_best_pareto_plan(
                        _renamed, obj_defs, _new_weights
                    )
                    would_recommend = _best.get("plan_id")
                except (RuntimeError, KeyError, ValueError):
                    would_recommend = None
            frontier_preview = {
                "vwun_by_plan": per_plan,
                "normalized_vwun_by_plan": normalized_per_plan,
                "lowest_vwun_plan": (
                    min(per_plan, key=per_plan.get)
                    if per_plan
                    else None
                ),
                "would_recommend": would_recommend,
                "recommendation_changed": (
                    would_recommend is not None
                    and would_recommend
                    != context.get("recommended_plan_id")
                ),
                "note": (
                    "Uses each plan's existing capacity assignment; only the "
                    "vulnerability weight changes."
                ),
            }

        return {
            "run_id": equity.get("run_id"),
            "vulnerability_weight": lam,
            "equity_threshold": threshold,
            "vulnerability_weighted_unmet_need": round(vwun, 2),
            "normalized_vulnerability_weighted_unmet_need": round(
                normalized_vwun, 6
            ),
            "equity_gap": round(gap, 6) if gap is not None else None,
            "equity_gap_note": gap_note,
            "coverage_concentration_index": (
                round(concentration, 6) if concentration is not None else None
            ),
            "coverage_concentration_index_note": concentration_note,
            "demand_tract_count": len(impacts),
            "covered_tract_count": sum(
                1 for d in impacts if d.get("coverage", 0) >= 1.0
            ),
            "frontier_preview": frontier_preview,
            "interpretation": (
                "VWUN = Σ P_exposed·(1−C)·(1+λ·SVI) for the "
                "recommended plan (P_exposed already includes the "
                "flooded fraction); equity_gap = mean coverage of "
                "high-SVI tracts "
                "minus low-SVI tracts (negative = most vulnerable "
                "underserved); coverage_concentration_index uses continuous "
                "population-weighted SVI ranks (positive = coverage favors "
                "higher-SVI tracts). frontier_preview: which plan wins the "
                "whole frontier at this lambda."
            ),
        }

    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index.html")

    # Bind to loopback by default: the HITL approval and assessment
    # endpoints must not be LAN-exposed. Set HOST=0.0.0.0 explicitly
    # (at your own risk) for LAN demos.
    host = os.environ.get("HOST", "127.0.0.1")
    print(
        f"Starting Web server on http://{host}:{_port}"
        + ("  (set HOST=0.0.0.0 to expose on the LAN)" if host == "127.0.0.1" else "")
    )
    uvicorn.run(app, host=host, port=_port)

# ---------------------------------------------------------------------
# 4. CLI interactive loop
# ---------------------------------------------------------------------
async def cli_main():
    load_dotenv()
    print("=" * 72)
    print("Disaster Spatial Agent — extensible hazard decider (flood plugin)")
    print("=" * 72)
    print("Type 'quit' to exit.")
    while True:
        task = input("\nUser > ").strip()
        if task.lower() in {"quit", "exit"}:
            break
        if not task:
            continue
        try:
            await run_once(task)
        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        start_web_server()
    else:
        asyncio.run(cli_main())
