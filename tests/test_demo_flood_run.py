# tests/test_demo_flood_run.py
#
# Offline end-to-end demo: the full FloodSkill pipeline (routing ->
# geocoding -> station verification -> multi-source fusion -> GEE
# flood extent -> areal-weighted exposure -> SVI -> resource
# optimization + equity ledger -> CDRI decision indices) runs
# against FakeMCP with no network access.
#
# Also a runnable demo of the disaster decider: swap the fixtures to
# replay any historical event or city. Run:
#   python -m pytest tests/test_demo_flood_run.py -v -s

import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.skills import FloodSkill, MasterRouter            # noqa: E402
from app.hitl import AdaptiveHITL                          # noqa: E402
from app.models import RunState                            # noqa: E402
from app.verification import Verifier                      # noqa: E402
from app.experiment import ExperimentLogger                # noqa: E402


# FakeMCP / FUSION_SOURCES / NOW live in app.demo_fixtures.py,
# shared by web demo mode and the regression tests.
from app.demo_fixtures import (          # noqa: E402
    FUSION_SOURCES,
    FakeMCP,
    NOW,
    _polygon_geojson,
)
@pytest.mark.asyncio
async def test_flood_decider_demo_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.delenv("NWS_WARNING_RADIUS_KM", raising=False)
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")
    monkeypatch.setenv(
        "RESOURCE_DISCOVERY_ARGUMENTS_JSON",
        json.dumps({"latitude": "{latitude}", "longitude": "{longitude}"}),
    )
    monkeypatch.setenv(
        "RESOURCE_ALLOCATION_OBJECTIVES_JSON",
        json.dumps([
            {"name": "risk_reduction", "direction": "maximize"},
            {"name": "coverage", "direction": "maximize"},
            {"name": "response_time", "direction": "minimize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "unmet_demand", "direction": "minimize"},
        ]),
    )
    monkeypatch.setenv(
        "RESOURCE_OBJECTIVE_WEIGHTS_JSON",
        json.dumps({
            "risk_reduction": 0.35, "coverage": 0.25,
            "response_time": 0.20, "cost": 0.10, "unmet_demand": 0.10,
        }),
    )
    monkeypatch.setenv("RESOURCE_MAX_PLAN_COMBINATIONS", "1000")
    # Coverage radius 2 km: the east hospital covers the
    # high-vulnerability tract and the west one the low-vulnerability
    # tract, letting the equity objective separate the two plans
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")

    # 1) Routing (plugin registry)
    router = MasterRouter()
    hazard = router.classify("assess flooding near Friendswood")
    target = router.extract_target("assess flooding near Friendswood", hazard)
    assert hazard == "flood"
    assert target == "Friendswood"

    # 2) Full pipeline
    state = RunState(run_id="demo-run")
    mcp = FakeMCP()
    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        target, station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
    )

    # 3) Assertions: the pipeline completes and stays honest
    assert result.status == "completed", (
        f"demo run failed: {result.status}; "
        f"issues: {[i.message for i in result.validation_issues]}"
    )

    # Multi-source fusion actually happened
    called_tools = {t for t, _ in mcp.calls}
    for expected in (
        "geocode_location", "get_station_metadata", "get_flood_observation",
        "get_nwps_gauge", "get_flood_extent", "get_social_vulnerability",
        "get_available_resources",
    ):
        assert expected in called_tools, f"missing tool call: {expected}"

    # Decision indices present with areal-weighted markers
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    assert attrs, "decision_indices evidence missing"
    assert attrs["inputs"]["hazard_basis"] == "water_severity"
    assert attrs["inputs"]["extent_ratio"] is not None
    assert attrs["data_gaps"] == []
    assert not any(
        "precipitation" in action.lower() and "obtain" in action.lower()
        for action in result.next_actions
    )
    # All components computed -> no degraded suffix on the label
    assert "(degraded" not in attrs["cdri_risk_label"]
    # City-boundary clipping works: the analysis area uses the
    # administrative boundary, not a circular buffer
    assert attrs["inputs"]["analysis_area_basis"] == "city_boundary"
    # Numeric intervals: affected population carries a method
    # interval, CDRI an envelope
    assert attrs["inputs"]["affected_population_interval"] is not None
    lo, hi = attrs["uncertainty"]["interval"]
    assert lo <= attrs["cdri"] <= hi
    # Authority tiers actually match, so data_confidence is not
    # dragged down by unknown-source 0.5s
    assert attrs["data_confidence"] >= 0.78
    # Time-alignment ledger present with no mismatches in realtime mode
    _ta = None
    for ev in result.evidence:
        if (ev.attributes or {}).get("time_alignment"):
            _ta = ev.attributes["time_alignment"]
    assert _ta and _ta["mode"] == "realtime"
    assert _ta["mismatched_sources"] == []
    # Affected population is areal-weighted (partial overlap is not
    # whole-tract population)
    assert attrs["inputs"]["affected_population_source"] == (
        "census_tract_areal_weighted"
    )
    # Both tracts partially overlap the flood polygon -> the total
    # stays in the open interval (0, 10000), neither zero nor any
    # whole-tract population (centroid counting would credit all
    # 4000 of tract1)
    assert 0 < attrs["inputs"]["affected_population"] < 10000

    # Resource optimization + equity ledger (VWUN and equity gap
    # actually computed)
    opt_attrs = {}
    for ev in result.evidence:
        if (ev.attributes or {}).get("optimization_status"):
            opt_attrs = ev.attributes
    assert opt_attrs.get("optimization_status") == "optimized"
    ledger = opt_attrs.get("equity_ledger") or {}
    assert ledger.get("demand_tract_count") == 2
    # Response-only plans are not evacuation plans, and equal-capacity
    # hospital plans are dominated when a cheaper/faster shelter or mixed
    # plan serves the same number of people.
    curve = {
        e["plan_id"]: e
        for e in ledger.get("frontier_equity_curve", [])
    }
    assert set(curve) == {"plan_shelter_top1", "plan_mixed_1_each"}
    # Cost-VWUN trade-off ladder: the cheapest plan leaves the most unmet
    # need; the mixed plan supplies more eligible capacity.
    assert (
        curve["plan_shelter_top1"]["vulnerability_weighted_unmet_need"]
        > curve["plan_mixed_1_each"]["vulnerability_weighted_unmet_need"]
    )
    assert curve["plan_mixed_1_each"]["served_people"] == 200
    assert curve["plan_mixed_1_each"]["unmet_people"] > 0
    supply = ledger.get("supply_demand") or {}
    assert supply.get("status") == "allocated"
    assert supply.get("method") == (
        "capacity_constrained_min_cost_flow_geodesic_proxy"
    )
    assert supply.get("served_people") == 200
    assert supply.get("unmet_people") == 4237
    assert supply.get("travel_distance_p95_km", 0) > 0
    assert supply.get("transfer_time_estimate_range_minutes") is None
    assert supply.get("planning_gaps") == ["transfer_time_estimate"]
    assert all(
        item["assigned_people"] <= item["capacity"]
        for item in supply.get("facility_utilization", [])
    )
    assert ledger.get("vulnerability_weighted_unmet_need", -1) >= 0
    assert 0 <= ledger.get(
        "normalized_vulnerability_weighted_unmet_need", -1
    ) <= 1
    # Negative is now a meaningful result: the high-SVI tract receives
    # a smaller served share after capacity constraints are enforced.
    assert -1.0 <= ledger.get("equity_gap", -2) <= 1.0
    # VWUN formula regression (closed-form recomputation):
    # exposed_population already includes the flooded fraction, so
    # the formula must not multiply by flooded_fraction again
    from app.social_good import compute_vulnerability_weighted_unmet_need
    ctx0 = ledger.get("sensitivity_context") or {}
    impacts_by_plan = ctx0.get("frontier_plan_impacts") or {}
    _expected = round(
        compute_vulnerability_weighted_unmet_need(
            impacts_by_plan["plan_shelter_top1"], 1.0
        ),
        2,
    )
    assert curve["plan_shelter_top1"][
        "vulnerability_weighted_unmet_need"
    ] == _expected
    # Frontend recomputation uses the actual capacity assignments.
    ctx = ledger.get("sensitivity_context") or {}
    assert ctx.get("frontier_plan_impacts")
    assert len(ctx.get("frontier_plans") or []) == 2
    assert ctx.get("base_objective_weights")
    base_weights = dict(ctx["base_objective_weights"])
    if any(o["name"] == "vulnerability_coverage" for o in ctx["optimization_objectives"]):
        base_weights["vulnerability_coverage"] = (
            base_weights.get("vulnerability_coverage", 0.0)
            + ctx["vulnerability_weight"] * ctx["population_weighted_svi"]
        )
    total_weight = sum(base_weights.get(o["name"], 0.0) for o in ctx["optimization_objectives"])
    for objective in ctx["optimization_objectives"]:
        name = objective["name"]
        assert ctx["weights_used"][name] == pytest.approx(
            base_weights.get(name, 0.0) / total_weight, abs=2e-6
        )
    scenarios = opt_attrs.get("plan_scenarios") or []
    assert scenarios
    assert all(item["plan_id"] in curve for item in scenarios)

    # Equity objective genuinely participates (weight applied +
    # objective active)
    assert opt_attrs.get("svi_weight_applied") is True
    recommended = opt_attrs.get("recommended_plan") or {}
    assert (recommended.get("objectives") or {}).get(
        "vulnerability_coverage"
    ) is not None

    # 4) Demo output (printed for human inspection with -s)
    print("\n===== Disaster Decider Demo（洪水，离线）=====")
    print(f"target={target}  status={result.status}")
    print(f"CDRI={attrs['cdri_percent']}% ({attrs['cdri_risk_label']})  "
          f"EPS={attrs['eps']}  "
          f"confidence={attrs['data_confidence']}")
    print(f"hazard basis={attrs['inputs']['hazard_basis']}  "
          f"extent_ratio={attrs['inputs']['extent_ratio']}")
    print(f"affected_pop={attrs['inputs']['affected_population']} "
          f"({attrs['inputs']['affected_population_source']})")
    print(f"recommended plan={recommended.get('plan_id')}  "
          f"vuln_coverage="
          f"{(recommended.get('objectives') or {}).get('vulnerability_coverage')}")
    print(f"equity ledger={json.dumps(ledger, ensure_ascii=False)}")
    print(f"spatial objects={len(result.spatial_objects)}  "
          f"evidence={len(result.evidence)}  "
          f"tool calls={len(mcp.calls)}")
    print("=============================================\n")


# ----------------------------------------------------------------------
# Degradation scenario 1: USGS station metadata 503 (Gallatin Gateway
# case) -- degrade gracefully instead of failing outright
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_station_metadata_outage_degrades_gracefully(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")

    state = RunState(run_id="degrade-meta")
    mcp = FakeMCP()

    # Make station metadata fail as if it returned 503
    async def _meta_error(args):
        return json.dumps({
            "status": "error",
            "error": "HTTP 503 Service Unavailable (all 3 attempts)",
        })

    mcp._fx_get_station_metadata = _meta_error

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood",
    )

    # Metadata failure is not assessment failure: observations carry
    # their own coordinates and the pipeline completes
    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "STATION_METADATA_UNAVAILABLE" in codes
    # Audit chain honestly marks the location as not independently verified
    events = {e.get("event"): e for e in state.events}
    assert "station_metadata_unavailable" in events
    assert events["station_spatial_verification"]["location_verified"] is False


# ----------------------------------------------------------------------
# Degradation scenario 2: no satellite flood extent (White Springs
# case) -- report it explicitly instead of staying silent
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_flood_extent_explicitly_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")

    state = RunState(run_id="no-extent")
    mcp = FakeMCP()

    # GEE: no flooding detected (null area, no geojson)
    async def _gee_dry(args):
        return json.dumps({
            "status": "ok",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False,
                "latest_post_scene": "2026-08-28T23:59:00Z",
                "pre_scene_count": 3, "post_scene_count": 2,
            },
            "spatial_extent": {
                "flooded_area_km2": None, "geojson": None,
            },
            "data_quality": {
                "geometry_returned": False,
                "area_reduction_available": False,
            },
        })

    mcp._fx_get_flood_extent = _gee_dry

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood",
    )

    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    # Policy since 2026-09-26: satellite "nothing detected" no longer
    # cancels the spatial picture when the verified gauge crossed
    # action stage (8.0 >= 7.0 here). A MODELED stage-buffer extent
    # substitutes and every surface discloses it.
    assert "EXTENT_FALLBACK_MODELED" in codes
    assert any(
        "MODELED" in a and "stage-buffer" in a
        for a in result.next_actions
    ), result.next_actions
    assert (state.gis_results.get("stats") or {}).get(
        "extent_provenance"
    ) == "modeled_stage_buffer"
    assert (state.gis_results or {}).get("map_path")
    # Decision indices record the degradation honestly
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    assert attrs["inputs"]["hazard_basis"] == (
        "water_severity"
    )
    # The modeled area feeds spatial analysis, while the missing observed
    # extent remains an explicit gap and the hazard stays water-driven.
    assert "flooded_area_km2" not in attrs["data_gaps"]
    assert "observed_flood_extent" in attrs["data_gaps"]
    assert "(degraded:" in attrs["cdri_risk_label"]
    assert attrs["inputs"]["extent_ratio"] is not None


# ----------------------------------------------------------------------
# Degradation scenario 3: station without an NWPS flood-category
# profile (Manhattan Kansas case) -- many valid USGS stations lack
# one; degrade instead of failing outright
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nwps_gauge_missing_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")

    state = RunState(run_id="no-nwps")
    mcp = FakeMCP()

    async def _no_nwps(args):
        return json.dumps({
            "status": "error",
            "error": "NWPS could not find USGS ID",
        })

    mcp._fx_get_nwps_gauge = _no_nwps

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Manhattan Kansas", station_id="06879650",
        raw_task="assess flooding near Manhattan, Kansas using USGS 06879650",
    )

    # Missing NWPS profile is not assessment failure: gauge
    # observation, fusion, and decision indices proceed normally
    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "NWPS_GAUGE_UNAVAILABLE" in codes
    events = {e.get("event") for e in state.events}
    assert "nwps_gauge_unavailable" in events
    # stageflow is skipped too (no gauge_id to query)
    assert "nwps_stageflow_unavailable" not in events  # skipped directly, not failed
    # Decision indices still present (unexplainable severity -> hazard
    # is the water ratio only)
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    assert attrs, "decision_indices missing despite NWPS outage"
    assert any(
        "flood-category thresholds" in a for a in result.next_actions
    )


# ----------------------------------------------------------------------
# Degradation scenario 4: station distance exceeded (disambiguation
# residue / multi-station network) -- warn, and the gate refuses
# optimization
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_far_station_warns_but_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")

    state = RunState(run_id="far-station")
    mcp = FakeMCP()

    # Target geocodes to Seattle (station fixture ~3000 km away in
    # Houston)
    async def _geocode_seattle(args):
        return f"{args['place_name']} 的坐标: 纬度 47.6062, 经度 -122.3321"

    mcp._fx_geocode_location = _geocode_seattle

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Manhattan", station_id="06879650",
        raw_task="assess flooding near Manhattan using USGS 06879650",
    )

    # The assessment completes with a warning; resource optimization
    # is refused by the evidence gate
    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "STATION_TARGET_DISTANCE_EXCEEDED" in codes
    opt_attrs = {}
    for ev in result.evidence:
        if (ev.attributes or {}).get("optimization_status"):
            opt_attrs = ev.attributes
    assert opt_attrs.get("optimization_status") in (
        None, "blocked_by_evidence_gate",
    ) or opt_attrs.get("optimization_status") != "optimized"


# ----------------------------------------------------------------------
# Feature regressions: uncertainty envelope / conflict quarantine /
# frontier equity curve / lambda recomputation
# ----------------------------------------------------------------------
def test_uncertainty_band_in_indices():
    """CDRI decision indices must carry a +/-0.1 component envelope
    and the top-2 sensitive components."""
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ))
    from app.engine import RiskEngine

    result = RiskEngine().compute_decision_indices(**_risk_inputs_like_demo())
    unc = result["uncertainty"]
    lo, hi = unc["interval"]
    assert lo <= result["cdri"] <= hi
    assert set(unc["sensitivity"]) == {
        "hazard", "exposure", "vulnerability"
    }
    assert len(unc["top_components"]) == 2
    assert unc["top_components"] == sorted(
        unc["sensitivity"], key=unc["sensitivity"].get, reverse=True
    )[:2]


def _risk_inputs_like_demo(**overrides):
    inputs = dict(
        water_level=8.0, action_stage=7.0, major_stage=15.0,
        flooded_area_km2=18.6, city_area_km2=100.0,
        fallback_analysis_radius_km=10.0,
        social_vulnerability={
            "status": "ok",
            "profile": {
                "population_weighted_svi": 0.6,
                "total_population": 10000,
            },
            "tracts": [],
        },
        gis_stats={"affected_population": 4437,
                   "affected_population_method": "areal_weighted",
                   "affected_facilities": 30, "facility_inventory_count": 60, "affected_roads": 0, "travel_time_min": None},
        fused_measurements={"facility_count": {"value": 60},
                            "road_count": {"value": 42}},
        fusion_sources=[{"tool": "get_road_status",
                         "arguments": {"radius_km": 10}}],
        observations=[{"quality_score": 0.8}, {"quality_score": 0.6}],
    )
    inputs.update(overrides)
    return inputs


def test_conflict_quarantine_in_fusion():
    """Multi-source dispersion above threshold on one variable:
    conflict=True plus per-source raw values."""
    import asyncio
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ))
    from app.engine.flood_fusion_engine import fuse_flood_evidence

    def obs(source, value):
        return {
            "source": source, "source_type": "hydrology",
            "tool": "get_flood_observation",
            "raw": json.dumps({
                "status": "ok",
                "measurements": {"water_level": {"value": value, "unit": "ft"}},
                "observed_at": "2026-08-29T12:00:00+00:00",
            }),
        }

    fused = fuse_flood_evidence([
        obs("usgs_a", 3.0), obs("nwps_b", 9.0),   # conflict ratio 6/9=0.67 > 0.5
    ])
    m = fused["fused_measurements"]["water_level"]
    assert m["conflict"] is True
    assert m["conflict_ratio"] > 0.5
    assert m["source_values"] == {"usgs_a": 3.0, "nwps_b": 9.0}

    calm = fuse_flood_evidence([
        obs("usgs_a", 5.0), obs("nwps_b", 5.4),    # conflict ratio 0.4/5.4 < 0.5
    ])
    m2 = calm["fused_measurements"]["water_level"]
    assert m2["conflict"] is False
    assert "source_values" not in m2


@pytest.mark.asyncio
async def test_frontier_equity_curve(tmp_path, monkeypatch):
    """Full-frontier equity curve: every non-dominated plan carries
    cost / VWUN / covered community lists."""
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")
    monkeypatch.setenv(
        "RESOURCE_DISCOVERY_ARGUMENTS_JSON",
        json.dumps({"latitude": "{latitude}", "longitude": "{longitude}"}),
    )
    monkeypatch.setenv(
        "RESOURCE_ALLOCATION_OBJECTIVES_JSON",
        json.dumps([
            {"name": "risk_reduction", "direction": "maximize"},
            {"name": "coverage", "direction": "maximize"},
            {"name": "response_time", "direction": "minimize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "unmet_demand", "direction": "minimize"},
        ]),
    )
    monkeypatch.setenv(
        "RESOURCE_OBJECTIVE_WEIGHTS_JSON",
        json.dumps({
            "risk_reduction": 0.35, "coverage": 0.25,
            "response_time": 0.20, "cost": 0.10, "unmet_demand": 0.10,
        }),
    )
    monkeypatch.setenv("RESOURCE_MAX_PLAN_COMBINATIONS", "1000")
    # Coverage radius 2 km: each plan covers one tract (a 10 km
    # full-coverage radius leaves equity_gap undefined with only one
    # grouped side, skipping frontier entries)
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")

    state = RunState(run_id="frontier")
    mcp = FakeMCP()
    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood",
    )
    assert result.status == "completed"

    ledger = {}
    for ev in result.evidence:
        _l = (ev.attributes or {}).get("equity_ledger")
        if isinstance(_l, dict):
            ledger = _l
    curve = ledger.get("frontier_equity_curve") or []
    # Only feasible, non-dominated evacuation plans enter the curve.
    assert len(curve) == 2
    for entry in curve:
        assert entry["plan_id"]
        assert entry["cost"] is not None
        assert entry["vulnerability_weighted_unmet_need"] >= 0
        assert 0 <= entry[
            "normalized_vulnerability_weighted_unmet_need"
        ] <= 1
        assert isinstance(entry["covered_tract_ids"], list)
        assert isinstance(entry["covered_high_svi_tract_ids"], list)
    # Ledger also carries the recommended plan's demand_impacts
    # (source data for lambda recomputation endpoints)
    assert isinstance(ledger.get("demand_impacts"), list)
    assert ledger.get("vulnerability_weight") == 1.0


# ----------------------------------------------------------------------
# Regressions: authority tier matching / HITL safe defaults /
# strategy gate / historical replay time alignment
# ----------------------------------------------------------------------
def test_authority_tier_matching():
    """Free-text sources must match an authority tier; unknown
    sources stay at 0.5."""
    from app.evidence_quality import _authority

    assert _authority("USGS Water Services via Flood Alert MCP") == (
        "authoritative_government", 1.0,
    )
    assert _authority("NWS")[1] == 1.0
    # Sentinel-derived flood extents are our own change-detection
    # product on an authoritative platform: revisit latency (6-12 d)
    # makes acquisition timing uncertain vs a flood peak and single-pair
    # change detection has known false positives, so it tiers BELOW
    # operational gauge/hydrology sources (2026-09-26 rework).
    assert _authority("Google Earth Engine / Sentinel-1 SAR") == (
        "authoritative_platform_derived_product", 0.6,
    )
    assert _authority("CDC/ATSDR SVI 2022")[1] == 1.0
    assert _authority("OSM Overpass + MANUAL capacity table")[1] == 0.8
    # Genuinely unknown sources still get 0.5 -- tiers are not
    # arbitrary weights
    assert _authority("some random vendor feed") == ("unknown", 0.5)


@pytest.mark.asyncio
async def test_hitl_timeout_safe_default(tmp_path):
    """Safety-critical checkpoints resolve toward refusal on timeout
    or when HITL is disabled -- never auto-approve."""
    import asyncio

    state = RunState(run_id="hitl-timeout")
    os.environ["HITL_ENABLED"] = "true"
    os.environ["HITL_TIMEOUT_SECONDS"] = "0.05"
    try:
        hitl = AdaptiveHITL(state)
        hitl.enable_web_mode()
        answer = await hitl.ask_async(
            reason="Authorize full resource mobilization?",
            question="Authorize?",
            proposed_value="yes",
            timeout_value="no",
        )
        assert answer == "no"  # timeout = safe default (refuse), not the proposal

        # Non-safety checkpoint (no timeout_value): a timeout still
        # accepts the default parameters
        answer2 = await hitl.ask_async(
            reason="Parameter confirmation",
            question="Continue with defaults?",
            proposed_value={"lambda": 1.0},
        )
        assert answer2 == {"lambda": 1.0}
    finally:
        os.environ.pop("HITL_TIMEOUT_SECONDS", None)

    # HITL disabled = unattended: the safety-critical default still
    # refuses
    os.environ["HITL_ENABLED"] = "false"
    try:
        state2 = RunState(run_id="hitl-disabled")
        hitl2 = AdaptiveHITL(state2)
        answer3 = await hitl2.ask_async(
            reason="Authorize mobilization?",
            question="Authorize?",
            proposed_value="yes",
            timeout_value="no",
        )
        assert answer3 == "no"
    finally:
        os.environ["HITL_ENABLED"] = "true"


def test_strategy_selector_never_shortcircuits_with_evidence():
    """A calm single gauge must not short-circuit the LLM when SAR
    shows flooding or alerts exist -- a report must be generated."""
    from app.agent import StrategySelector

    sel = StrategySelector()
    # Water ratio 0.4 < 0.5, but flooded area exists -> no short-circuit
    decision = sel.select(
        water_level=2.8, action_stage=7.0,
        flooded_area_km2=18.6, alert_count=0,
    )
    assert decision.skip_llm is False
    # Calm water but active alerts exist -> no short-circuit
    decision = sel.select(
        water_level=2.8, action_stage=7.0,
        flooded_area_km2=None, alert_count=2,
    )
    assert decision.skip_llm is False
    # Truly quiet (no flooding, no alerts, low water) -> keep the
    # cost short-circuit
    decision = sel.select(
        water_level=2.8, action_stage=7.0,
        flooded_area_km2=None, alert_count=0,
    )
    assert decision.skip_llm is True


async def _run_skill(mcp, state, tmp_path, overrides=None):
    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    return await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
        overrides=overrides,
    )


def _env_defaults(monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv(
        "FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES)
    )
    monkeypatch.setenv("RESOURCE_DISCOVERY_TOOL", "get_available_resources")
    monkeypatch.setenv(
        "RESOURCE_DISCOVERY_ARGUMENTS_JSON",
        json.dumps({"latitude": "{latitude}", "longitude": "{longitude}"}),
    )
    monkeypatch.setenv(
        "RESOURCE_ALLOCATION_OBJECTIVES_JSON",
        json.dumps([
            {"name": "risk_reduction", "direction": "maximize"},
            {"name": "coverage", "direction": "maximize"},
            {"name": "response_time", "direction": "minimize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "unmet_demand", "direction": "minimize"},
        ]),
    )
    monkeypatch.setenv(
        "RESOURCE_OBJECTIVE_WEIGHTS_JSON",
        json.dumps({
            "risk_reduction": 0.35, "coverage": 0.25,
            "response_time": 0.20, "cost": 0.10, "unmet_demand": 0.10,
        }),
    )
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")


@pytest.mark.asyncio
async def test_historical_replay_time_alignment(tmp_path, monkeypatch):
    """Historical event replay (2017-08-27) with strict time
    alignment.

    - Gauge queried over the event-day window (USGS startDT/endDT)
    - Realtime-only sources (alerts/forecast/weather/precipitation)
      are not collected
    - SAR acquisition inside the event window [-1, +6] days passes
      the gate
    - Static layers (Census/OSM/SVI) marked time_invariant
    """
    _env_defaults(monkeypatch)

    async def _gee_2017(args):
        return json.dumps({
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False,
                "latest_post_scene": "2017-08-28T23:59:00Z",
                "pre_scene_count": 3, "post_scene_count": 2,
                "pre_acquisition_times": ["2017-08-20T00:26:15Z"],
                "post_acquisition_times": ["2017-08-28T23:59:00Z"],
            },
            "spatial_extent": {
                "flooded_area_km2": 24.1,
                "geojson": _polygon_geojson(-95.115, 29.51, d=0.03),
            },
            "analysis": {"threshold_method": "fixed", "threshold_db": -3.0},
            "data_quality": {"geometry_returned": True},
        })

    state = RunState(run_id="harvey-replay")
    mcp = FakeMCP()
    mcp._fx_get_flood_extent = _gee_2017

    result = await _run_skill(
        mcp, state, tmp_path, overrides={"event_date": "2017-08-27"}
    )

    assert result.status == "completed", result.summary
    called = {t for t, _ in mcp.calls}

    # 1) Hydrology queries carry the event window (historical replay
    # is not the latest instantaneous value)
    hydro_calls = [a for t, a in mcp.calls if t == "get_flood_observation"]
    assert hydro_calls, "gauge never queried"
    assert all(
        a.get("start_dt") == "2017-08-27T00:00:00Z" for a in hydro_calls
    ), hydro_calls

    # 2) Realtime-only sources are not collected (prefer missing data
    # over introducing a time mismatch)
    for current_only in (
        "get_weather_observations", "get_precipitation",
        "get_forecast", "get_flood_warnings",
    ):
        assert current_only not in called, current_only
    assert "get_nwps_stageflow" not in called  # no historical archive endpoint

    # 3) Time-alignment ledger: historical mode, no mismatches, static
    # layers honestly labeled
    ta = None
    for ev in result.evidence:
        if (ev.attributes or {}).get("time_alignment"):
            ta = ev.attributes["time_alignment"]
    assert ta and ta["mode"] == "historical"
    assert ta["mismatched_sources"] == []
    assert {
        i["source_type"] for i in ta["items"]
        if i["status"] == "time_invariant"
    } >= {"exposure", "infrastructure", "road"}
    skipped_names = {
        s["source"] for s in ta["skipped_current_only_sources"]
    }
    assert skipped_names == {
        "weather", "precipitation", "forecast", "flood_warnings",
    }

    # 4) Flood extent passes the [-1, +6] gate -> two-component hazard
    # plus the full downstream analysis
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    assert attrs["inputs"]["hazard_basis"] == "water_severity"
    # 4b) Historical replay judges the event by its window PEAK, not
    # the window-end snapshot (Lodi NJ 2026-09-13 case); the primary
    # evidence carries the full hydrograph summary.
    main_ev = next(
        ev for ev in result.evidence
        if str(ev.evidence_id).startswith("usgs_station_")
    )
    assert main_ev.attributes["observation_semantics"] == "window_peak"
    assert main_ev.attributes["window_summary"]["peak_stage_ft"] == 8.0
    assert main_ev.attributes["window_summary"]["end_stage_ft"] == 6.2
    assert "peaked at" in result.summary
    # Satellite-observed provenance flag (not the modeled fallback)
    assert (state.gis_results.get("stats") or {}).get(
        "extent_provenance"
    ) == "satellite_sar"
    # 5) Replay freshness is judged against the event window: quality
    # is not penalized by old timestamps or the current date
    assert attrs["data_confidence"] >= 0.6
    # 6) Retrospective resource optimization still works (training /
    # replay scenarios); the equity ledger is intact
    opt_attrs = {}
    for ev in result.evidence:
        if (ev.attributes or {}).get("optimization_status"):
            opt_attrs = ev.attributes
    assert opt_attrs.get("optimization_status") == "optimized"
    assert any(
        "Historical replay" in a for a in result.next_actions
    )


@pytest.mark.asyncio
async def test_sar_stale_gate_blocks_misaligned_extent(tmp_path, monkeypatch):
    """Historical event + today's SAR scene -> the stale gate rejects,
    and the mismatch is reported."""
    _env_defaults(monkeypatch)

    async def _gee_today(args):
        # Acquisition time = now (9 years off the 2017 event). The
        # default fixture is date-aware, so the stale path needs this
        # explicit injection.
        _acq = datetime.now(timezone.utc).isoformat()
        return json.dumps({
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False,
                "latest_post_scene": _acq,
                "pre_scene_count": 3, "post_scene_count": 2,
                "pre_acquisition_times": [_acq],
                "post_acquisition_times": [_acq],
            },
            "spatial_extent": {
                "flooded_area_km2": 18.6,
                "geojson": _polygon_geojson(-95.115, 29.51, d=0.03),
            },
            "analysis": {"threshold_method": "fixed", "threshold_db": -3.0},
            "data_quality": {"geometry_returned": True},
        })

    state = RunState(run_id="stale-sar")
    mcp = FakeMCP()
    mcp._fx_get_flood_extent = _gee_today

    result = await _run_skill(
        mcp, state, tmp_path, overrides={"event_date": "2017-08-27"}
    )

    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "SAR_EXTENT_STALE" in codes
    # Policy since 2026-09-26 (prefer over-alert over silent miss): a
    # stale satellite layer is discarded, but when the verified gauge
    # peak reached action stage the spatial products continue on a
    # MODELED stage-buffer extent, fully disclosed.
    assert "EXTENT_FALLBACK_MODELED" in codes
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    # Stale satellite area must not enter CDRI: hazard stays the water
    # dimension only (extent is spatial context, never a hazard factor)
    assert attrs["inputs"]["hazard_basis"] == (
        "water_severity"
    )
    # The modeled extent substitutes for spatial analysis, but the absent
    # observation remains a disclosed gap.
    assert "flooded_area_km2" not in attrs["data_gaps"]
    assert "observed_flood_extent" in attrs["data_gaps"]
    assert "(degraded:" in attrs["cdri_risk_label"]
    assert attrs["inputs"]["extent_ratio"] is not None
    assert (state.gis_results.get("stats") or {}).get(
        "extent_provenance"
    ) == "modeled_stage_buffer"
    extent_obj = {
        o.object_id: o for o in result.spatial_objects
    }.get("flood_inundation_extent")
    assert extent_obj is not None
    assert extent_obj.attributes["extent_provenance"] == (
        "modeled_stage_buffer"
    )
    assert "Modeled stage-buffer" in extent_obj.source
    # Downstream GIS products run on the modeled layer (the mismatched
    # SATELLITE layer still must not drive rescue routes)
    assert (state.gis_results or {}).get("map_path")
    model = (state.gis_results.get("stats") or {}).get("extent_model")
    assert model and model["radius_basis"] == "nwps_category_anchored"


@pytest.mark.asyncio
async def test_zero_incity_overlap_keeps_buffer_context_layer(
    tmp_path, monkeypatch,
):
    """When the detected water body has zero intersection with the
    Friendswood city boundary, no "flood" is drawn, but detected water
    within the 30 km buffer stays as a gray context layer plus a
    SAR_EXTENT_NO_INCITY_OVERLAP warning (never silent)."""
    _env_defaults(monkeypatch)

    async def _geocode(args):
        return f"{args['place_name']} 的坐标: 纬度 29.5294, 经度 -95.2010"

    async def _offcity_extent(args):
        # Footprint center outside the fixture city square
        # (-95.25..-94.95, 29.35..29.65)
        return json.dumps({
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False,
                "latest_post_scene": NOW,
                "pre_scene_count": 3, "post_scene_count": 2,
                "pre_acquisition_times": [NOW],
                "post_acquisition_times": [NOW],
            },
            "spatial_extent": {
                "flooded_area_km2": 129.0,
                "geojson": _polygon_geojson(-95.55, 29.85, d=0.10),
            },
            "data_quality": {"geometry_returned": True},
        })

    state = RunState(run_id="zero-incity")
    mcp = FakeMCP()
    mcp._fx_geocode_location = _geocode
    mcp._fx_get_flood_extent = _offcity_extent

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
    )

    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "SAR_EXTENT_NO_INCITY_OVERLAP" in codes
    # Context layer file produced (frontend gray-layer data source)
    assert (state.gis_results or {}).get("context_extent_path")
    # CDRI still comes from the water-level dimension (0.88 ft
    # in-bank -> 0)
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    assert attrs["cdri_percent"] is not None


@pytest.mark.asyncio
async def test_uncorroborated_satellite_water_is_not_a_flood_extent(
    tmp_path, monkeypatch,
):
    """An in-bank gauge plus no flood warning cannot promote a raw SAR
    change footprint into an operational flood map."""
    _env_defaults(monkeypatch)

    async def _geocode(args):
        return f"{args['place_name']} 的坐标: 纬度 29.5294, 经度 -95.2010"

    async def _low_stage(args):
        return json.dumps({
            "status": "ok",
            "observation": {
                "station_id": "08077600", "water_level": 1.5, "unit": "ft",
                "observation_time": NOW, "source": "USGS",
                "latitude": 29.5175, "longitude": -95.1785,
                "metadata_verified": True,
            },
        })

    async def _big_extent(args):
        return json.dumps({
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False,
                "latest_post_scene": NOW,
                "pre_scene_count": 3, "post_scene_count": 2,
                "pre_acquisition_times": [NOW],
                "post_acquisition_times": [NOW],
            },
            "spatial_extent": {
                "flooded_area_km2": 17.4,
                "geojson": _polygon_geojson(-95.115, 29.51, d=0.08),
            },
            "data_quality": {"geometry_returned": True},
        })

    async def _no_warnings(args):
        return json.dumps({
            "status": "ok",
            "source": "NWS",
            "alerts": [],
        })

    state = RunState(run_id="inbank-contradiction")
    mcp = FakeMCP()
    mcp._fx_geocode_location = _geocode
    mcp._fx_get_flood_observation = _low_stage
    mcp._fx_get_flood_extent = _big_extent
    mcp._fx_get_flood_warnings = _no_warnings

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
    )

    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "SAR_EXTENT_UNCORROBORATED" in codes
    attrs = {}
    for ev in result.evidence:
        if "decision_indices" in (ev.attributes or {}):
            attrs = ev.attributes["decision_indices"]
    # Water dimension 0 (1.5 < action 7) -> hazard and CDRI go to zero
    assert attrs["inputs"]["water_severity"] == 0.0
    assert attrs["components"]["hazard"] == 0.0
    assert attrs["cdri"] == 0.0
    # The raw candidate stays auditable but cannot enter flood severity,
    # exposure, routing, or the map.
    assert attrs["inputs"]["extent_severity"] is None
    assert "flooded_area_km2" in attrs["data_gaps"]
    assert not (state.gis_results or {}).get("flood_boundary_path")
    assert not (state.gis_results or {}).get("map_path")
    assert (state.gis_results.get("stats") or {}).get(
        "unverified_surface_water_area_km2"
    ) is not None


@pytest.mark.asyncio
async def test_poi_search_failure_disclosed(tmp_path, monkeypatch):
    """A failed POI search must surface as POI_SEARCH_FAILED --
    silently skipping the layer makes users read "missing layer" as
    "no facilities available"."""
    _env_defaults(monkeypatch)

    async def _poi_fail(args):
        return json.dumps({
            "status": "error",
            "error": "Overpass unavailable (all mirrors)",
        })

    state = RunState(run_id="poi-fail")
    mcp = FakeMCP()
    mcp._fx_poi_search_osm = _poi_fail

    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
    )

    assert result.status == "completed", result.summary
    codes = {i.code for i in result.validation_issues}
    assert "POI_SEARCH_FAILED" in codes


@pytest.mark.asyncio
async def test_facility_search_uses_30_then_60_minute_service_areas(
    tmp_path, monkeypatch
):
    """At 30 km/h the facility query uses 15 km first and expands to
    30 km only when the primary service area is valid but empty."""
    _env_defaults(monkeypatch)
    monkeypatch.setenv("GIS_FACILITY_PLANNING_SPEED_KMH", "30")
    monkeypatch.setenv("GIS_FACILITY_PRIMARY_SERVICE_MINUTES", "30")
    monkeypatch.setenv("GIS_FACILITY_EXTENDED_SERVICE_MINUTES", "60")

    state = RunState(run_id="facility-service-areas")
    mcp = FakeMCP()
    default_poi = mcp._fx_poi_search_osm
    facility_radii = []

    def _poi_by_service_area(args):
        if "building" in str(args.get("amenity_types", "")):
            return default_poi(args)
        facility_radii.append(args["radius_m"])
        if len(facility_radii) == 1:
            return {
                "status": "ok",
                "output_path": str(tmp_path / "empty-primary.geojson"),
                "feature_list": [],
                "total_found": 0,
            }
        return default_poi(args)

    mcp._fx_poi_search_osm = _poi_by_service_area
    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / "traj.jsonl")),
    )
    result = await skill.run(
        "Friendswood", station_id="08077600",
        raw_task="assess flooding near Friendswood using USGS 08077600",
    )

    assert result.status == "completed", result.summary
    assert facility_radii == [15000.0, 30000.0]
    stats = state.gis_results["stats"]
    assert stats["poi_search_tier"] == "extended_60_minute"
    assert stats["poi_search_radius_km"] == 30.0
    assert stats["facility_primary_service_radius_km"] == 15.0
    assert stats["facility_extended_service_radius_km"] == 30.0
    assert state.gis_results.get("rescue_route_path")


@pytest.mark.asyncio
async def test_demo_mode_guarantees_slider_data(tmp_path, monkeypatch):
    """Web demo contract: a one-click run must produce the equity
    ledger plus flip-preview context (the lambda slider's data) in
    both realtime and historical replay modes."""
    _env_defaults(monkeypatch)

    from app.demo_fixtures import run_demo_assessment

    for event_date in (None, "2017-08-27"):
        result, state = await run_demo_assessment(event_date=event_date)
        assert result.status == "completed", result.summary
        ledger = {}
        for ev in result.evidence:
            _l = (ev.attributes or {}).get("equity_ledger")
            if isinstance(_l, dict):
                ledger = _l
        # Slider essentials: actual assignment impacts and frontier data.
        assert isinstance(ledger.get("demand_impacts"), list) and ledger["demand_impacts"]
        ctx = ledger.get("sensitivity_context") or {}
        assert ctx.get("frontier_plan_impacts") and ctx.get("frontier_plans")
        assert ctx.get("optimization_objectives") and ctx.get("weights_used")
