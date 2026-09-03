# tests/test_engine_units.py
#
# Regression net: engine layer (pareto / flood_risk / fusion),
# MasterRouter registry routing, adaptive HITL gates, and
# FinalDecision output validation. Fully offline.
# Run: python -m pytest tests/test_engine_units.py -v

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.engine.pareto_engine as pareto_engine          # noqa: E402
from app.engine import RiskEngine                          # noqa: E402
from app.engine.flood_fusion_engine import (               # noqa: E402
    fuse_flood_evidence,
    normalize_fusion_observation,
)
from app.engine.geometry_engine import GeometryEngine      # noqa: E402
from app.skills.master_router import MasterRouter          # noqa: E402
from app.skills import registry as skill_registry          # noqa: E402
from app.hitl import AdaptiveHITL                          # noqa: E402
from app.models import RunState                            # noqa: E402
from app.agent import FinalDecisionAgent                    # noqa: E402
from app.skills.flood_skill import FloodSkill               # noqa: E402


# ------------------------------------------------------------------
# 1. pareto_engine
# ------------------------------------------------------------------
def test_pareto_frontier_dominance():
    plans = [
        {"plan_id": "a", "objectives": {"risk_reduction": 0.5, "cost": 10}},
        {"plan_id": "b", "objectives": {"risk_reduction": 0.8, "cost": 5}},
        {"plan_id": "c", "objectives": {"risk_reduction": 0.4, "cost": 20}},
    ]
    objectives = [
        {"name": "risk_reduction", "direction": "maximize"},
        {"name": "cost", "direction": "minimize"},
    ]
    frontier = pareto_engine.pareto_frontier(plans, objectives)
    ids = {p["plan_id"] for p in frontier}
    # b dominates a and c (higher risk_reduction at lower cost)
    assert ids == {"b"}


def test_select_best_pareto_plan_weights():
    frontier = [
        {"plan_id": "x", "objectives": {"risk_reduction": 0.9, "cost": 50}},
        {"plan_id": "y", "objectives": {"risk_reduction": 0.6, "cost": 10}},
    ]
    objectives = [
        {"name": "risk_reduction", "direction": "maximize"},
        {"name": "cost", "direction": "minimize"},
    ]
    best = pareto_engine.select_best_pareto_plan(
        frontier, objectives,
        {"risk_reduction": 1.0, "cost": 0.0},
    )
    assert best["plan_id"] == "x"
    best = pareto_engine.select_best_pareto_plan(
        frontier, objectives,
        {"risk_reduction": 0.0, "cost": 1.0},
    )
    assert best["plan_id"] == "y"


# ------------------------------------------------------------------
# 2. RiskEngine (CDRI / EPS / data confidence + data-gap accounting)
# ------------------------------------------------------------------
def _risk_inputs(**overrides):
    inputs = dict(
        water_level=5.0, action_stage=10.0, major_stage=None,
        flooded_area_km2=10.0, city_area_km2=100.0,
        fallback_analysis_radius_km=10.0,
        social_vulnerability={
            "status": "ok",
            "profile": {"population_weighted_svi": 0.5, "total_population": 1000},
            "tracts": [],
        },
        gis_stats={"affected_population": 500, "affected_facilities": 30, "travel_time_min": None},
        fused_measurements={"facility_count": {"value": 60}, "road_count": {"value": 0.0}},
        fusion_sources=[{"tool": "get_road_status", "arguments": {"radius_km": 10}}],
        observations=[{"quality_score": 0.8}, {"quality_score": 0.6}],
    )
    inputs.update(overrides)
    return inputs


def test_risk_engine_indices_match_formula():
    result = RiskEngine().compute_decision_indices(**_risk_inputs())
    # water_severity: no major stage -> action-stage fallback, clamp01(5/10)=0.5
    # extent_severity: ratio 0.1 / 0.25 saturation reference = 0.4
    # hazard = max(0.5, 0.4) = 0.5 (worst dimension, not averaged)
    # water: no major -> action reference x = 0.5, saturating map 0.5/1.5 = 1/3
    assert result["inputs"]["water_severity"] == pytest.approx(1/3, abs=1e-4)
    # area: physical share 0.1 (no amplification)
    assert result["inputs"]["extent_severity"] == pytest.approx(0.1, abs=1e-4)
    assert result["components"]["hazard"] == pytest.approx(1/3, abs=1e-4)
    # exposure: population share 0.5 and facility share 0.5 -> max = 0.5
    assert result["components"]["exposure"] == pytest.approx(0.5, abs=1e-4)
    # CDRI = hazard x (0.5 + 0.5*exposure) x v = 1/3 x 0.75 x 0.5 = 0.125
    assert result["cdri"] == pytest.approx(0.125, abs=1e-6)
    assert result["cdri_risk_label"] == "Very High"
    assert "response_capacity" in result["components"]
    # EPS: water_sat 1/3 x (1+0.5)/2 x (1+0.5)/2 = 0.1875
    assert result["eps"] == pytest.approx(0.1875, abs=1e-4)
    assert result["data_confidence"] == pytest.approx(0.715, abs=1e-6)


def test_risk_engine_major_referenced_water_severity():
    """Major-stage reference: with action=7 and major=21, a record
    23 ft crest scores far above a crest just past action (8 ft) --
    the water-level dimension stays monotonic under record floods."""
    base = dict(city_area_km2=None)  # no extent -> no area dimension, hazard = water
    crest = RiskEngine().compute_decision_indices(
        **_risk_inputs(
            water_level=23.3, action_stage=7.0, major_stage=21.0,
            flooded_area_km2=None, **base,
        )
    )
    over_action = RiskEngine().compute_decision_indices(
        **_risk_inputs(
            water_level=8.0, action_stage=7.0, major_stage=15.0,
            flooded_area_km2=None, **base,
        )
    )
    # Continuous saturation (no clamping): 16% over major ->
    # x/(1+x) = 1.164/2.164 ~= 0.538, strictly above the
    # just-at-action 0.125/1.125 ~= 0.111 -- monotonic, never capped
    _x_crest = (23.3 - 7.0) / (21.0 - 7.0)
    assert crest["components"]["hazard"] == pytest.approx(
        _x_crest / (1.0 + _x_crest), abs=1e-4
    )
    _x_over = 0.125
    assert over_action["components"]["hazard"] == pytest.approx(
        _x_over / (1.0 + _x_over), abs=1e-4
    )
    assert crest["cdri"] > over_action["cdri"]


def test_risk_engine_data_gaps_accounting():
    # Missing satellite extent: area must not masquerade as a measured
    # 0 -- hazard degrades to water severity only
    result = RiskEngine().compute_decision_indices(**_risk_inputs(flooded_area_km2=None))
    assert "flooded_area_km2" in result["data_gaps"]
    assert (
        result["substitutions"]["extent_severity"]
        == "missing→hazard_degrades_to_water_severity_only"
    )
    assert (
        result["inputs"]["hazard_basis"] == "water_severity"
    )
    assert result["inputs"]["extent_ratio"] is None
    # Without a city boundary the area basis falls back to a circular
    # buffer, also recorded
    degraded = RiskEngine().compute_decision_indices(
        **_risk_inputs(flooded_area_km2=None, city_area_km2=None)
    )
    assert degraded["substitutions"]["analysis_area"] == "city_area_missing→circular_buffer"
    # Full inputs: no gaps, hazard is two-dimensional
    full = RiskEngine().compute_decision_indices(**_risk_inputs())
    assert full["data_gaps"] == []
    assert full["inputs"]["hazard_basis"] == "water_severity"
    assert full["inputs"]["extent_ratio"] == pytest.approx(0.1, abs=1e-4)


def test_risk_engine_gauge_unreadable_zeroes_hazard():
    """Water level is the primary hazard evidence: an unreadable gauge
    -> hazard = 0 (CDRI goes to 0); the satellite area share is kept
    only as spatial context, never supporting hazard."""
    result = RiskEngine().compute_decision_indices(
        **_risk_inputs(water_level=None, action_stage=None, major_stage=None)
    )
    assert "action_stage_or_water_level" in result["data_gaps"]
    assert result["inputs"]["hazard_basis"] == "water_severity_missing"
    assert result["components"]["hazard"] == 0.0
    assert result["cdri"] == 0.0
    # Area share still disclosed as spatial context (drives maps /
    # exposure / contradiction alerts)
    assert result["inputs"]["extent_severity"] == pytest.approx(0.1, abs=1e-4)


# ------------------------------------------------------------------
# 3. geometry_engine: areal-weighted exposure
# ------------------------------------------------------------------
def _ring(lat, lon, d=0.02):
    return [[lon - d, lat - d], [lon + d, lat - d],
            [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]


def test_areal_weighted_population_partial_overlap():
    """Flood covering half a tract's area counts only half its
    population, not the whole tract."""
    from shapely.geometry import Polygon

    engine = GeometryEngine()
    # Flood polygon: western half of the tract (half the lon span)
    flood = Polygon(_ring(29.50, -95.098, d=0.02))  # offset half a cell from the tract
    tracts = {
        "status": "ok",
        "tracts": [
            {"tract_id": "T1", "population": 1000, "svi": 0.5,
             "geometry": {"rings": [_ring(29.50, -95.10)]}},
        ],
    }
    records = engine.tract_exposure(flood, tracts)
    frac = records[0]["flooded_fraction"]
    assert 0.0 < frac < 1.0          # partial coverage, not a 0/1 binary
    assert records[0]["method"] == "areal_weighted"
    assert records[0]["exposed_population"] == pytest.approx(
        1000 * frac, abs=1.0
    )
    affected, count, method = engine.estimate_affected_population(
        flood, tracts
    )
    assert affected == pytest.approx(1000 * frac, abs=1.0)
    assert method == "areal_weighted"
    assert count == 1


def test_areal_weighted_population_no_overlap():
    from shapely.geometry import Polygon

    engine = GeometryEngine()
    flood = Polygon(_ring(29.90, -95.10, d=0.01))  # far from the tract
    tracts = {
        "status": "ok",
        "tracts": [
            {"tract_id": "T1", "population": 1000, "svi": 0.5,
             "geometry": {"rings": [_ring(29.50, -95.10)]}},
        ],
    }
    affected, count, method = engine.estimate_affected_population(
        flood, tracts
    )
    assert affected == 0.0
    assert count == 0
    assert method == "areal_weighted"


# ------------------------------------------------------------------
# 4. flood_fusion_engine: rejecting error sources + fusing the rest
# ------------------------------------------------------------------
def _fusion_obs(source, status="ok", value=1.0, observed_at="2026-08-29T00:00:00+00:00"):
    payload = {
        "source": source,
        "source_type": "hydrology",
        "tool": "get_flood_observation",
        "raw": json.dumps({
            "status": status,
            "observation": {
                "water_level": value,
                "observed_at": observed_at,
                "station_id": "08077600",
            },
        }),
    }
    if status == "error":
        payload["raw"] = json.dumps({
            "status": "error", "error_code": "NO_OBSERVATION_IMAGERY", "error": "x",
        })
    return payload


def test_fusion_rejects_error_source_but_fuses_rest():
    fused = fuse_flood_evidence([
        _fusion_obs("good1", value=2.0),
        _fusion_obs("bad", status="error"),
    ])
    assert fused["status"] == "fused"
    rejected = fused.get("rejected_sources", [])
    assert any(r.get("source") == "bad" for r in rejected)
    assert fused["fused_measurements"]["water_level"]["source_count"] == 1


def test_normalize_rejects_non_ok_status():
    obs = _fusion_obs("bad", status="error")
    with pytest.raises(Exception):
        normalize_fusion_observation(obs)


# ------------------------------------------------------------------
# 5. MasterRouter registry routing (hazard plugins)
# ------------------------------------------------------------------
def test_router_word_boundary_prevents_wildfire_misroute():
    router = MasterRouter()
    # A "river" substring match must not route wildfire queries into
    # FloodSkill
    with pytest.raises(ValueError):
        router.classify("wildfire crossed the river near Bastrop")
    with pytest.raises(ValueError):
        router.classify("fire situation at stage 2 of containment")
    assert router.classify("assess flooding near Friendswood river") == "flood"
    # Earthquakes are unsupported: explicit rejection, not misrouting
    with pytest.raises(ValueError):
        router.classify("earthquake magnitude 4.5 near Anchorage")
    # Multi-hazard ambiguity (with an unsupported hazard) still rejects
    with pytest.raises(ValueError):
        router.classify("flood and earthquake both mentioned")


def test_registry_plugin_interface():
    # Flood is registered via @register_skill; factory is constructible;
    # prompt rules and validator are retrievable
    assert skill_registry.registered_hazards() == ("flood",)
    assert skill_registry.get_spec("flood").skill_class is FloodSkill
    assert "FLOOD-SPECIFIC RULES" in skill_registry.hazard_prompt_rules("flood")
    assert callable(skill_registry.get_spec("flood").output_validator)


# ------------------------------------------------------------------
# 6. Adaptive HITL gates
# ------------------------------------------------------------------
def test_hitl_enabled_gate_and_confidence():
    state = RunState(run_id="t1")
    hitl = AdaptiveHITL(state)
    assert hitl.should_intervene(confidence=0.99) is False   # high confidence passes automatically
    assert hitl.should_intervene(confidence=0.10) is True
    assert hitl.should_intervene(evidence_conflict=True) is True


@pytest.mark.asyncio
async def test_hitl_ask_async_respects_disabled(monkeypatch):
    monkeypatch.setenv("HITL_ENABLED", "false")
    state = RunState(run_id="t2")
    hitl = AdaptiveHITL(state)
    answer = await hitl.ask_async(
        reason="test", question="q?", proposed_value="proceed",
    )
    # When disabled, return the proposed value immediately instead of
    # blocking on a human
    assert answer == "proceed"


# ------------------------------------------------------------------
# 7. FinalDecision output validation: dynamic city-name restrictions
# ------------------------------------------------------------------
def _packet(city=None):
    evidence = {
        "confidence": None,
        "attributes": ({"target": city} if city else {}),
    }
    return {
        "evidence": [evidence],
        "spatial_objects": [],
        "validation_issues": [],
    }


def test_validate_output_blocks_dynamic_city_claim():
    packet = _packet(city="Friendswood")
    # Hazard-specific validation lives in the FloodSkill plugin
    # (OUTPUT_VALIDATOR)
    with pytest.raises(RuntimeError):
        FloodSkill.OUTPUT_VALIDATOR(
            "Friendswood is flooding tonight.", packet
        )
    # No areal evidence + non-city-level wording -> passes
    FloodSkill.OUTPUT_VALIDATOR(
        "The station recorded 5.0 ft at 14:00 UTC.", packet
    )
    # The generic agent layer carries no flood-specific wording bans --
    # the same text passes generic validation
    FinalDecisionAgent._validate_output(
        "Friendswood is flooding tonight.", packet
    )


# ------------------------------------------------------------------
# 8. Place-name disambiguation: state qualifiers + geocode scoring
# ------------------------------------------------------------------
def test_extract_target_keeps_state_qualifier():
    router = MasterRouter()
    # Comma form: "Manhattan, Kansas" must go to geocoding as one unit
    assert router.extract_target(
        "assess flooding near Manhattan, Kansas using USGS station 06879650",
        "flood",
    ) == "Manhattan Kansas"
    # Space-separated form
    assert router.extract_target(
        "flooding around Manhattan Kansas with station 06879650",
        "flood",
    ) == "Manhattan Kansas"
    # No state qualifier: must not invent one
    assert router.extract_target(
        "assess flooding near Manhattan using USGS station 06879650",
        "flood",
    ) == "Manhattan"


def test_geocode_candidate_scoring_prefers_qualified_match():
    # Mirrors real Nominatim output: bare "Manhattan" prefers NYC
    # (importance 0.74); with the "Kansas" qualifier, token coverage
    # must outweigh importance.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers",
    ))
    from geocode import _pick_best  # noqa: E402

    candidates = [
        {"display_name": "Manhattan, New York County, New York, United States",
         "importance": 0.74, "lat": "40.78", "lon": "-73.96"},
        {"display_name": "Manhattan, Riley County, Kansas, United States",
         "importance": 0.57, "lat": "39.18", "lon": "-96.57"},
    ]
    # Bare "Manhattan": equal token coverage of 1, higher importance
    # wins (NYC)
    assert "New York" in _pick_best("Manhattan", candidates)["display_name"]
    # "Manhattan Kansas": the Kansas candidate covers 2 tokens (+4),
    # beating NYC's importance edge (0.74-0.57=0.17)
    assert "Kansas" in _pick_best("Manhattan Kansas", candidates)["display_name"]


# ------------------------------------------------------------------
# 9. Three-segment input "<hazard> <place> <station>" + station ID
# ------------------------------------------------------------------
def test_three_segment_parsing():
    router = MasterRouter()
    # Three segments: no prepositions, no "station" keyword
    assert router.classify("flood Manhattan 06887000") == "flood"
    assert router.extract_target("flood Manhattan 06887000", "flood") == "Manhattan"
    assert router.extract_station_id("flood Manhattan 06887000") == "06887000"
    # Three segments + state qualifier
    assert router.extract_target(
        "flood Manhattan Kansas 06887000", "flood"
    ) == "Manhattan Kansas"
    # Classic phrasing still works
    assert router.extract_station_id(
        "assess flooding near Manhattan, Kansas using USGS station 06879650"
    ) == "06879650"
    assert router.extract_station_id("flooding near Friendswood") is None
    # Fewer than 8 or more than 16 digits is not a station ID
    assert router.extract_station_id("flood Somewhere 1234567890123456") is None


def test_geocode_bias_picks_station_neighborhood():
    """Station-anchored disambiguation: a unique station ID anchors
    the geocode, so ambiguous names resolve near the station."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers",
    ))
    from geocode import _pick_best  # noqa: E402

    candidates = [
        {"display_name": "Manhattan, New York County, New York, United States",
         "importance": 0.74, "lat": "40.78", "lon": "-73.96"},
        {"display_name": "Manhattan, Riley County, Kansas, United States",
         "importance": 0.57, "lat": "39.18", "lon": "-96.57"},
    ]
    # No anchor: bare "Manhattan" lands on NYC (higher importance)
    assert "New York" in _pick_best("Manhattan", candidates)["display_name"]
    # Anchor = Kansas station (39.24, -96.61): only the KS candidate
    # is within the 150 km neighborhood
    picked = _pick_best("Manhattan", candidates, bias_lat=39.24, bias_lon=-96.61)
    assert "Kansas" in picked["display_name"]


def test_risk_engine_missing_affected_facilities_neutral_not_count_fabrication():
    """Unknown affected facilities take a neutral 0.5; the nearby
    facility count must never be fabricated into severity. Missing
    dimensions surface as data_gaps disclosures and a widened
    exposure uncertainty interval."""
    result = RiskEngine().compute_decision_indices(
        **_risk_inputs(
            social_vulnerability={
                "status": "ok",
                "profile": {
                    "population_weighted_svi": 0.5,
                    "total_population": 1000,
                },
                "tracts": [],
            },
            gis_stats={"affected_population": 100, "travel_time_min": None},
            fused_measurements={
                "facility_count": {"value": 141},
                "road_count": {"value": 0.0},
            },
        )
    )
    # Population share = 100/1000 = 0.1; facility dimension missing ->
    # neutral 0.5, so exposure = max(0.1, 0.5) = 0.5, not 141/100
    # clamped to 1.0
    assert result["components"]["exposure"] == 0.5
    # CDRI = h x (0.5 + 0.5*0.5) x v = 0.3333 x 0.75 x 0.5 = 12.5%
    assert result["cdri_percent"] == 12.5
    assert "affected_facilities" in result["data_gaps"]
    assert any(
        "affected_facilities missing" in str(v)
        for v in result["substitutions"].values()
    )
    # Missing dimension -> exposure unconstrained: the interval's
    # upper bound hits the exposure=1 ceiling (h x 1 x v with +/-10%
    # jitter ~= 20.8%), far above the narrow band of the
    # known-facilities path
    assert result["uncertainty"]["interval_percent"] == [6.53, 20.8]
    print("PASS missing_affected_facilities_neutral")
