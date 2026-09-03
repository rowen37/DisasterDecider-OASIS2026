# tests/test_offline_units.py
#
# Offline unit tests: no LLM calls, no external API access.
# Run: uv run python -m pytest tests/test_offline_units.py -v
# or directly: uv run python tests/test_offline_units.py

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import _extract_structured_data          # noqa: E402
from app.hitl import AdaptiveHITL                      # noqa: E402
from app.models import (                               # noqa: E402
    Evidence,
    RunState,
    SkillResult,
)
from app.engine.flood_fusion_engine import (        # noqa: E402
    normalize_fusion_observation,
)


# ------------------------------------------------------------------
# 1. Structured data extraction (main.py)
# ------------------------------------------------------------------
def _make_result() -> SkillResult:
    primary = Evidence(
        evidence_id="usgs_station_08077600",
        source="USGS Water Services via Flood Alert MCP",
        observation="USGS monitoring station 08077600 reported 1.21 ft.",
        quality_score=0.76,
        timestamp="2026-08-28T14:15:00-05:00",
        attributes={
            "station_id": "08077600",
            "station_name": "Clear Ck nr Friendswood, TX",
            "station_latitude": 29.5175,
            "station_longitude": -95.1785,
            "target_latitude": 29.5294,
            "target_longitude": -95.2010,
            "water_level": 1.21,
            "nwps_flood_categories": {
                "action": {"stage": 7}, "minor": {"stage": 12},
                "moderate": {"stage": 16}, "major": {"stage": 21},
            },
            "multi_source_fusion": {
                "status": "fused",
                "observations": [
                    {
                        "raw": {
                            "source_type": "population_exposure",
                            "population": {"total": 8500},
                        },
                    },
                    {
                        "raw": {
                            "source_type": "infrastructure",
                            "measurements": {
                                "facility_count": {"value": 141},
                                "hospital_count": {"value": 7},
                            },
                            "metadata": {"facilities": [
                                {"name": "HCA Houston", "type": "hospital",
                                 "latitude": 29.53, "longitude": -95.20},
                            ]},
                        },
                    },
                    {
                        "raw": {
                            "source_type": "road_network",
                            "measurements": {
                                "road_count": {"value": 26519},
                                "bridge_count": {"value": 416},
                            },
                        },
                    },
                ],
            },
            "gis_stats": {
                "affected_buildings": 246,
                "affected_facilities": 2,
                "affected_roads": 1946,
                "affected_population": 12693,
            },
            "decision_indices": {
                "cdri": 0.0002,
                "cdri_percent": 0.02,
                "cdri_risk_label": "Low",
                "eps": 0.0437,
                "data_confidence": 0.7355,
            },
        },
    )

    forecast = Evidence(
        evidence_id="forecast_0",
        source="forecast",
        observation="NWS forecast periods.",
        timestamp="2026-08-28T13:00:00-05:00",
        attributes={
            "source_type": "forecast",
            "forecast_count": 7,
            "forecasts": [
                {"period": "This Afternoon", "temperature": 97,
                 "temperature_unit": "F", "probability_of_precipitation": 34,
                 "wind_speed": "5 mph", "short_forecast": "Chance Showers",
                 "timestamp": "2026-08-28T13:00:00-05:00"},
                {"period": "Tonight", "temperature": 78,
                 "temperature_unit": "F", "probability_of_precipitation": 49,
                 "wind_speed": "0 to 10 mph", "short_forecast": "Showers",
                 "timestamp": "2026-08-28T18:00:00-05:00"},
            ],
        },
    )

    svi = Evidence(
        evidence_id="svi",
        source="CDC/ATSDR SVI 2022",
        # Mirrors production wording; the text mentions "population",
        # so the extractor must not mistake the tract count for it.
        observation=(
            "CDC/ATSDR SVI evidence covers 325 census tract(s) "
            "with a population-weighted SVI of 0.607."
        ),
        attributes={
            "population_weighted_svi": 0.6068,
            "total_population": 1371035,
            "tract_count": 325,
        },
    )

    return SkillResult(
        status="completed",
        summary="test",
        evidence=[primary, forecast, svi],
    )


def test_extract_structured_data():
    state = RunState(run_id="t")
    structured = _extract_structured_data(_make_result(), state)

    assert structured["station_name"] == "Clear Ck nr Friendswood, TX"
    assert structured["water_level"] == 1.21
    assert structured["action_stage"] == 7
    assert len(structured["forecast_periods"]) == 2
    assert structured["forecast_periods"][0]["probability_of_precipitation"] == 34
    assert structured["population_total"] == 1371035
    assert structured["population_affected"] == 12693
    # Population exposure comes only from the structured fusion path
    # (Census tract population) and must never be overridden by the
    # "325 census tract(s)" text in the SVI evidence.
    assert structured["population_exposure"] == 8500
    ic = structured["infrastructure_counts"]
    assert ic["affected_buildings"] == 246
    assert ic["affected_facilities"] == 2
    assert ic["affected_roads"] == 1946
    assert ic["nearby_facility_count"] == 141
    assert ic["nearby_road_count"] == 26519
    di = structured["decision_indices"]
    assert di["cdri_percent"] == 0.02
    assert di["cdri_risk_label"] == "Low"
    assert structured["osm_facilities"][0]["name"] == "HCA Houston"
    print("PASS extract_structured_data")


# ------------------------------------------------------------------
# 2. HITL web mode (async wait + submit response + timeout defaults)
# ------------------------------------------------------------------
def test_hitl_web_mode():
    state = RunState(run_id="t")
    hitl = AdaptiveHITL(state)
    hitl.enable_web_mode()

    async def scenario():
        # Scenario A: frontend submits within the timeout
        task = asyncio.create_task(
            hitl.ask_async(
                reason="Parameter confirmation",
                question="Continue with defaults?",
                proposed_value='{"vulnerability_weight": 1.0}',
            )
        )
        await asyncio.sleep(0.05)
        pending = hitl.get_pending_request()
        assert pending is not None
        assert pending["reason"] == "Parameter confirmation"
        hitl.submit_response("")  # empty input = accept default
        answer = await asyncio.wait_for(task, timeout=2)
        assert answer == '{"vulnerability_weight": 1.0}'

        # Scenario B: override value
        task2 = asyncio.create_task(
            hitl.ask_async(reason="r", question="q", proposed_value="default")
        )
        await asyncio.sleep(0.05)
        hitl.submit_response('{"vulnerability_weight": 2.0}')
        answer2 = await asyncio.wait_for(task2, timeout=2)
        assert answer2 == '{"vulnerability_weight": 2.0}'
        assert hitl.get_pending_request() is None

    asyncio.run(scenario())
    print("PASS hitl_web_mode")


# ------------------------------------------------------------------
# 3. Fusion normalization branches (no rain / forecast / GEE time)
# ------------------------------------------------------------------
def test_normalize_fusion_branches():
    # No precipitation
    obs = {
        "source": "precipitation", "source_type": "precipitation",
        "tool": "get_precipitation",
        "raw": json.dumps({
            "status": "ok", "source": "NWS", "source_type": "precipitation",
            "observation_semantics": "no_precipitation_observed",
            "note": "no rain", "stations_checked": 15,
            "timestamp": "2026-08-28T02:45:00+00:00",
            "location": {"latitude": 29.5, "longitude": -95.2},
        }),
    }
    items = normalize_fusion_observation(obs)
    assert items and items[0]["evidence_type"] == "status_observation"
    assert items[0]["stations_checked"] == 15

    # Forecast
    obs2 = {
        "source": "forecast", "source_type": "forecast",
        "tool": "get_forecast",
        "raw": json.dumps({
            "status": "ok", "source": "NWS", "source_type": "forecast",
            "location": {"latitude": 29.5, "longitude": -95.2},
            "forecasts": [{"period": "Tonight", "temperature": 78,
                           "probability_of_precipitation": 49}],
        }),
    }
    items2 = normalize_fusion_observation(obs2)
    assert items2 and items2[0]["evidence_type"] == "forecast"

    # GEE with acquisition time
    obs3 = {
        "source": "GEE", "source_type": "satellite_sar",
        "tool": "get_flood_extent",
        "raw": json.dumps({
            "status": "ok",
            "observation": {"latest_post_scene": "2026-08-25T00:26:15Z",
                            "pre_scene_count": 3, "post_scene_count": 2},
            "spatial_extent": {"flooded_area_km2": 19.2,
                               "geojson": {"type": "FeatureCollection",
                                           "features": [{"type": "Feature"}]}},
        }),
    }
    items3 = normalize_fusion_observation(obs3)
    it = items3[0]
    assert it["evidence_type"] == "spatial_extent"
    assert it["acquisition_time"] == "2026-08-25T00:26:15Z"
    assert it["timestamp"] == "2026-08-25T00:26:15Z"
    print("PASS normalize_fusion_branches")


# ------------------------------------------------------------------
# 4. Resource plan building (resource_manager, purely local)
# ------------------------------------------------------------------
def test_resource_plan_building():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers",
    ))
    os.environ.pop("MANUAL_RESOURCES_JSON", None)
    from resource_manager import _build_plans, DEFAULT_MANUAL  # noqa: E402

    facilities = [
        {"kind": "hospital", "name": "H1", "lat": 29.53, "lon": -95.2,
         "distance_km": 2.0, "tags": {}},
        {"kind": "hospital", "name": "H2", "lat": 29.54, "lon": -95.21,
         "distance_km": 4.0, "tags": {"beds": "120"}},
        {"kind": "shelter", "name": "S1", "lat": 29.52, "lon": -95.19,
         "distance_km": 1.5, "tags": {}},
        {"kind": "fire_station", "name": "F1", "lat": 29.51, "lon": -95.22,
         "distance_km": 3.0, "tags": {}},
    ]
    plans = _build_plans(facilities, dict(DEFAULT_MANUAL), 1000.0, 0.5)

    assert plans, "plans must not be empty"
    for p in plans:
        obj = p["objectives"]
        for key in ("risk_reduction", "coverage", "response_time",
                    "cost", "unmet_demand"):
            assert key in obj, f"missing objective {key}"
        assert p["allocations"]
    # A hospital with a beds tag must not get assumed capacity
    h2_plan = next(
        a for p in plans for a in p["allocations"] if a["resource"] == "H2"
    )
    assert h2_plan["capacity"] == 120.0
    assert h2_plan["capacity_assumed"] is False
    # Distance ordering: hospital_top1 must pick the nearest H1
    hosp_top1 = next(p for p in plans if p["plan_id"] == "hospital_top1")
    assert hosp_top1["allocations"][0]["resource"] == "H1"
    print("PASS resource_plan_building")


# ------------------------------------------------------------------
# 5. CDRI percent scale and label thresholds (asserts on the
#    production implementation, not a test-local copy)
# ------------------------------------------------------------------
def test_cdri_scale_and_labels():
    from app.engine.flood_risk_engine import cdri_risk_label

    # Label bands: <0.5% Low, 0.5-2% Moderate, 2-10% High,
    # >=10% Very High
    assert cdri_risk_label(0.0002) == "Low"
    assert cdri_risk_label(0.02) == "High"
    assert cdri_risk_label(0.08) == "High"
    assert cdri_risk_label(0.4) == "Very High"
    # Boundary values: a threshold itself maps to the higher band
    # (0.5% -> Moderate, 2% -> High, 10% -> Very High)
    assert cdri_risk_label(0.00499) == "Low"
    assert cdri_risk_label(0.005) == "Moderate"
    assert cdri_risk_label(0.02) == "High"
    assert cdri_risk_label(0.10) == "Very High"
    assert round(0.0002 * 100, 2) == 0.02
    print("PASS cdri_scale_and_labels")


def test_llm_evidence_compaction_bounded():
    """Regression for oversized LLM packets: geometry must be
    summarized, long lists capped, and the packet kept under the
    hard prompt limit."""
    from app.agent import _build_compact_packet, _LLM_MAX_PROMPT_CHARS

    big_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"geometry": {"coordinates": [[i * 0.001, i * 0.002]
                                          for i in range(2000)]}}
            for _ in range(50)
        ],
    }
    packet = {
        "task": "assess flooding near Friendswood",
        "summary": "ok",
        "evidence": [
            {"source": "GEE", "geojson": big_geojson,
             "note": "x" * 5000},
        ],
        "spatial_objects": [
            {"name": f"obj{i}", "geometry": {"coordinates": [[0, 0]]},
             "attributes": {"a": i}}
            for i in range(200)
        ],
    }
    compact = _build_compact_packet(packet)
    size = len(json.dumps(compact, ensure_ascii=False))
    # After summarization the packet fits the model context budget
    assert size < _LLM_MAX_PROMPT_CHARS
    # Geometry summarized (no coordinate strings); type and size kept
    ev0 = compact["evidence"][0]
    assert ev0["geojson"]["geometry_omitted"] is True
    assert ev0["geojson"]["approx_serialized_chars"] > 0
    assert ev0["note"].startswith("xxx") and "truncated" in ev0["note"]
    # Long lists capped: 200 objects -> 25 + 1 omission note
    assert len(compact["spatial_objects"]) == 26
    assert "more items omitted" in compact["spatial_objects"][-1]
    print("PASS llm_evidence_compaction_bounded")


if __name__ == "__main__":
    test_llm_evidence_compaction_bounded()
    test_extract_structured_data()
    test_hitl_web_mode()
    test_normalize_fusion_branches()
    test_resource_plan_building()
    test_cdri_scale_and_labels()
    test_vwun_formula_no_double_hazard()
    test_vulnerability_profile_counts()
    test_gis_io_out_path_restriction()
    test_demo_env_restored_after_run()
    print("\nALL OFFLINE UNIT TESTS PASSED")


# ------------------------------------------------------------------
# 6. Honesty: explicit accounting of missing inputs (risk engine)
# ------------------------------------------------------------------
def test_risk_engine_data_gaps_accounting():
    from app.engine.flood_risk_engine import RiskEngine

    base = dict(
        water_level=5.0, action_stage=10.0, major_stage=None,
        flooded_area_km2=50.0,
        city_area_km2=100.0, fallback_analysis_radius_km=10.0,
        social_vulnerability={
            "status": "ok",
            "profile": {"population_weighted_svi": 0.5, "total_population": 1000},
            "tracts": [],
        },
        gis_stats={"affected_population": 500, "affected_facilities": 30,
                   "travel_time_min": None},
        fused_measurements={"facility_count": {"value": 60},
                            "road_count": {"value": 0.0}},
        fusion_sources=[{"tool": "get_road_status",
                         "arguments": {"radius_km": 10}}],
        observations=[{"quality_score": 0.8}],
    )

    full = RiskEngine().compute_decision_indices(**base)
    assert full["data_gaps"] == []
    # All components computed -> no degraded suffix on the label
    assert "(degraded" not in full["cdri_risk_label"]

    gapped = RiskEngine().compute_decision_indices(**{
        **base,
        "action_stage": None,          # water_ratio missing
        "flooded_area_km2": None,      # extent_ratio missing
        "social_vulnerability": None,  # SVI missing
        "city_area_km2": None,         # area falls back to buffer
    })
    assert "action_stage_or_water_level" in gapped["data_gaps"]
    assert "flooded_area_km2" in gapped["data_gaps"]
    assert "population_weighted_svi" in gapped["data_gaps"]
    assert gapped["substitutions"]["water_severity"] == "missing→0"
    assert (
        gapped["substitutions"]["extent_severity"]
        == "missing→hazard_degrades_to_water_severity_only"
    )
    # Water level and area both missing -> hazard basis honestly marked
    assert gapped["inputs"]["hazard_basis"] == "water_severity_missing"
    # SVI missing -> neutral 0.5 point estimate (not 0), interval
    # spanning [0,1], label explicitly degraded: unknown vulnerability
    # must never read as low risk.
    assert gapped["substitutions"]["vulnerability"].startswith(
        "missing→neutral_0.5"
    )
    assert gapped["components"]["vulnerability"] == 0.5
    assert "(degraded" in gapped["cdri_risk_label"]
    assert "vulnerability" in gapped["uncertainty"]["unconstrained_components"]
    assert gapped["uncertainty"]["interval"][0] == 0.0
    assert gapped["substitutions"]["analysis_area"] == "city_area_missing→circular_buffer"
    print("PASS risk_engine_data_gaps_accounting")


# ------------------------------------------------------------------
# 7. Equity objective: real vulnerability_coverage + honest flag
# ------------------------------------------------------------------
def test_allocation_equity_objective():
    from app.engine.allocation_engine import AllocationEngine

    def ring(lat, lon, d=0.01):
        return [[lon-d, lat-d], [lon+d, lat-d], [lon+d, lat+d],
                [lon-d, lat+d], [lon-d, lat-d]]

    eng = AllocationEngine(
        objectives=[{"name": "coverage", "direction": "maximize"}],
        weights={"coverage": 1.0},
        max_plan_combinations=100,
        vulnerability_coverage_radius_km=10.0,
    )
    tracts = [
        {"svi": 0.9, "population": 1000,
         "geometry": {"rings": [ring(29.50, -95.10)]}},
        {"svi": 0.2, "population": 1000,
         "geometry": {"rings": [ring(29.70, -95.11)]}},  # ~22 km away
    ]
    resources = [
        {"plan_id": "A",
         "allocations": [{"resource": "H1", "lat": 29.505, "lon": -95.101,
                          "capacity": 100, "type": "hospital",
                          "distance_km": 1.0, "capacity_assumed": False}],
         "objectives": {"coverage": 0.9}, "metadata": {}},
        {"plan_id": "B",
         "allocations": [{"resource": "H2", "lat": 29.695, "lon": -95.111,
                          "capacity": 100, "type": "hospital",
                          "distance_km": 1.0, "capacity_assumed": False}],
         "objectives": {"coverage": 0.9}, "metadata": {}},
    ]

    plans = eng.generate_plans(resources, {}, svi_tracts=tracts)
    a = next(p for p in plans if p["plan_id"] == "A")
    b = next(p for p in plans if p["plan_id"] == "B")
    # Hand-computed: A covers 900/1100 of high-SVI population,
    # B covers 200/1100 of low-SVI
    assert abs(a["objectives"]["vulnerability_coverage"] - 900/1100) < 0.001
    assert abs(b["objectives"]["vulnerability_coverage"] - 200/1100) < 0.001

    # Equity objective active with weight -> plan A near the
    # high-vulnerability tract wins
    eff = eng.effective_objectives(plans)
    assert "vulnerability_coverage" in {o["name"] for o in eff}
    out = eng.optimize(plans, weights_override={
        "coverage": 0.0, "vulnerability_coverage": 1.0})
    assert out["recommended_plan"]["plan_id"] == "A"
    assert out["svi_weight_applied"] is True

    # No SVI data: not computed, not activated, and applied stays
    # false even with a weight
    plans_no_svi = eng.generate_plans(resources, {}, svi_tracts=None)
    assert all("vulnerability_coverage" not in p["objectives"]
               for p in plans_no_svi)
    out2 = eng.optimize(plans_no_svi, weights_override={
        "coverage": 1.0, "vulnerability_coverage": 5.0})
    assert out2["svi_weight_applied"] is False
    print("PASS allocation_equity_objective")


# ------------------------------------------------------------------
# 8. VWUN formula: exposed_population already includes the flooded
#    fraction; never multiply by it again
# ------------------------------------------------------------------
def test_vwun_formula_no_double_hazard():
    from app.social_good import compute_vulnerability_weighted_unmet_need

    # Hand-computed: P_exposed=3500 (= 4000 population x 0.875
    # flooded fraction), uncovered, SVI 0.92, lambda=1 ->
    # 3500 x 1 x (1 + 0.92) = 6720. Multiplying flooded_fraction
    # again would square the exposure fraction and understate VWUN.
    impacts = [{
        "tract_id": "t1",
        "exposed_population": 3500.0,
        "hazard_exposure": 0.875,   # audit field: not part of the formula
        "coverage": 0.0,
        "svi": 0.92,
    }]
    assert compute_vulnerability_weighted_unmet_need(impacts, 1.0) == 6720.0

    # Full coverage -> 0; lambda only scales the vulnerability-weighted term
    impacts_covered = [{**impacts[0], "coverage": 1.0}]
    assert compute_vulnerability_weighted_unmet_need(impacts_covered, 1.0) == 0.0
    assert compute_vulnerability_weighted_unmet_need(impacts, 0.0) == 3500.0
    print("PASS vwun_formula_no_double_hazard")


# ------------------------------------------------------------------
# 9. SVI profile: real high_vulnerability_tract_count
#    (resource_skill reads this key for its SVI signal)
# ------------------------------------------------------------------
def test_vulnerability_profile_counts():
    from app.social_good import vulnerability_profile

    tracts = [
        {"tract_id": "a", "population": 4000, "svi": 0.92},
        {"tract_id": "b", "population": 6000, "svi": 0.30},
        {"tract_id": "c", "population": 1000, "svi": 0.95},
    ]
    profile = vulnerability_profile(tracts)
    assert profile["high_vulnerability_threshold"] == 0.90
    assert profile["high_vulnerability_tract_count"] == 2
    assert profile["high_vulnerability_population"] == 5000
    assert profile["tract_count"] == 3
    print("PASS vulnerability_profile_counts")


# ------------------------------------------------------------------
# 10. gis_io output path restriction: escape paths must be rejected
#     (prevents arbitrary file writes)
# ------------------------------------------------------------------
def test_gis_io_out_path_restriction():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers",
    ))
    import gis_io

    # Empty path -> temp directory
    p = gis_io.out_path("", ".geojson")
    assert gis_io.is_allowed_path(p)
    # Explicit path inside the temp directory -> allowed
    tmp_explicit = os.path.join(
        tempfile.gettempdir(), "dsa_test_layer.geojson"
    )
    assert gis_io.out_path(tmp_explicit, ".geojson") == str(
        os.path.realpath(tmp_explicit)
    )
    # Inside the project static/maps directory -> allowed
    maps_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "static", "maps", "x.geojson",
    )
    assert gis_io.out_path(maps_dir, ".geojson")
    # Escape paths -> rejected (home dir traversal / arbitrary
    # absolute path / relative traversal)
    for evil in ("../../.zshrc", "/etc/cron.d/evil", "static/../../evil.txt"):
        try:
            gis_io.out_path(evil, ".geojson")
        except ValueError:
            pass
        else:
            raise AssertionError(f"out_path accepted escape path: {evil}")
    assert not gis_io.is_allowed_path("/etc/passwd")
    print("PASS gis_io_out_path_restriction")




# ------------------------------------------------------------------
# 12. Demo env vars: restored after the run (never leak into real
#     requests in the same process)
# ------------------------------------------------------------------
def test_demo_env_restored_after_run():
    os.environ.pop("SVI_RADIUS_KM", None)
    os.environ.pop("VULNERABILITY_COVERAGE_RADIUS_KM", None)
    try:
        from app.demo_fixtures import DEMO_ENV, _DemoEnv

        with _DemoEnv({"SVI_RADIUS_KM": "10",
                       "VULNERABILITY_COVERAGE_RADIUS_KM": "2"}):
            assert os.environ["SVI_RADIUS_KM"] == "10"
            assert os.environ["VULNERABILITY_COVERAGE_RADIUS_KM"] == "2"
        # Restored to "absent" after exit
        assert "SVI_RADIUS_KM" not in os.environ
        assert "VULNERABILITY_COVERAGE_RADIUS_KM" not in os.environ

        # A pre-existing explicit value wins (setdefault semantics)
        # and is left untouched on exit
        os.environ["SVI_RADIUS_KM"] = "25"
        with _DemoEnv({"SVI_RADIUS_KM": "10"}):
            assert os.environ["SVI_RADIUS_KM"] == "25"
        assert os.environ["SVI_RADIUS_KM"] == "25"

        # Restored on the exception path too; None placeholders do
        # not override explicit values
        try:
            with _DemoEnv({"SVI_RADIUS_KM": None,
                           "NWS_WARNING_RADIUS_KM": "25"}):
                # SVI_RADIUS_KM keeps the explicit 25 from above (None skipped)
                assert os.environ["SVI_RADIUS_KM"] == "25"
                assert os.environ["NWS_WARNING_RADIUS_KM"] == "25"
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert os.environ["SVI_RADIUS_KM"] == "25"
        assert "NWS_WARNING_RADIUS_KM" not in os.environ
        # Module-level DEMO_ENV is a declarative constant, never
        # filled in place; run_demo_assessment works on a copy.
        assert DEMO_ENV["FLOOD_FUSION_SOURCES_JSON"] is None
        assert DEMO_ENV["RESOURCE_OBJECTIVE_WEIGHTS_JSON"] is None
    finally:
        os.environ.pop("SVI_RADIUS_KM", None)
    print("PASS demo_env_restored_after_run")


def test_llm_evidence_compaction_bounded():
    """Regression for oversized LLM packets: geometry must be
    summarized, long lists capped, and the packet kept under the
    hard prompt limit."""
    from app.agent import _build_compact_packet, _LLM_MAX_PROMPT_CHARS

    big_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"geometry": {"coordinates": [[i * 0.001, i * 0.002]
                                          for i in range(2000)]}}
            for _ in range(50)
        ],
    }
    packet = {
        "task": "assess flooding near Friendswood",
        "summary": "ok",
        "evidence": [
            {"source": "GEE", "geojson": big_geojson,
             "note": "x" * 5000},
        ],
        "spatial_objects": [
            {"name": f"obj{i}", "geometry": {"coordinates": [[0, 0]]},
             "attributes": {"a": i}}
            for i in range(200)
        ],
    }
    compact = _build_compact_packet(packet)
    size = len(json.dumps(compact, ensure_ascii=False))
    # After summarization the packet fits the model context budget
    assert size < _LLM_MAX_PROMPT_CHARS
    # Geometry summarized (no coordinate strings); type and size kept
    ev0 = compact["evidence"][0]
    assert ev0["geojson"]["geometry_omitted"] is True
    assert ev0["geojson"]["approx_serialized_chars"] > 0
    assert ev0["note"].startswith("xxx") and "truncated" in ev0["note"]
    # Long lists capped: 200 objects -> 25 + 1 omission note
    assert len(compact["spatial_objects"]) == 26
    assert "more items omitted" in compact["spatial_objects"][-1]
    print("PASS llm_evidence_compaction_bounded")


if __name__ == "__main__":
    test_risk_engine_data_gaps_accounting()
    test_allocation_equity_objective()
    test_equity_gap_or_none_undefined_group()


def test_equity_gap_or_none_undefined_group():
    """With homogeneous SVI (no tract above the national top-10%
    threshold) the gap is undefined -> None, while other invalid
    inputs still raise."""
    from app.social_good import (
        SocialGoodError,
        compute_equity_gap,
        compute_equity_gap_or_none,
    )

    homogeneous = [
        {"tract_id": "A", "exposed_population": 100.0, "coverage": 1.0, "svi": 0.59},
        {"tract_id": "B", "exposed_population": 50.0, "coverage": 0.0, "svi": 0.07},
    ]
    gap, note = compute_equity_gap_or_none(homogeneous, 0.90)
    assert gap is None
    assert "no demand tract" in note
    # The strict function still raises on the same input; callers
    # pick their policy
    with __import__("pytest").raises(SocialGoodError):
        compute_equity_gap(homogeneous, 0.90)

    mixed = homogeneous + [
        {"tract_id": "C", "exposed_population": 10.0, "coverage": 1.0, "svi": 0.95}
    ]
    gap2, note2 = compute_equity_gap_or_none(mixed, 0.90)
    assert gap2 is not None and note2 == "ok"
    # Invalid input is not swallowed into None: out-of-range SVI
    # still raises
    with __import__("pytest").raises(SocialGoodError):
        compute_equity_gap_or_none(
            [{"tract_id": "X", "exposed_population": 1.0, "coverage": 2.0, "svi": 0.5}],
            0.90,
        )
    print("PASS equity_gap_or_none_undefined_group")




# ------------------------------------------------------------------
# N. Extraction-layer regressions (Harvey 2017 replay)
# ------------------------------------------------------------------
def test_alerts_not_misextracted_from_flood_alert_mcp_source():
    """Gauge evidence whose source contains "Flood Alert MCP" must
    not be treated as alert evidence: the naive
    '"alert" in ev.source.lower()' check misfires on that source.
    """
    usgs = Evidence(
        evidence_id="usgs_station_08077600",
        source="USGS Water Services via Flood Alert MCP",
        observation="USGS monitoring station 08077600 reported 23.53 ft.",
        timestamp="2017-08-27T19:00:00.000-05:00",
        attributes={
            "station_id": "08077600",
            "water_level": 23.53,
            "unit": "ft",
            "nwps_flood_categories": {
                "action": {"stage": 7},
                "minor": {"stage": 12},
                "moderate": {"stage": 16},
                "major": {"stage": 21},
            },
        },
    )
    state = RunState(run_id="t")
    structured = _extract_structured_data(
        SkillResult(status="completed", summary="t", evidence=[usgs]),
        state,
    )
    assert structured["alerts"] is None, (
        "仅凭 source 子串 'Flood Alert MCP' 不得报出 Active Alerts"
    )
    assert structured["water_level"] == 23.53

    # Real alert evidence: zero alerts must show "0 active" (truthful
    # falsy-zero handling), never fall through to "Active".
    zero_alerts = Evidence(
        evidence_id="alert_0",
        source="flood_warnings",
        observation="0 active NWS alert(s) within the warning radius.",
        attributes={"source_type": "warning", "alert_count": 0, "alerts": []},
    )
    structured = _extract_structured_data(
        SkillResult(status="completed", summary="t", evidence=[usgs, zero_alerts]),
        RunState(run_id="t"),
    )
    assert structured["alerts"] == "0 active"
    assert structured["alert_count"] == 0

    # Alert evidence without a count field keeps the conservative
    # "Active" semantics
    unknown_count = Evidence(
        evidence_id="alert_1",
        source="flood_warnings",
        observation="NWS alert(s) within the warning radius.",
        attributes={"source_type": "warning"},
    )
    structured = _extract_structured_data(
        SkillResult(status="completed", summary="t", evidence=[unknown_count]),
        RunState(run_id="t"),
    )
    assert structured["alerts"] == "Active"
    print("PASS alerts_not_misextracted_from_flood_alert_mcp_source")


def test_population_affected_source_uses_method_label():
    """gis_stats.affected_population_method must pass through as the
    source label, consistent with
    decision_indices.inputs.affected_population_source."""
    gis_ev = Evidence(
        evidence_id="gis",
        source="GIS pipeline",
        observation="Flood footprint intersections.",
        attributes={
            "gis_stats": {
                "affected_population": 5527,
                "affected_population_method": "areal_weighted",
                "affected_population_interval": [0, 5528.0],
            },
            "decision_indices": {
                "cdri": 0.1115,
                "inputs": {
                    "affected_population": 5527,
                    "affected_population_source": "census_tract_areal_weighted",
                },
            },
        },
    )
    structured = _extract_structured_data(
        SkillResult(status="completed", summary="t", evidence=[gis_ev]),
        RunState(run_id="t"),
    )
    assert structured["population_affected"] == 5527
    assert structured["population_affected_source"] == "areal_weighted", (
        "受影响人口来源必须是 gis_stats 的 method 标签（面积加权），"
        "而不是质心法默认值"
    )
    di_inputs = (structured.get("decision_indices") or {}).get("inputs") or {}
    assert di_inputs.get("affected_population_source") == "census_tract_areal_weighted"
    print("PASS population_affected_source_uses_method_label")
