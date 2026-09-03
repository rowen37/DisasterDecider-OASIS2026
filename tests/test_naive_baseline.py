"""Baseline comparison — naive catch-all pipeline vs the reference pipeline.

Runs BOTH pipelines over the SAME offline fixtures (app.demo_fixtures) under
the SAME fault injections (§4.3 of the paper) and prints/validates the
comparison table.  Run verbosely to see the table:

    uv run python -m pytest tests/test_naive_baseline.py -s -q
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.demo_fixtures import FUSION_SOURCES, FakeMCP  # noqa: E402
from app.experiment import ExperimentLogger  # noqa: E402
from app.hitl import AdaptiveHITL  # noqa: E402
from app.models import RunState  # noqa: E402
from app.naive_baseline import NaiveBaseline  # noqa: E402
from app.skills import FloodSkill  # noqa: E402
from app.verification import Verifier  # noqa: E402

TARGET = "Friendswood"
STATION = "08077600"


def _full_env(monkeypatch) -> None:
    """Same environment as the nominal offline-replay test."""
    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("FLOOD_STATION_MAX_DISTANCE_KM", "50")
    monkeypatch.setenv("NWS_WARNING_RADIUS_KM", "25")
    monkeypatch.setenv("SVI_RADIUS_KM", "10")
    monkeypatch.setenv("SVI_MAX_FEATURES", "500")
    monkeypatch.setenv("VULNERABILITY_WEIGHT", "1.0")
    monkeypatch.setenv("EQUITY_HIGH_VULNERABILITY_THRESHOLD", "0.90")
    monkeypatch.setenv("FLOOD_FUSION_SOURCES_JSON", json.dumps(FUSION_SOURCES))
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
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")


# -- fault injections (identical to the degradation-matrix tests) ------

async def _meta_error(args):
    return json.dumps({
        "status": "error",
        "error": "HTTP 503 Service Unavailable (all 3 attempts)",
    })


async def _no_nwps(args):
    return json.dumps({
        "status": "error",
        "error": "NWPS could not find USGS ID",
    })


async def _geocode_seattle(args):
    return f"{args['place_name']} 的坐标: 纬度 47.6062, 经度 -122.3321"


async def _gee_dry(args):
    return json.dumps({
        "status": "ok",
        "observation": {
            "requested_observation_date": args.get("observation_date"),
            "window_adjusted": False,
            "latest_post_scene": "2026-08-28T23:59:00Z",
            "pre_scene_count": 3, "post_scene_count": 2,
        },
        "spatial_extent": {"flooded_area_km2": None, "geojson": None},
        "data_quality": {"geometry_returned": False,
                         "area_reduction_available": False},
    })


def _gee_today_scene():
    async def _gee_today(args):
        acq = datetime.now(timezone.utc).isoformat()
        return json.dumps({
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": args.get("observation_date"),
                "window_adjusted": False, "latest_post_scene": acq,
                "pre_scene_count": 3, "post_scene_count": 2,
                "pre_acquisition_times": [acq],
                "post_acquisition_times": [acq],
            },
            "spatial_extent": {
                "flooded_area_km2": 18.6,
                "geojson": __import__(
                    "app.demo_fixtures", fromlist=["_polygon_geojson"]
                )._polygon_geojson(-95.115, 29.51, d=0.03),
            },
            "analysis": {"threshold_method": "fixed", "threshold_db": -3.0},
            "data_quality": {"geometry_returned": True},
        })
    return _gee_today


SCENARIOS = (
    "S0-nominal", "S1-metadata-503", "S2-no-nwps",
    "S3-far-station", "S4-no-extent", "S5-stale-sar",
)


def _apply_fault(mcp: FakeMCP, scenario: str) -> None:
    if scenario == "S1-metadata-503":
        mcp._fx_get_station_metadata = _meta_error
    elif scenario == "S2-no-nwps":
        mcp._fx_get_nwps_gauge = _no_nwps
    elif scenario == "S3-far-station":
        mcp._fx_geocode_location = _geocode_seattle
    elif scenario == "S4-no-extent":
        mcp._fx_get_flood_extent = _gee_dry
    elif scenario == "S5-stale-sar":
        mcp._fx_get_flood_extent = _gee_today_scene()


def _scenario_args(scenario: str) -> tuple[str, str, str | None]:
    """(target, station_id, event_date) per scenario.

    S3 keeps the healthy Friendswood station so that the ONLY fault is the
    geocoder resolving the target to Seattle (~3,000 km away).
    """
    if scenario == "S3-far-station":
        return TARGET, STATION, None
    if scenario == "S5-stale-sar":
        return TARGET, STATION, "2017-08-27"
    return TARGET, STATION, None


async def _run_ours(monkeypatch, tmp_path, scenario: str):
    _full_env(monkeypatch)
    state = RunState(run_id=f"base-{scenario}")
    mcp = FakeMCP()
    _apply_fault(mcp, scenario)
    skill = FloodSkill(
        state, Verifier(), AdaptiveHITL(state), mcp,
        ExperimentLogger(path=str(tmp_path / f"{scenario}.jsonl")),
    )
    target, station, event_date = _scenario_args(scenario)
    kwargs = {"station_id": station}
    if event_date:
        kwargs["overrides"] = {"event_date": event_date}
    result = await skill.run(target, raw_task=f"assess flooding near {target}", **kwargs)

    attrs, opt = {}, {}
    for ev in result.evidence:
        a = ev.attributes or {}
        if "decision_indices" in a:
            attrs = a["decision_indices"]
        if a.get("optimization_status"):
            opt = a
    return {
        "completed": result.status == "completed",
        "cdri_percent": attrs.get("cdri_percent"),
        "cdri_label": attrs.get("cdri_risk_label"),
        "band": (attrs.get("uncertainty") or {}).get("interval_percent"),
        "exposed": (attrs.get("inputs") or {}).get("affected_population"),
        "exposed_interval": (attrs.get("inputs") or {}).get(
            "affected_population_interval"
        ),
        "data_gaps": attrs.get("data_gaps") or [],
        "issue_codes": sorted({i.code for i in result.validation_issues}),
        "opt_status": opt.get("optimization_status"),
        "recommended": ((opt.get("recommended_plan") or {}).get("plan_id")),
        "vwun": (opt.get("equity_ledger") or {}).get(
            "vulnerability_weighted_unmet_need"
        ),
    }


async def _run_naive(scenario: str) -> dict:
    mcp = FakeMCP()
    _apply_fault(mcp, scenario)
    target, station, _ = _scenario_args(scenario)
    return await NaiveBaseline(coverage_radius_km=2.0).run(
        mcp, target=target, station_id=station
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_baseline_comparison_row(tmp_path, monkeypatch, scenario):
    """One assertion block per scenario row of the comparison table."""
    naive = await _run_naive(scenario)
    ours = await _run_ours(monkeypatch, tmp_path, scenario)
    _print_row(scenario, naive, ours)

    assert ours["completed"], f"[{scenario}] reference pipeline failed"
    assert naive["status"] == "completed"

    if scenario == "S0-nominal":
        # Nominal conventions differ (naive: unclipped extent over a
        # circular buffer → inflated severity) but land in the SAME
        # label band (naive 4.26 vs ours 4.4); the honesty divergence
        # shows up under faults.
        assert abs(naive["cdri_percent"] - ours["cdri_percent"]) <= 1.5
        assert naive["band"] is None and ours["band"] is not None
    elif scenario == "S1-metadata-503":
        assert "STATION_METADATA_UNAVAILABLE" in ours["issue_codes"]
        # naive keeps recommending allocation from unverified geography
        assert naive["allocation_emitted"] is True
    elif scenario == "S2-no-nwps":
        # naive: confident Low, nothing disclosed; ours: degraded label
        assert "degraded" not in naive["cdri_label"]
        assert "degraded" in (ours["cdri_label"] or "")
        assert "NWPS_GAUGE_UNAVAILABLE" in ours["issue_codes"]
    elif scenario == "S3-far-station":
        # naive recommends allocation ~3000 km from the gauge
        assert naive["allocation_emitted"] is True
        assert naive["station_distance_km"] > 1000
        assert ours["opt_status"] != "optimized"
    elif scenario == "S4-no-extent":
        # naive: dangerously optimistic zeros
        assert naive["exposed_population"] == 0
        assert naive["vwun_recommended"] == 0
        # ours: degradation disclosed, unmet need NOT collapsed to zero
        assert "flooded_area_km2" in ours["data_gaps"]
        assert "degraded" in (ours["cdri_label"] or "")
    elif scenario == "S5-stale-sar":
        # naive silently consumes the misaligned scene
        assert naive["exposed_population"] > 0
        # ours rejects it at the time-alignment gate
        assert "SAR_EXTENT_STALE" in ours["issue_codes"]


def _print_row(scenario: str, naive: dict, ours: dict) -> None:
    def f(x, nd=2):
        return "-" if x is None else round(x, nd)

    print(f"\n=== {scenario} ===")
    print(f"  naive  : CDRI {f(naive['cdri_percent'])}% ({naive['cdri_label']}, "
          f"no band) | exposed {f(naive['exposed_population'], 0)} | "
          f"VWUN {f(naive['vwun_recommended'], 1)} | plan "
          f"{naive['recommended_plan']} | alloc={naive['allocation_emitted']} | "
          f"disclosed 0 / silent {len(naive['silent_issues'])}: "
          f"{naive['silent_issues']}")
    print(f"  ours   : CDRI {f(ours['cdri_percent'])}% ({ours['cdri_label']}, "
          f"band {ours['band']}) | exposed {f(ours['exposed'], 0)} "
          f"interval {ours['exposed_interval']} | VWUN {f(ours['vwun'], 1)} | "
          f"plan {ours['recommended']} | opt={ours['opt_status']} | "
          f"issues {ours['issue_codes']}")
