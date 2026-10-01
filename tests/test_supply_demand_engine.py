from __future__ import annotations

import pytest

from app.engine.allocation_engine import AllocationEngine
from app.engine.supply_demand_engine import SupplyDemandEngine


def _demand(tract_id, people, svi, lat, lon):
    return {
        "tract_id": tract_id,
        "exposed_population": people,
        "flooded_fraction": 1.0,
        "svi": svi,
        "centroid": (lat, lon),
    }


def _facility(name, capacity, lat, lon, assumed=False):
    return {
        "resource": name,
        "type": "shelter",
        "capacity": capacity,
        "capacity_assumed": assumed,
        "lat": lat,
        "lon": lon,
    }


def test_capacity_constraint_reports_unmet_demand_and_utilization():
    engine = SupplyDemandEngine(max_travel_km=20.0)
    result = engine.allocate(
        [
            _demand("high", 100, 0.9, 29.50, -95.10),
            _demand("low", 100, 0.2, 29.52, -95.10),
        ],
        [_facility("S1", 120, 29.51, -95.10)],
        vulnerability_weight=1.0,
    )

    assert result["status"] == "allocated"
    assert result["total_demand_people"] == 200
    assert result["served_people"] == 120
    assert result["unmet_people"] == 80
    assert result["coverage"] == pytest.approx(0.6)
    assert result["unmet_fraction"] == pytest.approx(0.4)
    assert result["facility_utilization"][0]["utilization"] == 1.0
    assert sum(item["assigned_people"] for item in result["assignments"]) == 120


def test_scarce_capacity_prioritizes_vulnerable_demand():
    engine = SupplyDemandEngine(max_travel_km=20.0)
    result = engine.allocate(
        [
            _demand("high", 100, 0.9, 29.50, -95.10),
            _demand("low", 100, 0.1, 29.50, -95.12),
        ],
        [_facility("S1", 100, 29.50, -95.11)],
        vulnerability_weight=1.0,
    )

    impacts = {item["tract_id"]: item for item in result["demand_impacts"]}
    assert impacts["high"]["assigned_people"] == 100
    assert impacts["low"]["assigned_people"] == 0
    assert result["vulnerability_weighted_unmet_need"] == pytest.approx(110.0)


def test_transfer_time_is_a_range_only_when_speed_bounds_are_configured():
    demand = [_demand("T1", 10, 0.5, 29.50, -95.10)]
    facility = [_facility("S1", 10, 29.50, -95.00)]
    without_bounds = SupplyDemandEngine(max_travel_km=20.0).allocate(
        demand, facility
    )
    assert without_bounds["transfer_time_estimate_range_minutes"] is None
    assert without_bounds["time_estimate_basis"] == "not_estimated"
    assert without_bounds["planning_gaps"] == ["transfer_time_estimate"]

    with_bounds = SupplyDemandEngine(
        max_travel_km=20.0,
        planning_speed_range_kmh=(20.0, 40.0),
    ).allocate(demand, facility)
    estimate = with_bounds["transfer_time_estimate_range_minutes"]
    assert estimate["min"] < estimate["max"]
    assert "not a live-traffic" in with_bounds["time_estimate_basis"]
    assert with_bounds["planning_gaps"] == []


def test_allocation_engine_recomputes_plan_objectives_from_assignment():
    engine = AllocationEngine(
        objectives=[
            {"name": "coverage", "direction": "maximize"},
            {"name": "response_time", "direction": "minimize"},
            {"name": "unmet_demand", "direction": "minimize"},
        ],
        weights={"coverage": 0.5, "response_time": 0.2, "unmet_demand": 0.3},
        max_plan_combinations=10,
        vulnerability_coverage_radius_km=20.0,
    )
    resources = [
        {
            "plan_id": "small",
            "allocations": [_facility("S1", 40, 29.50, -95.10)],
            "objectives": {
                "coverage": 1.0,
                "response_time": 99.0,
                "unmet_demand": 0.0,
            },
            "metadata": {},
        },
        {
            "plan_id": "large",
            "allocations": [_facility("S2", 100, 29.50, -95.10)],
            "objectives": {
                "coverage": 0.0,
                "response_time": 99.0,
                "unmet_demand": 1.0,
            },
            "metadata": {},
        },
    ]
    demand = [_demand("T1", 100, 0.8, 29.50, -95.10)]

    plans = engine.generate_plans(
        resources,
        {},
        demand_records=demand,
        vulnerability_weight=1.0,
    )
    by_id = {plan["plan_id"]: plan for plan in plans}

    assert by_id["small"]["objectives"]["coverage"] == pytest.approx(0.4)
    assert by_id["small"]["objectives"]["unmet_demand"] == pytest.approx(0.6)
    assert by_id["large"]["objectives"]["coverage"] == 1.0
    assert by_id["large"]["objectives"]["unmet_demand"] == 0.0
    assert by_id["large"]["objectives"]["vulnerability_coverage"] == 1.0
    assert by_id["large"]["supply_demand"]["served_people"] == 100


def test_geodesic_fallback_excludes_response_only_facilities():
    engine = SupplyDemandEngine(max_travel_km=20.0)
    fire_station = _facility("F1", 100, 29.50, -95.10)
    fire_station["type"] = "fire_station"

    result = engine.allocate(
        [_demand("T1", 100, 0.8, 29.50, -95.10)],
        [fire_station],
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_eligible_facilities"
    assert result["excluded_facility_types"] == ["fire_station"]


def test_unbounded_assignment_uses_discovered_facilities_beyond_two_km():
    engine = SupplyDemandEngine()
    result = engine.allocate(
        [_demand("T1", 100, 0.8, 40.89, -74.08)],
        [_facility("S1", 60, 40.89, -73.93)],
    )

    assert result["status"] == "allocated"
    assert result["served_people"] == 60
    assert result["coverage"] == pytest.approx(0.6)
    assert result["max_travel_km"] is None


def test_zero_service_and_response_only_plans_cannot_be_recommended():
    engine = AllocationEngine(
        objectives=[
            {"name": "risk_reduction", "direction": "maximize"},
            {"name": "coverage", "direction": "maximize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "unmet_demand", "direction": "minimize"},
        ],
        weights={
            "risk_reduction": 0.4,
            "coverage": 0.3,
            "cost": 0.1,
            "unmet_demand": 0.2,
        },
        max_plan_combinations=10,
        vulnerability_coverage_radius_km=2.0,
    )
    response_only = _facility("F1", 1000, 40.89, -74.08)
    response_only["type"] = "fire_station"
    resources = [
        {
            "plan_id": "fire_expensive",
            "allocations": [response_only],
            "objectives": {
                "risk_reduction": 1.0,
                "coverage": 1.0,
                "cost": 500.0,
                "unmet_demand": 0.0,
            },
            "metadata": {"severity_water_ratio_used": 1.0},
        },
        {
            "plan_id": "shelter_valid",
            "allocations": [_facility("S1", 40, 40.89, -73.93)],
            "objectives": {
                "risk_reduction": 0.0,
                "coverage": 0.0,
                "cost": 10.0,
                "unmet_demand": 1.0,
            },
            "metadata": {"severity_water_ratio_used": 1.0},
        },
    ]
    plans = engine.generate_plans(
        resources,
        {},
        demand_records=[_demand("T1", 100, 0.8, 40.89, -74.08)],
    )
    result = engine.optimize(plans)

    assert result["status"] == "optimized"
    assert result["feasible_plan_count"] == 1
    assert result["recommended_plan"]["plan_id"] == "shelter_valid"
    assert result["recommended_plan"]["objectives"]["coverage"] == 0.4
    assert result["recommended_plan"]["objectives"]["risk_reduction"] == 0.4
