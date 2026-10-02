from app.plan_scenarios import build_plan_scenarios, load_plan_profiles


def test_profiles_select_existing_frontier_plans():
    curve = [
        {
            "plan_id": "balanced",
            "served_people": 80,
            "unmet_people": 20,
            "travel_distance_p95_km": 5,
            "vulnerability_weighted_unmet_need": 25,
            "cost": 10,
        },
        {
            "plan_id": "coverage",
            "served_people": 100,
            "unmet_people": 0,
            "travel_distance_p95_km": 8,
            "vulnerability_weighted_unmet_need": 15,
            "cost": 20,
        },
        {
            "plan_id": "nearby",
            "served_people": 70,
            "unmet_people": 30,
            "travel_distance_p95_km": 2,
            "vulnerability_weighted_unmet_need": 40,
            "cost": 8,
        },
    ]
    scenarios = build_plan_scenarios(curve, "balanced")
    selected = {item["scenario_id"]: item["plan_id"] for item in scenarios}
    assert selected["balanced"] == "balanced"
    assert selected["coverage_first"] == "coverage"
    assert selected["vulnerability_first"] == "coverage"
    assert selected["shorter_transfer"] == "nearby"


def test_profiles_are_maintained_in_config():
    profiles = load_plan_profiles()
    assert {profile["id"] for profile in profiles} >= {
        "balanced",
        "coverage_first",
        "vulnerability_first",
        "shorter_transfer",
        "community_compatible",
    }


def test_community_profile_prefers_sourced_match_within_frontier():
    curve = [
        {
            "plan_id": "unknown",
            "served_people": 100,
            "unmet_people": 0,
            "cost": 5,
            "community_requirements_total": 2,
            "community_requirements_met": 0,
            "community_unmet_count": 0,
            "community_unknown_count": 2,
        },
        {
            "plan_id": "matched",
            "served_people": 90,
            "unmet_people": 10,
            "cost": 10,
            "community_requirements_total": 2,
            "community_requirements_met": 2,
            "community_unmet_count": 0,
            "community_unknown_count": 0,
        },
    ]
    scenarios = build_plan_scenarios(curve, "unknown")
    selected = {item["scenario_id"]: item["plan_id"] for item in scenarios}
    assert selected["community_compatible"] == "matched"


def test_community_profile_is_hidden_without_event_requirements():
    curve = [{"plan_id": "p1", "community_requirements_total": 0}]
    scenarios = build_plan_scenarios(curve, "p1")
    assert "community_compatible" not in {
        item["scenario_id"] for item in scenarios
    }
