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
    }
