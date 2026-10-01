import pytest

from app.historical_validation import (
    load_default_manifest,
    validate_full_agent_output,
    validate_manifest,
)


def test_held_out_event_is_separate_and_passes_declared_scope():
    result = validate_manifest(load_default_manifest())

    assert result["status"] == "passed"
    assert result["held_out_event_count"] == 5
    held_out = result["results"][0]
    assert held_out["independent_of_development_set"] is True
    assert held_out["event_id"] not in result["development_event_ids"]
    assert "CDRI categorical band cut points" in held_out["not_validated"]


def test_manifest_schema_version_is_enforced():
    manifest = load_default_manifest()
    manifest["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest(manifest)


def test_missing_event_field_raises_with_field_name():
    manifest = load_default_manifest()
    held_out = next(
        item
        for item in manifest["events"]
        if item.get("role") == "held_out_validation"
    )
    del held_out["flood_categories_ft"]
    with pytest.raises(ValueError, match="flood_categories_ft"):
        validate_manifest(manifest)


def test_result_discloses_the_schema_version_it_validated():
    result = validate_manifest(load_default_manifest())

    assert result["schema_version"] == 2


def test_held_out_sample_covers_all_five_hydrologic_strata():
    result = validate_manifest(load_default_manifest())

    assert result["balanced_severity_coverage"] is True
    assert result["severity_stratum_counts"] == {
        "below_action": 1,
        "action": 1,
        "minor": 1,
        "moderate": 1,
        "major": 1,
    }
    assert result["full_agent_event_ids"] == [
        "imelda_friendswood_2019",
        "ida_manville_2021",
    ]


def test_manifest_rejects_mislabeled_severity_stratum():
    manifest = load_default_manifest()
    held_out = next(
        item
        for item in manifest["events"]
        if item.get("role") == "held_out_validation"
    )
    held_out["severity_stratum"] = "major"

    with pytest.raises(ValueError, match="severity_stratum"):
        validate_manifest(manifest)


def test_full_agent_acceptance_checks_the_complete_artifact_chain():
    event = next(
        item
        for item in load_default_manifest()["events"]
        if item.get("event_id") == "imelda_friendswood_2019"
    )
    output = {
        "status": "completed",
        "structured": {
            "station_id": "08077600",
            "assessment_mode": "historical",
            "water_level": 11.64,
            "hydrologic_category": "action",
            "extent_provenance": "satellite_sar",
            "population_affected": 6211,
        },
        "optimization": {
            "status": "optimized",
            "recommended_plan_id": "shelter_top3",
            "plan_scenarios": [
                {"scenario_id": "balanced"},
                {"scenario_id": "shorter_transfer"},
            ],
        },
    }

    result = validate_full_agent_output(event, output)

    assert result["status"] == "passed"
    assert all(check["passed"] for check in result["checks"])
    assert "does not validate extent accuracy" in result["claim_limit"]


def test_full_agent_acceptance_fails_when_resource_plan_is_missing():
    event = next(
        item
        for item in load_default_manifest()["events"]
        if item.get("event_id") == "imelda_friendswood_2019"
    )
    output = {
        "status": "completed",
        "structured": {
            "station_id": "08077600",
            "assessment_mode": "historical",
            "water_level": 11.64,
            "hydrologic_category": "action",
            "extent_provenance": "satellite_sar",
            "population_affected": 6211,
        },
        "optimization": {},
    }

    result = validate_full_agent_output(event, output)

    assert result["status"] == "failed"
    failed_names = {
        check["name"] for check in result["checks"] if not check["passed"]
    }
    assert "resource_optimization_completed" in failed_names
    assert "pareto_recommendation_produced" in failed_names
