import json

import pytest

from app.community import (
    annotate_plan_community,
    enrich_resource_plans,
    load_facility_capability_overrides,
    normalize_community_requirements,
)


def test_normalize_requirements_rejects_unknown_and_duplicates():
    assert normalize_community_requirements([
        "safe_transport", "not_supported", "safe_transport"
    ]) == ["safe_transport"]


def test_missing_facility_capability_remains_unknown():
    plans = [{
        "plan_id": "p1",
        "allocations": [{"resource": "Shelter A", "type": "shelter"}],
    }]
    enrich_resource_plans(plans, overrides={})
    annotate_plan_community(plans, ["temporary_shelter"])

    summary = plans[0]["community_summary"]
    assert summary["unknown_requirements"] == ["temporary_shelter"]
    assert summary["unmet_requirements"] == []
    assert summary["live_feed_used"] is False


def test_sourced_override_matches_facility_and_satisfies_requirement(tmp_path):
    config = tmp_path / "community_facilities.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "facilities": [{
            "facility_id": "osm:node:123",
            "name": "Community Center",
            "source": "county emergency facility registry",
            "source_url": "https://example.test/facility/123",
            "capabilities": {
                "temporary_shelter": True,
                "wheelchair_accessible": True,
            },
        }],
    }), encoding="utf-8")
    overrides = load_facility_capability_overrides(config)
    plans = [{
        "plan_id": "p1",
        "allocations": [{
            "facility_id": "osm:node:123",
            "resource": "Community Center",
            "type": "shelter",
        }],
    }]

    enrich_resource_plans(plans, overrides=overrides)
    annotate_plan_community(
        plans,
        ["temporary_shelter", "special_population_support"],
        source="field_report",
    )

    summary = plans[0]["community_summary"]
    assert summary["community_requirements_met"] == 2
    assert summary["community_unknown_count"] == 0
    assert summary["source"] == "field_report"
    evidence = plans[0]["allocations"][0]["community_capability_evidence"]
    assert any(
        item.get("source") == "county emergency facility registry"
        for item in evidence
    )


def test_confirmed_false_is_unmet_only_when_all_matching_capabilities_known():
    plans = [{
        "plan_id": "p1",
        "allocations": [{
            "resource": "Transport A",
            "community_capabilities": {
                "pickup_service": False,
                "accessible_transport": False,
            },
        }],
    }]
    enrich_resource_plans(plans, overrides={})
    annotate_plan_community(plans, ["safe_transport"])
    assert plans[0]["community_summary"]["unmet_requirements"] == [
        "safe_transport"
    ]


@pytest.mark.asyncio
async def test_offline_demo_runs_community_matching_end_to_end():
    from app.demo_fixtures import run_demo_assessment

    requirements = [
        "emergency_rescue",
        "safe_transport",
        "temporary_shelter",
        "special_population_support",
    ]
    result, _ = await run_demo_assessment(
        event_date="2019-09-19",
        community_requirements=requirements,
        community_source="historical_replay",
        community_note="offline acceptance test",
    )
    assert result.status == "completed"

    optimization = next(
        ev.attributes
        for ev in result.evidence
        if (ev.attributes or {}).get("optimization_status") is not None
    )
    assert optimization["community_input"] == {
        "requirements": requirements,
        "source": "historical_replay",
        "note": "offline acceptance test",
        "live_feed_used": False,
    }
    scenarios = {
        item["scenario_id"]: item
        for item in optimization["plan_scenarios"]
    }
    community = scenarios["community_compatible"]
    assert community["metrics"]["community_requirements_met"] == 4
    assert community["metrics"]["community_unknown_count"] == 0
