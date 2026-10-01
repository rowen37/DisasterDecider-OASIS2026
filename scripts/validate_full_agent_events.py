#!/usr/bin/env python3
"""Run reproducible, real-service end-to-end historical Agent checks.

This is intentionally separate from ``validate_historical_events``.  The
historical validator checks official peaks and categories for the balanced
five-stratum sample.  This runner executes the full Agent pipeline for the
manifest events explicitly marked ``validation_tier=full_agent``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.experiment import ExperimentLogger
from app.historical_validation import (
    load_default_manifest,
    validate_full_agent_output,
)
from app.hitl import AdaptiveHITL
from app.main import _extract_structured_data
from app.models import RunState
from app.ops import OpsExecutor
from app.skills.flood_skill import FloodSkill
from app.verification import Verifier


def _compact_output(
    event: dict[str, Any],
    result: Any,
    state: RunState,
) -> dict[str, Any]:
    structured = _extract_structured_data(result, state)
    optimization = structured.get("resource_optimization") or {}
    equity = optimization.get("equity_ledger") or {}
    time_alignment = structured.get("time_alignment") or {}
    return {
        "event_id": event["event_id"],
        "run_id": state.run_id,
        "status": result.status,
        "summary": result.summary,
        "validation_issues": [
            item.model_dump(mode="json") for item in result.validation_issues
        ],
        "structured": {
            "station_id": structured.get("station_id"),
            "station_name": structured.get("station_name"),
            "water_level": structured.get("water_level"),
            "observation_time": structured.get("observation_time"),
            "observation_semantics": structured.get("observation_semantics"),
            "window_summary": structured.get("window_summary"),
            "assessment_mode": structured.get("assessment_mode"),
            "hydrologic_category": structured.get("hydrologic_category"),
            "extent_provenance": structured.get("extent_provenance"),
            "extent_model": structured.get("extent_model"),
            "flooded_area_km2": structured.get("flooded_area_km2"),
            "population_affected": structured.get("population_affected"),
            "population_affected_source": structured.get(
                "population_affected_source"
            ),
            "infrastructure_counts": structured.get("infrastructure_counts"),
            "svi": structured.get("svi"),
            "svi_tracts": structured.get("svi_tracts"),
            "time_alignment": {
                "mode": time_alignment.get("mode"),
                "event_date": time_alignment.get("event_date"),
                "mismatched_sources": time_alignment.get(
                    "mismatched_sources", []
                ),
                "skipped_current_only_sources": time_alignment.get(
                    "skipped_current_only_sources", []
                ),
            },
        },
        "optimization": {
            "status": optimization.get("status"),
            "recommended_plan_id": optimization.get("recommended_plan_id"),
            "served_people": optimization.get("served_people"),
            "unmet_people": optimization.get("unmet_people"),
            "travel_distance_p95_km": optimization.get(
                "travel_distance_p95_km"
            ),
            "planning_gaps": optimization.get("planning_gaps", []),
            "equity_ledger": {
                key: equity.get(key)
                for key in (
                    "vulnerability_weighted_unmet_need",
                    "normalized_vulnerability_weighted_unmet_need",
                    "equity_gap",
                    "equity_gap_note",
                    "coverage_concentration_index",
                    "coverage_concentration_index_note",
                    "equity_threshold",
                    "vulnerability_weight",
                    "demand_tract_count",
                )
            },
            "plan_scenarios": optimization.get("plan_scenarios", []),
        },
        "evidence_ids": [item.evidence_id for item in result.evidence],
    }


async def _run_event(
    event: dict[str, Any],
    *,
    config_path: str,
) -> dict[str, Any]:
    state = RunState(run_id=f"historical-validation-{uuid.uuid4()}")
    hitl = AdaptiveHITL(state)
    hitl.enable_web_mode()
    ops = OpsExecutor(config_path)
    try:
        await ops.connect()
        skill = FloodSkill(
            state,
            Verifier(),
            hitl,
            ops,
            ExperimentLogger(),
        )
        event_date = event["event_date"]
        result = await skill.run(
            target=event["location"],
            station_id=event["station_id"],
            raw_task=(
                "Run a full historical flood impact and resource-allocation "
                f"assessment for {event['location']} on {event_date} at "
                f"USGS {event['station_id']}."
            ),
            overrides={"event_date": event_date},
        )
        output = _compact_output(event, result, state)
        output["acceptance"] = validate_full_agent_output(event, output)
        return output
    finally:
        await ops.close()


async def _main_async(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file)
    os.environ["HITL_TIMEOUT_SECONDS"] = str(args.hitl_timeout)
    manifest = load_default_manifest()
    candidates = [
        event
        for event in manifest.get("events", [])
        if event.get("role") == "held_out_validation"
        and event.get("validation_tier") == "full_agent"
    ]
    requested = set(args.event or [])
    events = [
        event for event in candidates
        if not requested or event["event_id"] in requested
    ]
    unknown = requested - {event["event_id"] for event in candidates}
    if unknown:
        raise SystemExit(
            "unknown full-agent validation event(s): " + ", ".join(sorted(unknown))
        )

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for event in events:
        output = await _run_event(event, config_path=args.mcp_config)
        outputs.append(output)
        if output_dir:
            path = output_dir / f"{event['event_id']}.json"
            path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    report = {
        "experiment": "full_agent_historical_acceptance",
        "event_count": len(outputs),
        "status": (
            "passed"
            if outputs
            and all(item["acceptance"]["status"] == "passed" for item in outputs)
            else "failed"
        ),
        "results": outputs,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        action="append",
        help="event_id to run; repeat for multiple events (default: all full_agent events)",
    )
    parser.add_argument("--output-dir", help="optional directory for compact JSON artifacts")
    parser.add_argument("--mcp-config", default="config/mcp.json")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--hitl-timeout", type=float, default=2.0)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
