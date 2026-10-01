#!/usr/bin/env python3
"""Run the held-out historical validation manifest."""

from __future__ import annotations

import argparse
import json

from app.historical_validation import (
    load_default_manifest,
    refresh_usgs_peak,
    validate_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-usgs",
        action="store_true",
        help="verify stored held-out peaks against the official USGS API",
    )
    args = parser.parse_args()
    manifest = load_default_manifest()
    refresh_checks = []
    if args.refresh_usgs:
        for event in manifest.get("events") or []:
            if event.get("role") != "held_out_validation":
                continue
            current = refresh_usgs_peak(event)
            refresh_checks.append(
                {
                    "event_id": event["event_id"],
                    **current,
                    "matches_snapshot": (
                        current["observed_peak_stage_ft"]
                        == event["observed_peak_stage_ft"]
                        and current["observed_peak_time_range"]
                        == event["observed_peak_time_range"]
                    ),
                }
            )
    result = validate_manifest(manifest)
    if refresh_checks:
        result["source_refresh_checks"] = refresh_checks
        if not all(item["matches_snapshot"] for item in refresh_checks):
            result["status"] = "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
