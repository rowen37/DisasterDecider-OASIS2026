"""Build operator-selectable plans from the already computed Pareto frontier.

Profiles live in ``config/plan_priorities.json`` so local teams can change the
decision choices without encoding community claims or facility exclusions in
Python.  A profile only ranks existing, evidence-backed plans; it never edits
the underlying allocation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "plan_priorities.json"


def load_plan_profiles(path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("profiles"), list):
        raise ValueError("plan priority configuration must use schema_version 1")
    profiles = []
    seen: set[str] = set()
    for raw in payload["profiles"]:
        if not isinstance(raw, dict):
            raise ValueError("each plan priority profile must be an object")
        profile_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        if not profile_id or not label or profile_id in seen:
            raise ValueError("plan priority profile ids and labels must be unique and non-empty")
        seen.add(profile_id)
        profiles.append(raw)
    return profiles


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pick_ranked(
    plans: list[dict[str, Any]], rank_by: list[dict[str, Any]]
) -> dict[str, Any] | None:
    eligible = [
        plan
        for plan in plans
        if all(_number(plan.get(rule.get("field"))) is not None for rule in rank_by)
    ]
    if not eligible:
        return None

    def key(plan: dict[str, Any]) -> tuple[float, ...]:
        values = []
        for rule in rank_by:
            value = _number(plan.get(rule["field"])) or 0.0
            values.append(-value if rule.get("direction") == "max" else value)
        return tuple(values)

    return min(eligible, key=key)


def _pick_community_compatible(
    plans: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick within the existing frontier; do not change Pareto objectives.

    A confirmed mismatch is worse than missing capability data.  Unknown is
    still penalized so that a plan with sourced evidence is preferred, but it
    is never silently treated as a confirmed service failure.
    """

    eligible = [
        plan
        for plan in plans
        if (_number(plan.get("community_requirements_total")) or 0) > 0
    ]
    if not eligible:
        return None

    def key(plan: dict[str, Any]) -> tuple[float, ...]:
        return (
            _number(plan.get("community_unmet_count")) or 0,
            _number(plan.get("community_unknown_count")) or 0,
            -(_number(plan.get("community_requirements_met")) or 0),
            _number(plan.get("unmet_people")) or 0,
            _number(plan.get("cost")) or 0,
        )

    return min(eligible, key=key)


def build_plan_scenarios(
    frontier_curve: list[dict[str, Any]],
    recommended_plan_id: str | None,
    profiles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    plans = [p for p in frontier_curve if isinstance(p, dict) and p.get("plan_id")]
    if not plans:
        return []
    by_id = {str(plan["plan_id"]): plan for plan in plans}
    scenarios = []
    for profile in profiles or load_plan_profiles():
        if profile.get("selector") == "recommended":
            selected = by_id.get(str(recommended_plan_id)) or plans[0]
        elif profile.get("selector") == "community_compatible":
            selected = _pick_community_compatible(plans)
        else:
            selected = _pick_ranked(plans, profile.get("rank_by") or [])
        if selected is None:
            continue
        scenarios.append(
            {
                "scenario_id": profile["id"],
                "label": profile["label"],
                "description": profile.get("description", ""),
                "plan_id": selected["plan_id"],
                "metrics": {
                    key: selected.get(key)
                    for key in (
                        "served_people",
                        "unmet_people",
                        "travel_distance_p95_km",
                        "vulnerability_weighted_unmet_need",
                        "equity_gap",
                        "vulnerability_coverage",
                        "cost",
                        "community_requirements_met",
                        "community_requirements_total",
                        "community_unmet_count",
                        "community_unknown_count",
                        "community_fit",
                        "community_summary",
                    )
                },
            }
        )
    return scenarios
