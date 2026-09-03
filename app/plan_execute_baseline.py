"""PlanExecuteBaseline — plan-then-execute scaffold baseline (§4.3,
third condition).

A DORA-inspired agentic scaffold: the FULL tool chain is planned
upfront — before any observation exists, which is the scaffold's
defining property — and then executed step by step, threading earlier
observations into later arguments. Per the scaffold's contract it
NEVER re-plans: a failed observation does not revise the remaining
chain, whose steps execute against broken context with planned
defaults substituted silently (``@step.field`` references that resolve
to nothing become 0.0).

Scoring conventions are NaiveBaseline's, inherited verbatim through
``_score`` — no verifier, no evidence state, no degradation policy,
allocation always allowed. The scaffold changes WHO sequences the
calls (an upfront plan instead of hand-written code), not what happens
when evidence breaks.

``execution_report`` (instrumentation, not part of any real agent's
answer) records: plan_steps, replans (0 by construction),
first_failed_step, steps_after_first_failure, and
arg_default_substitutions.
"""

from __future__ import annotations

import re
from typing import Any

from .naive_baseline import NaiveBaseline, _call_json


class PlanExecuteBaseline(NaiveBaseline):
    """Plans the full tool chain upfront; executes without re-planning."""

    name = "plan_execute"

    # Planned before any observation: literals + "@step.field" references.
    PLAN: tuple[tuple[str, str, dict[str, Any]], ...] = (
        ("geocode", "geocode_location", {"place_name": "{target}"}),
        ("observation", "get_flood_observation",
         {"station_id": "{station_id}",
          "latitude": "@geocode.latitude",
          "longitude": "@geocode.longitude"}),
        ("metadata", "get_station_metadata", {"station_id": "{station_id}"}),
        ("nwps", "get_nwps_gauge", {"station_id": "{station_id}"}),
        ("extent", "get_flood_extent",
         {"latitude": "@geocode.latitude", "longitude": "@geocode.longitude",
          "buffer_km": 10, "observation_date": None}),
        ("svi", "get_social_vulnerability",
         {"latitude": "@geocode.latitude", "longitude": "@geocode.longitude",
          "radius_km": 10}),
        ("infra", "get_critical_infrastructure",
         {"latitude": "@geocode.latitude", "longitude": "@geocode.longitude",
          "radius_km": 10}),
        ("resources", "get_available_resources",
         {"latitude": "@geocode.latitude", "longitude": "@geocode.longitude"}),
    )

    def __init__(self, coverage_radius_km: float = 2.0):
        super().__init__(coverage_radius_km)
        self.execution_report: dict[str, Any] = {}

    async def run(self, mcp: Any, target: str, station_id: str) -> dict[str, Any]:
        result = self._score(await self._observe(mcp, target, station_id))
        result.update(self.execution_report)
        return result

    def _resolve(self, template: Any, ctx: dict[str, Any]) -> Any:
        """Resolve a planned argument; unresolvable references default to 0."""
        if isinstance(template, str) and template.startswith("@"):
            ref, field = template[1:].split(".", 1)
            value = (ctx.get(ref) or {}).get(field)
            if value is None:
                self.execution_report["arg_default_substitutions"] += 1
                return 0.0
            return value
        return template

    async def _observe(  # type: ignore[override]
        self, mcp: Any, target: str, station_id: str
    ) -> dict[str, Any]:
        self.execution_report = {
            "scaffold": "plan_then_execute",
            "plan": [tool for _, tool, _ in self.PLAN],
            "plan_steps": len(self.PLAN),
            "replans": 0,
            "first_failed_step": None,
            "steps_after_first_failure": 0,
            "arg_default_substitutions": 0,
        }

        silent: list[str] = []
        ctx: dict[str, dict[str, Any]] = {}
        literals = {"{target}": target, "{station_id}": station_id}
        for position, (step, tool, arg_templates) in enumerate(self.PLAN):
            args = {
                key: (
                    literals.get(value)
                    if value in literals
                    else self._resolve(value, ctx)
                )
                for key, value in arg_templates.items()
            }
            if step == "geocode":
                # Same minimal perception as the dashboard baseline.
                raw = await mcp.call(tool, args)
                m = re.search(
                    r"纬度\s*(-?\d+\.?\d*).*?经度\s*(-?\d+\.?\d*)", str(raw)
                )
                if not m:
                    silent.append(f"geocode failed for {target!r} → coords (0, 0)")
                    lat, lon = 0.0, 0.0
                    self._note_failure(position, step)
                else:
                    lat, lon = float(m.group(1)), float(m.group(2))
                ctx[step] = {"latitude": lat, "longitude": lon}
                continue

            observation = await self._call_and_note(mcp, tool, args, position, step)
            ctx[step] = observation

        return {
            "target": target, "station_id": station_id,
            "lat": ctx["geocode"]["latitude"], "lon": ctx["geocode"]["longitude"],
            "silent": silent,
            "observation": ctx["observation"], "metadata": ctx["metadata"],
            "nwps": ctx["nwps"], "extent": ctx["extent"], "svi": ctx["svi"],
            "infra": ctx["infra"], "resources": ctx["resources"],
        }

    async def _call_and_note(
        self, mcp: Any, tool: str, args: dict[str, Any], position: int, step: str
    ) -> dict[str, Any]:
        observation = await _call_json(mcp, tool, args)
        if observation.get("status") != "ok":
            self._note_failure(position, step)
        return observation

    def _note_failure(self, position: int, step: str) -> None:
        report = self.execution_report
        if report["first_failed_step"] is None:
            report["first_failed_step"] = step
            report["steps_after_first_failure"] = len(self.PLAN) - position - 1
