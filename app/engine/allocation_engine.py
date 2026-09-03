"""Allocation Engine — resource plan generation and Pareto optimization.

Configuration (objectives, weights, plan-count cap) is injected by the
Skill; no MCP, no state.
"""

from __future__ import annotations


from typing import Any

from ..utils import geodesic_km, safe_float as _safe_float
from .pareto_engine import (
    pareto_frontier,
    select_best_pareto_plan,
)


class AllocationEngine:
    """Pure computation: resources + fused evidence in, ranked plans out."""

    def __init__(
        self,
        objectives: list[dict[str, Any]],
        weights: dict[str, float],
        max_plan_combinations: int,
        vulnerability_coverage_radius_km: float = 10.0,
    ):
        self.objectives = objectives
        self.weights = weights
        self.max_plan_combinations = max_plan_combinations
        # Coverage radius (km) for the vulnerability_coverage objective:
        # a tract's population counts as covered when its centroid lies
        # within this radius of any allocated facility.
        self.vulnerability_coverage_radius_km = (
            vulnerability_coverage_radius_km
        )

    def generate_plans(
        self,
        resources: list[dict[str, Any]],
        fused_evidence: dict[str, Any],
        svi_tracts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:

        if not resources:
            return []

        objectives = (
            self.objectives
        )

        if not objectives:
            raise RuntimeError(
                "RESOURCE_ALLOCATION_OBJECTIVES_JSON "
                "is not configured."
            )

        plans = []

        for resource in resources:

            if not isinstance(
                resource,
                dict,
            ):
                continue

            plan_id = resource.get(
                "plan_id"
            )

            allocations = resource.get(
                "allocations"
            )

            objective_values = resource.get(
                "objectives"
            )

            if not plan_id:
                continue

            if not isinstance(
                allocations,
                list,
            ):
                continue

            if not isinstance(
                objective_values,
                dict,
            ):
                continue

            valid = True

            normalized_objectives = {}

            for objective in objectives:

                name = objective.get(
                    "name"
                )

                if not name:
                    valid = False
                    break

                value = _safe_float(
                    objective_values.get(
                        name
                    )
                )

                if value is None:
                    valid = False
                    break

                normalized_objectives[
                    name
                ] = value

            if not valid:
                continue

            plans.append(
                {
                    "plan_id": str(
                        plan_id
                    ),
                    "allocations": allocations,
                    "objectives": (
                        normalized_objectives
                    ),
                    "metadata": resource.get(
                        "metadata",
                        {},
                    ),
                }
            )

        if len(plans) > self.max_plan_combinations:

            plans = plans[
                :self.max_plan_combinations
            ]

        self._apply_vulnerability_coverage(plans, svi_tracts)

        return plans

    # ============================================================
    # Equity objective: population-weighted SVI coverage
    # ============================================================

    @staticmethod
    def _tract_centroid(tract: dict[str, Any]) -> tuple[float, float] | None:
        """Mean center of an ArcGIS-style rings geometry (same convention as geometry_engine)."""

        geometry = tract.get("geometry") or {}
        rings = geometry.get("rings") if isinstance(geometry, dict) else None
        if not (isinstance(rings, list) and rings):
            return None
        ring = rings[0]
        if not (
            isinstance(ring, list)
            and len(ring) >= 3
            and all(
                isinstance(pt, (list, tuple)) and len(pt) >= 2
                for pt in ring[:3]
            )
        ):
            return None
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    def _apply_vulnerability_coverage(
        self,
        plans: list[dict[str, Any]],
        svi_tracts: list[dict[str, Any]] | None,
    ) -> None:
        """
        Compute vulnerability_coverage for each plan:

            sum(population * svi | centroid within R km of any allocated facility)
            -----------------------------------------------------------------------
                    sum(population * svi | all valid tracts)

        Preconditions (any failure skips the whole computation; no
        half-trusted data):
        - svi_tracts non-empty, with population/svi/geometry
        - every plan's allocations carry lat/lon
        """
        if not plans or not svi_tracts:
            return

        weighted: list[tuple[tuple[float, float], float]] = []
        for tract in svi_tracts:
            if not isinstance(tract, dict):
                continue
            population = _safe_float(tract.get("population"))
            svi = _safe_float(tract.get("svi"))
            centroid = self._tract_centroid(tract)
            if population is None or svi is None or centroid is None:
                continue
            weighted.append((centroid, population * max(0.0, svi)))

        if not weighted:
            return

        radius = self.vulnerability_coverage_radius_km

        for plan in plans:
            allocations = plan.get("allocations") or []
            points = [
                (a["lat"], a["lon"])
                for a in allocations
                if _safe_float(a.get("lat")) is not None
                and _safe_float(a.get("lon")) is not None
            ]
            if len(points) != len(allocations) or not points:
                return  # any plan missing coordinates -> skip the whole computation (honesty first)

            covered = 0.0
            for (t_lat, t_lon), weight in weighted:
                for f_lat, f_lon in points:
                    if geodesic_km(t_lat, t_lon, f_lat, f_lon) <= radius:
                        covered += weight
                        break

            total = sum(w for _, w in weighted)
            plan["objectives"]["vulnerability_coverage"] = (
                round(covered / total, 4) if total > 0 else 0.0
            )
            plan["vulnerability_coverage_computed"] = True

    def effective_objectives(
        self,
        plans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Configured objectives, plus vulnerability_coverage when actually computed for the plans."""

        names = {o["name"] for o in self.objectives}
        if (
            plans
            and all(p.get("vulnerability_coverage_computed") for p in plans)
            and "vulnerability_coverage" not in names
        ):
            return list(self.objectives) + [
                {"name": "vulnerability_coverage", "direction": "maximize"}
            ]
        return list(self.objectives)

    # ============================================================
    # Pareto-based resource optimization
    # ============================================================



    def optimize(
            self,
            plans: list[dict[str, Any]],
            weights_override: dict[str, float] | None = None,
        ) -> dict[str, Any]:

            if not plans:
                return {
                    "status":          "no_feasible_plan",
                    "pareto_frontier": [],
                    "recommended_plan": None,
                }

            # Effective objectives = configured objectives plus the equity
            # coverage objective when actually computed; it is never added
            # otherwise, so weight injection cannot pretend to take effect.
            objectives = self.effective_objectives(plans)

            frontier = pareto_frontier(plans, objectives)

            if not frontier:
                return {
                    "status":          "no_pareto_solution",
                    "pareto_frontier": [],
                    "recommended_plan": None,
                }

            # weights_override is passed by run() when SVI data is
            # available; without it, fall back to the static
            # env-configured weights.
            weights = (
                weights_override
                if weights_override is not None
                else self.weights
            )

            recommended = select_best_pareto_plan(
                frontier,
                objectives,
                weights,
            )

            return {
                "status":               "optimized",
                "candidate_plan_count": len(plans),
                "pareto_plan_count":    len(frontier),
                "pareto_frontier":      frontier,
                "recommended_plan":     recommended,
                # Weights actually used, so SVI application is auditable.
                "weights_used":         weights,
                "svi_weight_applied":   (
                    weights_override is not None
                    and any(o["name"] == "vulnerability_coverage" for o in objectives)
                    and float(weights_override.get("vulnerability_coverage", 0.0) or 0.0) > 0
                ),
            }

    # ============================================================
    # Build allocation evidence
    # ============================================================



    @staticmethod
    @staticmethod
    
    def optimization_allowed(
    
        fused_evidence: dict[str, Any],
    
        station_spatial_verified: bool,
    
    ) -> tuple[bool, list[str]]:
    
    
    
        reasons: list[str] = []
    
    
    
        if not station_spatial_verified:
    
            reasons.append("Target-station spatial relationship is not verified.")
    
    
    
        source_count = int(fused_evidence.get("source_count", 0) or 0)
    
        if source_count <= 0:
    
            reasons.append("No verified multi-source evidence is available.")
    
    
    
        has_measurements    = bool(fused_evidence.get("fused_measurements"))
    
        has_alerts          = bool(fused_evidence.get("alerts"))
    
        has_forecasts       = bool(fused_evidence.get("forecasts"))
    
    
    
        # Spatial flood-extent evidence
    
        has_spatial_extent  = any(
    
            item.get("evidence_type") == "spatial_extent"
    
            for item in fused_evidence.get("observations", [])
    
        )
    
    
    
        if not (has_measurements or has_alerts or has_forecasts or has_spatial_extent):
    
            reasons.append(
    
                "No actionable flood-related evidence category "
    
                "(measurement / alert / forecast / spatial_extent) is available."
    
            )
    
    
    
        return (len(reasons) == 0, reasons)
    
    
    
    # ============================================================
    
    # Main workflow
    
    # ============================================================
    
    

