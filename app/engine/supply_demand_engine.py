"""Capacity-constrained allocation of exposed demand to facilities.

The engine is deliberately independent of MCP and application state.  It
accepts the tract-level demand records already produced by ``social_good``
and a candidate plan's facilities, then solves a minimum-cost flow problem.
Unserved flow is allowed at an explicit penalty so insufficient capacity is
reported instead of making the optimization infeasible.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from ..utils import geodesic_km, safe_float as _safe_float
from .evacuation_policy import (
    DEFAULT_ELIGIBLE_FACILITY_TYPES,
    normalize_facility_type,
    normalize_facility_types,
)


class SupplyDemandEngine:
    """Pure capacity-aware demand assignment."""

    def __init__(
        self,
        max_travel_km: float | None = None,
        planning_speed_range_kmh: tuple[float, float] | None = None,
        eligible_facility_types: tuple[str, ...] | list[str] | str | None = None,
    ) -> None:
        if max_travel_km is not None and (
            not math.isfinite(max_travel_km) or max_travel_km <= 0
        ):
            raise ValueError(
                "max_travel_km must be finite and positive when provided"
            )
        if planning_speed_range_kmh is not None:
            low, high = planning_speed_range_kmh
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low <= 0
                or high <= 0
                or low > high
            ):
                raise ValueError(
                    "planning_speed_range_kmh must contain positive low/high values"
                )
        self.max_travel_km = max_travel_km
        self.planning_speed_range_kmh = planning_speed_range_kmh
        self.eligible_facility_types = normalize_facility_types(
            eligible_facility_types or DEFAULT_ELIGIBLE_FACILITY_TYPES
        )

    @staticmethod
    def _weighted_percentile(
        values: list[tuple[float, int]], percentile: float
    ) -> float | None:
        if not values:
            return None
        ordered = sorted(values, key=lambda item: item[0])
        total = sum(weight for _, weight in ordered)
        if total <= 0:
            return None
        threshold = total * percentile
        cumulative = 0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= threshold:
                return value
        return ordered[-1][0]

    def allocate(
        self,
        demand_records: list[dict[str, Any]],
        allocations: list[dict[str, Any]],
        vulnerability_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Assign exposed population to reachable facilities.

        Demand and capacity are integerized to people because NetworkX's
        network simplex implementation requires integral flows.  Fractional
        areal-exposure estimates are rounded once and the rounded total is
        disclosed in the result.
        """
        if not math.isfinite(vulnerability_weight) or vulnerability_weight < 0:
            raise ValueError("vulnerability_weight must be finite and non-negative")

        cells: list[dict[str, Any]] = []
        for index, record in enumerate(demand_records or []):
            exposed = _safe_float(record.get("exposed_population"))
            centroid = record.get("centroid")
            svi = _safe_float(record.get("svi"))
            if (
                exposed is None
                or exposed <= 0
                or not isinstance(centroid, (list, tuple))
                or len(centroid) < 2
                or _safe_float(centroid[0]) is None
                or _safe_float(centroid[1]) is None
                or svi is None
            ):
                continue
            people = max(1, int(round(exposed)))
            cells.append(
                {
                    "node": f"demand:{index}",
                    "tract_id": record.get("tract_id") or f"demand_{index}",
                    "people": people,
                    "exposed_population": float(exposed),
                    "flooded_fraction": _safe_float(record.get("flooded_fraction")) or 0.0,
                    "svi": max(0.0, min(1.0, float(svi))),
                    "lat": float(centroid[0]),
                    "lon": float(centroid[1]),
                }
            )

        facilities: list[dict[str, Any]] = []
        excluded_types: set[str] = set()
        for index, allocation in enumerate(allocations or []):
            facility_type = normalize_facility_type(
                allocation.get("type") or allocation.get("facility_type")
            )
            if facility_type not in self.eligible_facility_types:
                excluded_types.add(facility_type or "unspecified")
                continue
            capacity = _safe_float(allocation.get("capacity"))
            lat = _safe_float(allocation.get("lat"))
            lon = _safe_float(allocation.get("lon"))
            if capacity is None or capacity <= 0 or lat is None or lon is None:
                continue
            facilities.append(
                {
                    "node": f"facility:{index}",
                    "facility_id": str(
                        allocation.get("resource")
                        or allocation.get("facility_id")
                        or f"facility_{index}"
                    ),
                    "capacity": max(1, int(math.floor(capacity))),
                    "capacity_assumed": bool(allocation.get("capacity_assumed", False)),
                    "facility_type": facility_type,
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )

        if not cells:
            return {"status": "unavailable", "reason": "no_valid_demand_records"}
        if not facilities:
            total_demand = sum(cell["people"] for cell in cells)
            return {
                "status": "unavailable",
                "reason": "no_eligible_facilities",
                "eligible_facility_types": list(self.eligible_facility_types),
                "excluded_facility_types": sorted(excluded_types),
                "total_demand_people": total_demand,
                "served_people": 0,
                "unmet_people": total_demand,
                "coverage": 0.0,
                "unmet_fraction": 1.0,
                "vulnerability_coverage": 0.0,
                "vulnerability_weighted_unmet_need": round(
                    sum(
                        cell["people"]
                        * (1.0 + vulnerability_weight * cell["svi"])
                        for cell in cells
                    ),
                    2,
                ),
                "demand_impacts": [
                    {
                        "tract_id": cell["tract_id"],
                        "exposed_population": cell["people"],
                        "hazard_exposure": cell["flooded_fraction"],
                        "coverage": 0.0,
                        "assigned_people": 0,
                        "unmet_people": cell["people"],
                        "svi": cell["svi"],
                    }
                    for cell in cells
                ],
            }

        total_demand = sum(cell["people"] for cell in cells)
        graph = nx.DiGraph()
        graph.add_node("source", demand=-total_demand)
        graph.add_node("sink", demand=total_demand)

        distance_lookup: dict[tuple[str, str], float] = {}

        for cell in cells:
            graph.add_node(cell["node"], demand=0)
            graph.add_edge(
                "source", cell["node"], capacity=cell["people"], weight=0
            )

        for facility in facilities:
            graph.add_node(facility["node"], demand=0)
            graph.add_edge(
                facility["node"],
                "sink",
                capacity=facility["capacity"],
                weight=0,
            )

        for cell in cells:
            for facility in facilities:
                distance = geodesic_km(
                    cell["lat"], cell["lon"], facility["lat"], facility["lon"]
                )
                if (
                    self.max_travel_km is not None
                    and distance > self.max_travel_km
                ):
                    continue
                distance_lookup[(cell["node"], facility["node"])] = distance
                graph.add_edge(
                    cell["node"],
                    facility["node"],
                    capacity=cell["people"],
                    weight=max(1, int(round(distance * 1000))),
                )

        if not distance_lookup:
            return {
                "status": "unavailable",
                "reason": "no_reachable_facilities",
                "total_demand_people": total_demand,
                "served_people": 0,
                "unmet_people": total_demand,
                "coverage": 0.0,
                "unmet_fraction": 1.0,
                "vulnerability_coverage": 0.0,
                "vulnerability_weighted_unmet_need": round(
                    sum(
                        cell["people"]
                        * (1.0 + vulnerability_weight * cell["svi"])
                        for cell in cells
                    ),
                    2,
                ),
                "max_travel_km": self.max_travel_km,
                "demand_impacts": [
                    {
                        "tract_id": cell["tract_id"],
                        "exposed_population": cell["people"],
                        "hazard_exposure": cell["flooded_fraction"],
                        "coverage": 0.0,
                        "assigned_people": 0,
                        "unmet_people": cell["people"],
                        "svi": cell["svi"],
                    }
                    for cell in cells
                ],
            }

        # Unserved demand must cost more than assignment to any discovered
        # candidate; otherwise a distant but valid facility can lose to the
        # synthetic "leave unserved" edge.  When max_travel_km is omitted,
        # resource discovery itself defines the candidate scope and no second,
        # arbitrary service-radius cutoff is applied here.
        farthest_candidate_km = max(distance_lookup.values())
        penalty_reference_km = max(
            farthest_candidate_km,
            self.max_travel_km or 0.0,
        )
        base_unserved_penalty = int(
            round((penalty_reference_km + 1.0) * 1000)
        )
        for cell in cells:
            unserved_penalty = int(
                round(
                    base_unserved_penalty
                    * (1.0 + vulnerability_weight * cell["svi"])
                )
            )
            graph.add_edge(
                cell["node"],
                "sink",
                capacity=cell["people"],
                weight=unserved_penalty,
            )

        _, flow = nx.network_simplex(graph)

        assignments: list[dict[str, Any]] = []
        demand_impacts: list[dict[str, Any]] = []
        facility_served = {facility["node"]: 0 for facility in facilities}
        travel_distances: list[tuple[float, int]] = []
        total_served = 0
        weighted_total = 0.0
        weighted_served = 0.0
        weighted_unmet = 0.0

        for cell in cells:
            served = 0
            for facility in facilities:
                amount = int(flow.get(cell["node"], {}).get(facility["node"], 0))
                if amount <= 0:
                    continue
                distance = distance_lookup[(cell["node"], facility["node"])]
                served += amount
                total_served += amount
                facility_served[facility["node"]] += amount
                travel_distances.append((distance, amount))
                assignments.append(
                    {
                        "tract_id": cell["tract_id"],
                        "facility_id": facility["facility_id"],
                        "assigned_people": amount,
                        "distance_km": round(distance, 3),
                    }
                )

            unmet = max(0, cell["people"] - served)
            coverage = served / cell["people"] if cell["people"] else 0.0
            weighted_mass = cell["people"] * cell["svi"]
            weighted_total += weighted_mass
            weighted_served += weighted_mass * coverage
            weighted_unmet += unmet * (1.0 + vulnerability_weight * cell["svi"])
            demand_impacts.append(
                {
                    "tract_id": cell["tract_id"],
                    "exposed_population": cell["people"],
                    "hazard_exposure": cell["flooded_fraction"],
                    "coverage": round(coverage, 6),
                    "assigned_people": served,
                    "unmet_people": unmet,
                    "svi": cell["svi"],
                }
            )

        p95_distance = self._weighted_percentile(travel_distances, 0.95)
        time_range = None
        if p95_distance is not None and self.planning_speed_range_kmh is not None:
            low_speed, high_speed = self.planning_speed_range_kmh
            time_range = {
                "min": round(p95_distance / high_speed * 60.0, 2),
                "max": round(p95_distance / low_speed * 60.0, 2),
            }
        facility_utilization = []
        for facility in facilities:
            served = facility_served[facility["node"]]
            facility_utilization.append(
                {
                    "facility_id": facility["facility_id"],
                    "capacity": facility["capacity"],
                    "assigned_people": served,
                    "utilization": round(served / facility["capacity"], 6),
                    "capacity_assumed": facility["capacity_assumed"],
                }
            )

        unmet = total_demand - total_served
        vulnerability_coverage = (
            weighted_served / weighted_total
            if weighted_total > 0
            else total_served / total_demand
        )
        return {
            "status": "allocated",
            "method": "capacity_constrained_min_cost_flow_geodesic_proxy",
            "total_demand_people": total_demand,
            "served_people": total_served,
            "unmet_people": unmet,
            "coverage": round(total_served / total_demand, 6),
            "unmet_fraction": round(unmet / total_demand, 6),
            "vulnerability_coverage": round(vulnerability_coverage, 6),
            "vulnerability_weighted_unmet_need": round(weighted_unmet, 2),
            "travel_distance_p95_km": (
                round(p95_distance, 3) if p95_distance is not None else None
            ),
            "transfer_time_estimate_range_minutes": time_range,
            "time_estimate_basis": (
                "p95 straight-line assignment distance divided by the configured "
                "planning speed range; not a live-traffic or clearance-time estimate"
                if time_range is not None
                else "not_estimated"
            ),
            "planning_gaps": (
                [] if time_range is not None else ["transfer_time_estimate"]
            ),
            "max_travel_km": self.max_travel_km,
            "planning_speed_range_kmh": (
                list(self.planning_speed_range_kmh)
                if self.planning_speed_range_kmh is not None
                else None
            ),
            "capacity_assumed_facility_count": sum(
                1 for facility in facilities if facility["capacity_assumed"]
            ),
            "eligible_facility_types": list(self.eligible_facility_types),
            "excluded_facility_types": sorted(excluded_types),
            "excluded_facility_count": sum(
                1
                for item in (allocations or [])
                if normalize_facility_type(
                    item.get("type") or item.get("facility_type")
                ) not in self.eligible_facility_types
            ),
            "assignments": assignments,
            "facility_utilization": facility_utilization,
            "demand_impacts": demand_impacts,
        }
