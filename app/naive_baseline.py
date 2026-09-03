"""NaiveBaseline — a deliberately naive catch-all pipeline for baseline
comparison (paper §4.3).

Consumes the SAME offline fixtures (app.demo_fixtures.FakeMCP) as the
reference pipeline, so both systems see identical inputs; only the
pipeline semantics differ. Naive conventions (each is the optimistic
counterpart of one honesty mechanism in the reference pipeline):

    1.  Missing inputs are substituted silently with zeros (no
        data-gap ledger, no degraded labels, no uncertainty band).
    2.  Population exposure is all-or-nothing: a census tract counts
        in full when its centroid falls inside the flood polygon,
        else not at all (no areal weighting, no method interval).
    3.  The satellite extent is consumed as returned — no city-
        boundary clipping, no re-measurement, no acquisition-time
        gate (a stale or undated scene is accepted silently).
    4.  Station metadata outages are ignored: whatever coordinates
        are available are used without an independent verification,
        and resource allocation is ALWAYS allowed (no evidence gate).
    5.  Route access is assumed perfect, and no source-conflict
        quarantine exists (single-value pipeline).

`silent_issues` is experiment instrumentation, not part of a real
dashboard's output; `disclosed_issues` is empty by construction.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .utils import geodesic_km


def _nz(x: float) -> float:
    """Threshold-segmented lower bound: negatives clamp to 0; linear
    and uncapped above."""
    return max(0.0, x)


def _clamp01(x: float) -> float:
    # Only for naive scoring (bounding response_time / cost); not on
    # the severity path.
    return max(0.0, min(1.0, x))


def _polygon_rings(geojson: dict[str, Any]) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    if not isinstance(geojson, dict):
        return rings
    for feat in geojson.get("features") or []:
        geom = feat.get("geometry") or {}
        for ring in geom.get("coordinates") or []:
            if isinstance(ring, list) and len(ring) >= 3:
                rings.append(ring)
    return rings


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


async def _call_json(mcp: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """One MCP call, parsed; unusable responses become a status-error dict."""
    raw = await mcp.call(tool, args)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"status": "error", "error": "unparseable"}
    return parsed if isinstance(parsed, dict) else {"status": "error", "error": "unparseable"}


class NaiveBaseline:
    """Same inputs as the reference pipeline; catch-all semantics.

    `run` splits into `_observe` (the hand-written sequential collection
    order) and `_score` (the optimistic conventions shared verbatim with
    `app.plan_execute_baseline.PlanExecuteBaseline`, so both baselines
    differ ONLY in how the tool chain is sequenced, never in arithmetic).
    """

    name = "naive_dashboard"

    def __init__(self, coverage_radius_km: float = 2.0):
        self.coverage_radius_km = coverage_radius_km

    async def run(self, mcp: Any, target: str, station_id: str) -> dict[str, Any]:
        return self._score(await self._observe(mcp, target, station_id))

    async def _observe(  # noqa: C901
        self, mcp: Any, target: str, station_id: str
    ) -> dict[str, Any]:
        """Sequential, hard-coded collection (dashboard convention)."""
        silent: list[str] = []

        # -- geocode (no station-based disambiguation check) ----------
        raw = await mcp.call("geocode_location", {"place_name": target})
        m = re.search(r"纬度\s*(-?\d+\.?\d*).*?经度\s*(-?\d+\.?\d*)", str(raw))
        if not m:
            silent.append(f"geocode failed for {target!r} → coords (0, 0)")
            lat, lon = 0.0, 0.0
        else:
            lat, lon = float(m.group(1)), float(m.group(2))

        obs = await _call_json(
            mcp,
            "get_flood_observation",
            {"station_id": station_id, "latitude": lat, "longitude": lon},
        )
        meta = await _call_json(mcp, "get_station_metadata", {"station_id": station_id})
        nwps = await _call_json(mcp, "get_nwps_gauge", {"station_id": station_id})
        gee = await _call_json(
            mcp,
            "get_flood_extent",
            {"latitude": lat, "longitude": lon, "buffer_km": 10,
             "observation_date": None},
        )
        svi = await _call_json(
            mcp,
            "get_social_vulnerability",
            {"latitude": lat, "longitude": lon, "radius_km": 10},
        )
        infra = await _call_json(
            mcp,
            "get_critical_infrastructure",
            {"latitude": lat, "longitude": lon, "radius_km": 10},
        )
        res = await _call_json(
            mcp, "get_available_resources", {"latitude": lat, "longitude": lon}
        )
        return {
            "target": target, "station_id": station_id,
            "lat": lat, "lon": lon, "silent": silent,
            "observation": obs, "metadata": meta, "nwps": nwps,
            "extent": gee, "svi": svi, "infra": infra, "resources": res,
        }

    def _score(self, o: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """Optimistic conventions, shared verbatim with the PE baseline."""
        lat, lon = o["lat"], o["lon"]
        silent = o["silent"]
        obs, meta, nwps = o["observation"], o["metadata"], o["nwps"]
        gee, svi, infra, res = o["extent"], o["svi"], o["infra"], o["resources"]
        if obs.get("status") != "ok":
            silent.append("observation failed → water_level = 0")
            water_level = 0.0
        else:
            water_level = float(obs["observation"]["water_level"])

        # -- station metadata (no verification, no distance gate) ----
        if meta.get("status") != "ok":
            silent.append(
                "station metadata failed → observation coords used, "
                "verification skipped"
            )
            s_lat, s_lon = obs.get("observation", {}).get("latitude"), obs.get(
                "observation", {}
            ).get("longitude")
            if s_lat is None:
                s_lat, s_lon = 0.0, 0.0
        else:
            s_lat = meta["metadata"]["latitude"]
            s_lon = meta["metadata"]["longitude"]

        # -- action stage ---------------------------------------------
        if nwps.get("status") != "ok":
            silent.append("no NWPS flood categories → water_severity = 0")
            action_stage = None
            major_stage = None
        else:
            cats = nwps["flood_categories"]
            action_stage = cats["action"]["stage"]
            major_stage = (cats.get("major") or {}).get("stage")

        # Same severity convention as the reference engine (major-
        # referenced), but missing inputs fold silently to 0 with no
        # ledger.
        if action_stage and action_stage > 0:
            if major_stage and major_stage > action_stage:
                _x = _nz(
                    (water_level - action_stage) / (major_stage - action_stage)
                )
                water_severity = _x / (1.0 + _x)
            else:
                _x = _nz(water_level / action_stage)
                water_severity = _x / (1.0 + _x)
        else:
            water_severity = 0.0

        # -- satellite extent (as returned: no clip, no re-measure,
        #    no acquisition-time gate) ---------------------------------
        flood_geojson = None
        if gee.get("status") == "ok":
            spatial = gee.get("spatial_extent") or {}
            flooded_area_km2 = spatial.get("flooded_area_km2")
            flood_geojson = spatial.get("geojson")
            if flooded_area_km2 is None:
                silent.append("no flood extent → flooded_area_km2 = 0")
                flooded_area_km2 = 0.0
            else:
                acq = (gee.get("observation") or {}).get("latest_post_scene")
                if not acq:
                    silent.append("extent has no acquisition time → accepted")
        else:
            silent.append("extent source failed → flooded_area_km2 = 0")
            flooded_area_km2 = 0.0

        analysis_area_km2 = math.pi * 10.0 ** 2  # naive: circular buffer
        extent_ratio = flooded_area_km2 / analysis_area_km2
        # Mirror the reference engine: hazard = water dimension (area
        # not in the index).
        hazard = water_severity

        # -- exposure: centroid-in-polygon, all-or-nothing ------------
        rings = _polygon_rings(flood_geojson or {})
        tracts = svi.get("tracts") or [] if svi.get("status") == "ok" else []
        total_pop = 0.0
        exposed_pop = 0.0
        per_tract: list[dict[str, Any]] = []
        for tract in tracts:
            pop = float(tract.get("population") or 0)
            rings_tract = (tract.get("geometry") or {}).get("rings") or []
            total_pop += pop
            centroid = _ring_centroid(rings_tract[0]) if rings_tract else None
            hit = bool(
                centroid
                and any(_point_in_ring(centroid[0], centroid[1], r) for r in rings)
            )
            if hit:
                exposed_pop += pop
            per_tract.append(
                {"tract_id": tract.get("tract_id"), "population": pop,
                 "svi": float(tract.get("svi") or 0), "exposed": pop if hit else 0.0,
                 "_centroid": centroid}
            )
        pop_factor = _nz(exposed_pop / total_pop) if total_pop else 0.0

        facility_count = (
            infra["measurements"]["facility_count"]["value"]
            if infra.get("status") == "ok"
            else (silent.append("facilities failed → 0") or 0)
        )
        facility_factor = _nz(facility_count / 100.0)
        facility_severity = _clamp01(facility_factor)
        pop_severity = pop_factor  # physical share (naive denominator unfiltered)
        exposure = max(pop_severity, facility_severity)

        # Same 3-factor structure as the reference engine; response
        # capacity (roads/routes) does not enter the index.
        svi_weighted = (
            sum(t["svi"] * t["population"] for t in per_tract) / total_pop
            if total_pop
            else 0.0
        )
        cdri = hazard * (0.5 + 0.5 * exposure) * svi_weighted
        eps = (
            water_severity
            * ((1.0 + svi_weighted) / 2.0)
            * ((1.0 + pop_severity) / 2.0)
        )

        def _label(value: float) -> str:
            if value < 0.01:
                return "Low"
            if value < 0.05:
                return "Moderate"
            if value < 0.15:
                return "High"
            return "Very High"

        # -- allocation: ALWAYS allowed (no evidence gate) -------------
        plans = res.get("resources") or [] if res.get("status") == "ok" else []
        best_plan, best_score = None, -1.0
        for plan in plans:
            obj = plan.get("objectives") or {}
            values = {
                "risk_reduction": float(obj.get("risk_reduction") or 0),
                "coverage": float(obj.get("coverage") or 0),
                "response_time": float(obj.get("response_time") or 0),
                "cost": float(obj.get("cost") or 0),
                "unmet_demand": float(obj.get("unmet_demand") or 0),
            }
            score = (
                values["risk_reduction"] + values["coverage"]
                + (1.0 - _clamp01(values["response_time"] / 60.0))
                + (1.0 - _clamp01(values["cost"] / 200.0))
                + (1.0 - values["unmet_demand"])
            ) / 5.0
            if score > best_score:
                best_score, best_plan = score, plan
        recommended = (best_plan or {}).get("plan_id")

        # naive VWUN: all-or-nothing exposure, λ = 1, binary coverage
        vwun = 0.0
        if best_plan:
            points = [
                (a["lat"], a["lon"])
                for a in best_plan.get("allocations") or []
            ]
            for t in per_tract:
                covered = False
                tr = t.get("_centroid")
                if tr is not None:
                    for f_lat, f_lon in points:
                        if geodesic_km(tr[1], tr[0], f_lat, f_lon) <= (
                            self.coverage_radius_km
                        ):
                            covered = True
                            break
                vwun += t["exposed"] * (0 if covered else 1) * (
                    1.0 + t["svi"]
                )

        return {
            "status": "completed",
            "cdri_percent": round(cdri * 100.0, 2),
            "cdri_label": _label(cdri),
            "band": None,
            "eps": round(eps, 3),
            "data_confidence": None,
            "exposed_population": exposed_pop,
            "total_population": total_pop,
            "recommended_plan": recommended,
            "allocation_emitted": best_plan is not None,
            "vwun_recommended": round(vwun, 2),
            "disclosed_issues": [],
            "silent_issues": silent,
            "station_distance_km": (
                round(geodesic_km(lat, lon, s_lat, s_lon), 1)
                if s_lat is not None
                else None
            ),
        }
