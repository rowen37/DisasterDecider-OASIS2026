"""Offline demo fixtures — FakeMCP and demo-run entry.

Reusable offline stand-in shared by the Web UI Demo mode
(/api/assess {"demo": true}) and regression tests, so the full
pipeline output (including the lambda slider / equity ledger)
reproduces on any date with no external network dependency.

The GEE fixture is date-aware: given a past observation_date, the
satellite acquisition time lands on that date + 1 day, so historical
replay stays time-aligned (in real runs GEE returns what it returns
and stale scenes are rejected by the freshness gate).
"""

from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any

from .utils import geodesic_km

# Hermetic "now": fixture timestamps are generated relative to the real
# current time so freshness / data_confidence do not drift with the
# test run date (reproducibility).
NOW = datetime.now(timezone.utc).isoformat()
TOMORROW = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

DEMO_SCENARIOS = {
    "nominal",
    "metadata_503",
    "no_sar",
    "stale_sar",
    "no_nwps",
    "far_station",
    "fusion_conflict",
}


def _ring(lon, lat, d=0.02):
    return [[lon - d, lat - d], [lon + d, lat - d],
            [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]


def _polygon_geojson(lon, lat, d=0.02):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"flooded": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_ring(lon, lat, d)],
                },
            }
        ],
    }


def _write_fixture_file(content: str, suffix: str) -> str | None:
    """Write fixture GIS artifacts to real files on disk (unique temp
    names prevent concurrent clobbering).

    The Web layer copies returned output_path/map_path into
    static/maps and silently drops products whose file does not
    exist, so artifacts must be real. Returns None on failure (the
    caller then receives a nonexistent path).
    """
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(content)
        return handle.name
    except OSError:
        return None


_DEMO_MAP_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Demo flood map</title>
<style>body{font-family:sans-serif;margin:24px;color:#1a3c5e;}</style>
</head><body>
<h2>Demo situational map (offline fixtures)</h2>
<p>Flood extent, affected buildings and the rescue route are rendered
as GeoJSON layers on the main Leaflet map; this artifact stands in for
the GIS visualization product in demo mode.</p>
</body></html>
"""


# ── Shortest path on a road network (demo rescue route) ──────────
# Prefers the real OSM road cache (built by scripts/build_demo_road_cache.py
# with osmnx): endpoints snap to the nearest nodes, flooded edges are removed,
# and Dijkstra follows the cached directed street graph. A deterministic
# synthetic grid remains the offline fallback when the cache is unavailable.
_TRAVEL_SPEED_MPS = {
    "drive": 30.0 / 3.6,
    "walk": 4.0 / 3.6,
    "bike": 12.0 / 3.6,
}

_DEMO_ROAD_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "demo_road_network.json",
)
_demo_road_cache = None
_demo_road_cache_loaded = False

def _demo_jitter(i: int, j: int, scale: float) -> float:
    """Deterministic node jitter (integer-hash modulo): street-like placement, reproducible."""
    h = ((i + 8) * 73856093) ^ ((j + 16) * 19349663)
    return (((h % 2001) / 1000.0) - 1.0) * scale


def _demo_street_grid(origin, dest, dlat=0.006, dlon=0.008):
    """Deterministic grid covering the origin/destination corridor."""
    lat0 = min(origin[0], dest[0]) - 2 * dlat
    lon0 = min(origin[1], dest[1]) - 2 * dlon
    ni = int(math.ceil((max(origin[0], dest[0]) + 2 * dlat - lat0) / dlat)) + 1
    nj = int(math.ceil((max(origin[1], dest[1]) + 2 * dlon - lon0) / dlon)) + 1
    nodes = {}
    for i in range(ni):
        for j in range(nj):
            nodes[(i, j)] = (
                lat0 + i * dlat + _demo_jitter(i, j, 0.12 * dlat),
                lon0 + j * dlon + _demo_jitter(j, i, 0.12 * dlon),
            )
    return nodes


def _load_demo_road_cache():
    """Load the generated OSM street cache or return ``None``."""

    global _demo_road_cache, _demo_road_cache_loaded
    if _demo_road_cache_loaded:
        return _demo_road_cache
    _demo_road_cache_loaded = True
    try:
        with open(_DEMO_ROAD_CACHE_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        nodes = {
            str(node_id): (float(value[0]), float(value[1]))
            for node_id, value in payload["nodes"].items()
        }
        adjacency = {}
        for edge in payload["edges"]:
            source, destination = str(edge[0]), str(edge[1])
            length_m = float(edge[2])
            geometry = edge[3] if len(edge) >= 4 else None
            adjacency.setdefault(source, []).append(
                (destination, length_m, geometry)
            )
        _demo_road_cache = (nodes, adjacency)
    except (OSError, ValueError, KeyError, TypeError):
        _demo_road_cache = None
    return _demo_road_cache


def _point_in_ring(lon, lat, ring):
    """Ray casting: is the point inside the polygon's outer ring (demo flood polygons have no holes)?"""
    inside = False
    n = len(ring)
    for k in range(n):
        lat1, lon1 = ring[k][1], ring[k][0]
        lat2, lon2 = ring[(k + 1) % n][1], ring[(k + 1) % n][0]
        if (lat1 > lat) != (lat2 > lat):
            lon_at = (lon2 - lon1) * (lat - lat1) / (lat2 - lat1) + lon1
            if lon < lon_at:
                inside = not inside
    return inside


def _demo_flood_predicate(avoid_path):
    """Build an edge-blocking predicate blocked(coords).

    Prefers shapely edge-polyline vs flood-polygon intersection tests
    (same as the real tool); falls back to a midpoint ray-casting test
    when shapely is unavailable.
    """
    rings = []
    features = []
    if avoid_path and os.path.exists(avoid_path):
        try:
            with open(avoid_path, encoding="utf-8") as fh:
                gj = json.load(fh)
            for feat in gj.get("features", []):
                geom = feat.get("geometry") or {}
                if geom.get("type") == "Polygon":
                    rings.append(geom["coordinates"][0])
                    features.append(geom)
                elif geom.get("type") == "MultiPolygon":
                    for poly in geom["coordinates"]:
                        rings.append(poly[0])
                    features.append(geom)
        except (OSError, ValueError, KeyError, TypeError):
            rings = []
            features = []
    if not rings:
        return None
    try:
        from shapely.geometry import LineString, shape
        shapes = [shape(g) for g in features]

        def blocked(coords):
            line = LineString([(p[0], p[1]) for p in coords])
            return any(s.intersects(line) for s in shapes)

        return blocked
    except ImportError:
        def blocked(coords):
            mid = coords[len(coords) // 2]
            return any(_point_in_ring(mid[0], mid[1], ring) for ring in rings)

        return blocked


def _demo_dijkstra(adj, source, goal):
    """Dijkstra on the stdlib heapq; adj = {u: [(v, weight_m, geom), ...]}.

    Returns (total_m, legs) with legs = [(u, v, geom), ...] the chosen
    edges; None if goal is unreachable.
    """
    best = {source: 0.0}
    prev = {}
    done = set()
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == goal:
            break
        for v, w, geom in adj.get(u, ()):
            nd = d + w
            if nd < best.get(v, float("inf")) - 1e-9:
                best[v] = nd
                prev[v] = (u, geom)
                heapq.heappush(heap, (nd, v))
    if goal not in best:
        return None
    legs = []
    cur = goal
    while cur != source:
        u, geom = prev[cur]
        legs.append((u, cur, geom))
        cur = u
    legs.reverse()
    return best[goal], legs


class FakeMCP:
    """In-memory MCP stand-in that returns offline fixtures by tool name."""

    def __init__(self, scenario: str = "nominal"):
        if scenario not in DEMO_SCENARIOS:
            raise ValueError(
                f"Unknown demo scenario {scenario!r}; expected one of: "
                + ", ".join(sorted(DEMO_SCENARIOS))
            )
        self.scenario = scenario
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool_name: str, arguments: dict, **_policy) -> str:
        # **_policy absorbs per-call timeout/retry overrides
        # (timeout/max_retries); FakeMCP is in-memory and does not
        # simulate retry policy.
        self.calls.append((tool_name, arguments))
        handler = getattr(self, f"_fx_{tool_name}", None)
        if handler is None:
            return json.dumps(
                {"status": "error", "error": f"no fixture for {tool_name}"}
            )
        payload = handler(arguments)
        import inspect
        if inspect.isawaitable(payload):
            payload = await payload
        return payload if isinstance(payload, str) else json.dumps(payload)

    def print_catalog(self):
        pass

    # -- Geocoding -----------------------------------------------------
    def _fx_geocode_location(self, args):
        if self.scenario == "far_station":
            return (
                f"{args['place_name']} 的坐标: 纬度 47.6062, "
                "经度 -122.3321"
            )
        return (
            f"{args['place_name']} 的坐标: 纬度 29.5294, 经度 -95.2010"
        )

    def _fx_geocode_boundary(self, args):
        return {
            "status": "ok",
            "name": args["place_name"],
            "osm_type": "relation",
            "bbox": [-95.4, 29.4, -95.0, 29.7],
            "geojson": _polygon_geojson(-95.10, 29.50, d=0.15),
        }

    # -- USGS / NWPS chain ---------------------------------------------
    def _fx_get_station_metadata(self, args):
        if self.scenario == "metadata_503":
            return {
                "status": "error",
                "error": "HTTP 503 Service Unavailable (offline fault demo)",
            }
        return {
            "status": "ok",
            "metadata": {
                "station_id": "08077600",
                "station_name": "Clear Ck nr Friendswood, TX",
                "latitude": 29.5175,
                "longitude": -95.1785,
                "state": "TX",
                "metadata_verified": True,
            },
        }

    def _fx_get_flood_observation(self, args):
        # Historical replay: observation time lands inside the
        # requested window (time alignment is verified by the pipeline).
        # Windowed queries return the window PEAK as the primary value
        # plus a full hydrograph summary (same contract as the real
        # USGS IV MCP since 2026-09-26).
        obs_time = NOW
        window_summary = None
        semantics = "latest_instantaneous"
        if args.get("start_dt"):
            try:
                _start = datetime.fromisoformat(
                    str(args["start_dt"]).replace("Z", "+00:00")
                )
                _end = datetime.fromisoformat(
                    str(args["end_dt"]).replace("Z", "+00:00")
                )
                obs_time = (_start + timedelta(hours=6)).isoformat()
                semantics = "window_peak"
                window_summary = {
                    "start": args["start_dt"],
                    "end": args["end_dt"],
                    "value_count": 97,
                    "peak_stage_ft": 8.0,
                    "peak_time": obs_time,
                    "end_stage_ft": 6.2,
                    "end_time": (_end - timedelta(hours=1)).isoformat(),
                    "min_stage_ft": 2.1,
                }
            except ValueError:
                pass
        water_level = 20.0 if args.get("demo_conflict") else 8.0
        if window_summary is not None and args.get("demo_conflict"):
            window_summary = dict(window_summary)
            window_summary["peak_stage_ft"] = water_level
        return {
            "status": "ok",
            "query_window": (
                {"start": args.get("start_dt"), "end": args.get("end_dt")}
                if args.get("start_dt")
                else None
            ),
            "observation": {
                "station_id": "08077600",
                "water_level": water_level,
                "unit": "ft",
                "observation_time": obs_time,
                "observation_semantics": semantics,
                "window_summary": window_summary,
                "source": "USGS",
                "station_name": "Clear Ck nr Friendswood, TX",
                "latitude": 29.5175,
                "longitude": -95.1785,
                "metadata_verified": True,
                "location_verified": True,
            },
        }

    def _fx_get_waterway_network(self, args):
        # Two waterways through the fixture city square
        # (-95.25..-94.95, 29.35..29.65): a diagonal main stem and a
        # north-south tributary crossing the flood polygon.
        return {
            "status": "ok",
            "source": "OSM",
            "source_type": "waterway_network",
            "timestamp": NOW,
            "location": {
                "latitude": args.get("latitude"),
                "longitude": args.get("longitude"),
            },
            "waterways": [
                {
                    "osm_id": 1001,
                    "name": "Clear Creek",
                    "waterway": "river",
                    "coordinates": [
                        [-95.30, 29.40], [-95.20, 29.45],
                        [-95.10, 29.50], [-95.00, 29.55],
                        [-94.90, 29.60],
                    ],
                },
                {
                    "osm_id": 1002,
                    "name": "Mary's Creek",
                    "waterway": "stream",
                    "coordinates": [
                        [-95.13, 29.65], [-95.12, 29.55],
                        [-95.11, 29.45], [-95.10, 29.35],
                    ],
                },
            ],
            "waterway_count": 2,
            "radius_km": args.get("radius_km", 15),
            "metadata_verified": True,
        }

    def _fx_get_nwps_gauge(self, args):
        if self.scenario == "no_nwps":
            return {
                "status": "error",
                "error": "NWPS could not find a matching gauge (offline fault demo)",
            }
        return {
            "status": "ok",
            "gauge": {
                "lid": "FRIT2",
                "usgs_id": "08077600",
                "name": "Clear Creek at Friendswood",
                "latitude": 29.5175,
                "longitude": -95.1785,
            },
            "flood_categories": {
                "action": {"stage": 7.0},
                "minor": {"stage": 9.0},
                "moderate": {"stage": 12.0},
                "major": {"stage": 15.0},
            },
            "threshold_verified": True,
            "threshold_source": "NWPS flood_categories (per-gauge authoritative)",
            "metadata_verified": True,
        }

    def _fx_get_nwps_stageflow(self, args):
        return {
            "status": "ok",
            "observed": [{"stage_ft": 8.0, "time": NOW}],
            "forecast": [
                {"stage_ft": 9.4, "time": TOMORROW},
                {"stage_ft": 9.8, "time": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()},
            ],
            "stageflow_verified": True,
            "metadata_verified": True,
        }

    # -- Fusion sources ------------------------------------------------
    def _fx_get_weather_observations(self, args):
        return {
            "status": "ok",
            "source": "NWS",
            "measurements": {
                "temperature": {"value": 24.0, "unit": "C"},
                "humidity": {"value": 82.0, "unit": "%"},
            },
            "timestamp": NOW,
        }

    def _fx_get_precipitation(self, args):
        return {
            "status": "ok",
            "source": "NWS",
            "measurements": {
                "precipitation_6h": {"value": 12.5, "unit": "mm"},
            },
            "timestamp": NOW,
        }

    def _fx_get_forecast(self, args):
        return {
            "status": "ok",
            "source": "NWS",
            "forecasts": [
                {
                    "period": "Tonight",
                    "temperature": 76,
                    "probability_of_precipitation": 60,
                    "timestamp": TOMORROW,
                }
            ],
        }

    def _fx_get_flood_warnings(self, args):
        return {
            "status": "ok",
            "source": "NWS",
            "alerts": [
                {
                    "event": "Flood Warning",
                    "severity": "Moderate",
                    "headline": "Flood Warning issued for Clear Creek",
                    "area": "Harris County",
                    "effective": NOW,
                }
            ],
        }

    def _fx_get_population_exposure(self, args):
        # Matches the real mcp_servers/population_exposure.py response
        # contract (population.total, not a measurements wrapper) so
        # fixtures take the same normalization path as production
        # (fusion Case 7).
        return {
            "status": "ok",
            "source": "CENSUS",
            "source_type": "population_context",
            "population_role": "containing_tract_population",
            "population": {"total": 8500, "unit": "people"},
            "note": (
                "population returned for the census tract containing "
                "the point"
            ),
            "timestamp": NOW,
        }

    def _fx_get_flood_extent(self, args):
        # Date-aware: a past observation_date puts the satellite
        # acquisition on that date + 1 day (historical replay stays
        # time-aligned); default/today uses 12 hours ago so the
        # real-time gate [-30, +1] always passes.
        if self.scenario == "no_sar":
            return {
                "status": "error",
                "error_code": "NO_OBSERVATION_IMAGERY",
                "error": "No Sentinel-1 SAR scenes in the event window (offline fault demo)",
            }

        requested = args.get("observation_date")
        try:
            _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if requested and str(requested) < _today:
                _acq = (
                    datetime.fromisoformat(f"{requested}T23:59:00+00:00")
                    + timedelta(days=1)
                ).isoformat()
            else:
                _acq = (
                    datetime.now(timezone.utc) - timedelta(hours=12)
                ).isoformat()
        except (TypeError, ValueError):
            _acq = (
                datetime.now(timezone.utc) - timedelta(hours=12)
            ).isoformat()
        if self.scenario == "stale_sar":
            # The scenario forces a historical event date in
            # run_demo_assessment; today's acquisition must therefore
            # fail the event-window freshness gate.
            _acq = datetime.now(timezone.utc).isoformat()
        return {
            "status": "ok",
            "source": "Google Earth Engine / Sentinel-1 SAR",
            "observation": {
                "requested_observation_date": requested,
                "window_adjusted": False,
                "latest_post_scene": _acq,
                "pre_scene_count": 3,
                "post_scene_count": 2,
                "pre_acquisition_times": [
                    (datetime.now(timezone.utc) - timedelta(days=4)).isoformat()
                ],
                "post_acquisition_times": [_acq],
            },
            "spatial_extent": {
                "flooded_area_km2": 18.6,
                "geojson": _polygon_geojson(-95.115, 29.51, d=0.03),
            },
            "analysis": {
                "satellite": "Sentinel-1",
                "polarization": "VV",
                "orbit_pass": None,
                "geometry_scale_m": 300,
                "threshold_method": "fixed",
                "threshold_db": -3.0,
            },
            "data_quality": {
                "pre_scene_count": 3,
                "post_scene_count": 2,
                "geometry_returned": True,
                "area_reduction_available": True,
            },
        }

    def _fx_get_critical_infrastructure(self, args):
        facilities = [
            {
                "id": i,
                "type": "school",
                "latitude": 29.50 if i < 6 else 29.58,
                "longitude": -95.11 if i < 6 else -95.18,
            }
            for i in range(12)
        ]
        return {
            "status": "ok",
            "source": "OSM",
            "source_type": "infrastructure",
            "measurements": {
                "facility_count": {"value": 12, "unit": "facilities"},
            },
            "metadata": {"facilities": facilities},
            "timestamp": NOW,
        }

    def _fx_get_road_status(self, args):
        return {
            "status": "ok",
            "source": "OSM",
            "measurements": {
                "road_count": {"value": 42, "unit": "segments"},
            },
            "timestamp": NOW,
        }

    # -- Social vulnerability (SVI tracts with geometry) ---------------
    def _fx_get_social_vulnerability(self, args):
        return {
            "status": "ok",
            "source": "CDC/ATSDR SVI 2022",
            "tract_count": 2,
            "tracts": [
                {
                    "tract_id": "48201450100",
                    "population": 4000,
                    "svi": 0.92,
                    "geometry": {"rings": [_ring(-95.10, 29.50, d=0.02)]},
                },
                {
                    "tract_id": "48201450200",
                    "population": 6000,
                    "svi": 0.30,
                    "geometry": {"rings": [_ring(-95.14, 29.47, d=0.02)]},
                },
            ],
        }

    # -- Resource discovery (six candidates: five non-dominated + one
    #    dominated) ---------------------------------------------------
    # Objective units/semantics match resource_manager: unmet_demand
    # is a fraction [0,1], coverage is a capacity/demand ratio,
    # response_time is minutes.
    # Coverage design (R = 2 km): east/mixed/top2 cover the high-SVI
    # tract (tract1) and top2 also covers tract2; shelter/fire station
    # cover nothing — yielding a step-shaped efficiency-equity frontier
    # (cost vs VWUN): 10→7938.57, 50→7938.57, 60→1218.57,
    # 115→1218.57, 200→0.
    # (VWUN = Σ P_exposed·(1−C)·(1+λ·SVI); P_exposed already includes
    # flooded-fraction weighting — see social_good.py.)
    def _fx_get_available_resources(self, args):
        payload = {
            "status": "ok",
            "resources": [
                {
                    "plan_id": "plan_shelter_top1",
                    "allocations": [{
                        "resource": "S1", "lat": 29.46, "lon": -95.11,
                        "capacity": 100, "type": "shelter",
                        "distance_km": 3.4, "capacity_assumed": True,
                    }],
                    "objectives": {
                        "risk_reduction": 0.35, "coverage": 0.30,
                        "response_time": 12.0, "cost": 10.0,
                        "unmet_demand": 0.85,
                    },
                    "metadata": {"strategy": "dispatch nearest shelter"},
                },
                {
                    "plan_id": "plan_fire_top1",
                    "allocations": [{
                        "resource": "F1", "lat": 29.53, "lon": -95.20,
                        "capacity": 2, "type": "fire_station",
                        "distance_km": 6.9, "capacity_assumed": True,
                    }],
                    "objectives": {
                        "risk_reduction": 0.55, "coverage": 0.45,
                        "response_time": 8.0, "cost": 50.0,
                        "unmet_demand": 0.70,
                    },
                    "metadata": {"strategy": "dispatch nearest fire station"},
                },
                {
                    "plan_id": "plan_hospital_east",
                    "allocations": [{
                        "resource": "H1", "lat": 29.505, "lon": -95.101,
                        "capacity": 100, "type": "hospital",
                        "distance_km": 1.0, "capacity_assumed": False,
                    }],
                    "objectives": {
                        "risk_reduction": 0.80, "coverage": 0.75,
                        "response_time": 18.0, "cost": 60.0,
                        "unmet_demand": 0.30,
                    },
                    "metadata": {"assumed": False},
                },
                {
                    "plan_id": "plan_hospital_west",
                    "allocations": [{
                        "resource": "H2", "lat": 29.462, "lon": -95.141,
                        "capacity": 100, "type": "hospital",
                        "distance_km": 1.0, "capacity_assumed": False,
                    }],
                    # Same five objective values as east but only
                    # covers the low-SVI tract: dominated by east,
                    # demonstrating the frontier filter.
                    "objectives": {
                        "risk_reduction": 0.80, "coverage": 0.75,
                        "response_time": 18.0, "cost": 60.0,
                        "unmet_demand": 0.30,
                    },
                    "metadata": {"assumed": False},
                },
                {
                    "plan_id": "plan_mixed_1_each",
                    "allocations": [
                        {"resource": "H1", "lat": 29.505, "lon": -95.101,
                         "capacity": 100, "type": "hospital",
                         "distance_km": 1.0, "capacity_assumed": False},
                        {"resource": "F1", "lat": 29.53, "lon": -95.20,
                         "capacity": 2, "type": "fire_station",
                         "distance_km": 6.9, "capacity_assumed": True},
                        {"resource": "S1", "lat": 29.46, "lon": -95.11,
                         "capacity": 100, "type": "shelter",
                         "distance_km": 3.4, "capacity_assumed": True},
                    ],
                    "objectives": {
                        "risk_reduction": 0.90, "coverage": 0.85,
                        "response_time": 15.0, "cost": 115.0,
                        "unmet_demand": 0.10,
                    },
                    "metadata": {"strategy": "one facility of each kind"},
                },
                {
                    "plan_id": "plan_hospital_top2",
                    "allocations": [
                        {"resource": "H1", "lat": 29.505, "lon": -95.101,
                         "capacity": 100, "type": "hospital",
                         "distance_km": 1.0, "capacity_assumed": False},
                        {"resource": "H2", "lat": 29.462, "lon": -95.141,
                         "capacity": 100, "type": "hospital",
                         "distance_km": 1.0, "capacity_assumed": False},
                    ],
                    "objectives": {
                        "risk_reduction": 0.97, "coverage": 0.98,
                        "response_time": 26.0, "cost": 200.0,
                        "unmet_demand": 0.02,
                    },
                    "metadata": {"strategy": "dispatch 2 nearest hospitals"},
                },
            ],
        }
        # Sourced fixture capabilities make the offline demo exercise the
        # community-matching path without pretending these are live facts.
        fixture_capabilities = {
            "F1": {
                "emergency_rescue": True,
                "high_water_rescue": True,
                "pickup_service": True,
            },
            "S1": {
                "temporary_shelter": True,
                "overnight_shelter": True,
                "family_support": True,
                "wheelchair_accessible": True,
            },
            "H1": {
                "medical_support": True,
                "wheelchair_accessible": True,
                "accessible_transport": True,
            },
            "H2": {
                "medical_support": True,
                "wheelchair_accessible": True,
            },
        }
        for plan in payload["resources"]:
            for allocation in plan.get("allocations", []):
                resource = allocation.get("resource")
                allocation["facility_id"] = f"demo:{resource}"
                allocation["community_capabilities"] = dict(
                    fixture_capabilities.get(resource, {})
                )
                allocation["community_capability_evidence"] = [{
                    "source": "Friendswood offline demo fixture",
                    "verification_status": "synthetic_demo",
                }]
        return payload

    # -- GIS vector tools -----------------------------------------------
    def _fx_poi_search_osm(self, args):
        # Building surveys and facility searches return different layer
        # paths so downstream vec_intersect can distinguish the two
        # query types (flooded buildings vs flooded facilities).
        is_buildings = "building" in str(args.get("amenity_types", ""))
        feature = (
            {
                "type": "Feature",
                "properties": {
                    "name": "Demo building", "building": "yes", "osm_id": "b1",
                },
                "geometry": {"type": "Point", "coordinates": [-95.102, 29.506]},
            }
            if is_buildings
            else {
                "type": "Feature",
                "properties": {
                    "name": "Demo Hospital", "amenity": "hospital", "osm_id": "1",
                },
                "geometry": {"type": "Point", "coordinates": [-95.102, 29.506]},
            }
        )
        fallback = (
            "/tmp/demo_buildings.geojson"
            if is_buildings
            else "/tmp/demo_poi.geojson"
        )
        return {
            "status": "ok",
            "output_path": _write_fixture_file(
                json.dumps({"type": "FeatureCollection",
                            "features": [feature]}),
                ".geojson",
            ) or fallback,
            "feature_list": [
                {"lat": 29.506, "lon": -95.102, "name": "Demo Hospital",
                 "amenity": "hospital", "osm_id": "1"}
            ],
            "feature_count": 1,
        }

    def _fx_vec_buffer(self, args):
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"buffer": "demo"},
                "geometry": {"type": "Polygon",
                             "coordinates": [_ring(-95.115, 29.51, d=0.05)]},
            }],
        }
        return {
            "status": "ok",
            "output_path": _write_fixture_file(
                json.dumps(geojson), ".geojson"
            ) or "/tmp/demo_buffer.geojson",
        }

    def _fx_vec_intersect(self, args):
        # Distinguished by query layer: POI (hospital/shelter) ∩ flood
        # extent vs buildings ∩ flood extent. Real runs count actual
        # OSM-geometry / flood-polygon overlap; the fixture returns
        # distinct plausible constants (1 flooded hospital / 7 flooded
        # buildings).
        layer2 = str(args.get("layer2_path", ""))
        is_poi = "poi" in layer2
        count = 1 if is_poi else 7
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": (
                            f"Demo {'facility' if is_poi else 'building'} {i + 1}"
                        )
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-95.115 + 0.004 * i, 29.51 + 0.003 * i],
                    },
                }
                for i in range(count)
            ],
        }
        fallback = (
            "/tmp/demo_intersect_poi.geojson"
            if is_poi
            else "/tmp/demo_intersect.geojson"
        )
        return {
            "status": "ok",
            "intersected_count": count,
            "output_path": _write_fixture_file(
                json.dumps(geojson), ".geojson"
            ) or fallback,
        }

    def _fx_vec_shortest_path(self, args):
        # Computed over the cached OSM street graph (deterministic grid only
        # when the cache is missing): snap endpoints → optionally remove
        # flooded edges → Dijkstra → polyline GeoJSON.
        # The hospital sits inside the demo flood polygon, so an
        # unreachable flood-avoiding route falls back to the same base
        # graph and discloses route_avoids_flood=False.
        origin = (float(args["origin_lat"]), float(args["origin_lon"]))
        dest = (float(args["dest_lat"]), float(args["dest_lon"]))
        travel_mode = str(args.get("travel_mode", "drive") or "drive")
        speed = _TRAVEL_SPEED_MPS.get(
            travel_mode, _TRAVEL_SPEED_MPS["drive"]
        )

        blocked = _demo_flood_predicate(
            str(args.get("avoid_polygon_path", "") or "")
        )

        road_cache = _load_demo_road_cache()
        using_osm_cache = road_cache is not None
        if using_osm_cache:
            nodes, base_adjacency = road_cache
        else:
            nodes = _demo_street_grid(origin, dest)
            base_adjacency = None

        def snap(p):
            return min(
                nodes,
                key=lambda n: geodesic_km(
                    p[0], p[1], nodes[n][0], nodes[n][1]
                ),
            )

        src, goal = snap(origin), snap(dest)
        adj = {}
        removed = 0
        if using_osm_cache:
            for node, outgoing in base_adjacency.items():
                for neighbor, weight, geometry in outgoing:
                    coordinates = geometry or [
                        [nodes[node][1], nodes[node][0]],
                        [nodes[neighbor][1], nodes[neighbor][0]],
                    ]
                    if blocked is not None and blocked(coordinates):
                        removed += 1
                        continue
                    adj.setdefault(node, []).append(
                        (neighbor, weight, geometry)
                    )
        else:
            for node in nodes:
                i, j = node
                for neighbor in ((i + 1, j), (i, j + 1)):
                    if neighbor not in nodes:
                        continue
                    weight = geodesic_km(
                        nodes[node][0], nodes[node][1],
                        nodes[neighbor][0], nodes[neighbor][1],
                    ) * 1000.0
                    coordinates = [
                        [nodes[node][1], nodes[node][0]],
                        [nodes[neighbor][1], nodes[neighbor][0]],
                    ]
                    if blocked is not None and blocked(coordinates):
                        removed += 1
                        continue
                    adj.setdefault(node, []).append(
                        (neighbor, weight, coordinates)
                    )
                    adj.setdefault(neighbor, []).append(
                        (node, weight, list(reversed(coordinates)))
                    )

        result = _demo_dijkstra(adj, src, goal)
        if result is None:
            if blocked is None:
                return json.dumps({
                    "error": "no_path_found",
                    "reason": "No connected base route",
                    "flooded_edges_removed": removed,
                }, ensure_ascii=False)
            fallback_args = dict(args)
            fallback_args["avoid_polygon_path"] = ""
            fallback = json.loads(self._fx_vec_shortest_path(fallback_args))
            fallback["flooded_edges_removed"] = removed
            fallback["route_avoids_flood"] = False
            return json.dumps(fallback, ensure_ascii=False)
        total_m, legs = result

        coords = []
        for u, v, geom in legs:
            pts = geom or [
                [nodes[u][1], nodes[u][0]],
                [nodes[v][1], nodes[v][0]],
            ]
            if coords and coords[-1] == pts[0]:
                coords.extend(pts[1:])
            else:
                coords.extend(pts)
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "path_length_m": round(total_m, 1),
                    "flooded_edges_avoided": removed,
                    "travel_mode": travel_mode,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }],
        }
        return json.dumps({
            "output_path": _write_fixture_file(
                json.dumps(geojson), ".geojson"
            ) or "/tmp/demo_route.geojson",
            "path_length_m": round(total_m, 1),
            "path_length_km": round(total_m / 1000.0, 2),
            "travel_time_min": round(total_m / speed / 60.0, 1),
            "travel_time_basis": {
                "type": "conservative_assumption",
                "speed_kmh": round(speed * 3.6, 1),
            },
            "node_count": len(legs) + 1,
            "flooded_edges_removed": removed,
            "route_avoids_flood": blocked is not None,
            "download_geometry": "origin_destination_corridor",
            "corridor_half_width_km": float(
                args.get("search_radius_km", 2.0)
            ),
            "origin": {"lat": origin[0], "lon": origin[1]},
            "destination": {"lat": dest[0], "lon": dest[1]},
        }, ensure_ascii=False)

    def _fx_vis_flood_map(self, args):
        # The pipeline reads the output_path key (consistent with the
        # gis_vector tool family).
        return {
            "status": "ok",
            "output_path": _write_fixture_file(
                _DEMO_MAP_HTML, ".html"
            ) or "/tmp/demo_map.html",
        }


FUSION_SOURCES = [
    {"name": "river_level", "source_type": "hydrology",
     "tool": "get_flood_observation", "enabled": True,
     "arguments": {"station_id": "{station_id}",
                   "start_dt": "{event_start}", "end_dt": "{event_end}"}},
    {"name": "weather", "source_type": "weather",
     "tool": "get_weather_observations", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "precipitation", "source_type": "precipitation",
     "tool": "get_precipitation", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "forecast", "source_type": "forecast",
     "tool": "get_forecast", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "flood_warnings", "source_type": "warning",
     "tool": "get_flood_warnings", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "population_exposure", "source_type": "exposure",
     "tool": "get_population_exposure", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "GEE Sentinel-1 Flood Extent", "source_type": "satellite_sar",
     "tool": "get_flood_extent", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}",
                   "observation_date": "{event_date}",
                   "buffer_km": "{flood_analysis_radius_km}",
                   "post_days": 3, "return_geometry": True}},
    {"name": "critical_infrastructure", "source_type": "infrastructure",
     "tool": "get_critical_infrastructure", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
    {"name": "road_network", "source_type": "road",
     "tool": "get_road_status", "enabled": True,
     "arguments": {"latitude": "{latitude}", "longitude": "{longitude}"}},
]


# Environment variables for the demo run (applied temporarily for the
# run only, restored afterwards; operator-set values in .env / the
# process environment take precedence — setdefault semantics).
DEMO_ENV = {
    "FLOOD_STATION_MAX_DISTANCE_KM": "50",
    "SVI_RADIUS_KM": "10",
    "SVI_MAX_FEATURES": "500",
    "VULNERABILITY_WEIGHT": "1.0",
    "EQUITY_HIGH_VULNERABILITY_THRESHOLD": "0.90",
    "FLOOD_FUSION_SOURCES_JSON": None,  # filled by json.dumps below
    "RESOURCE_DISCOVERY_TOOL": "get_available_resources",
    "RESOURCE_DISCOVERY_ARGUMENTS_JSON": None,
    "RESOURCE_ALLOCATION_OBJECTIVES_JSON": None,
    "RESOURCE_OBJECTIVE_WEIGHTS_JSON": None,
    "RESOURCE_MAX_PLAN_COMBINATIONS": "1000",
    # 2 km: each demo candidate covers exactly one tract, keeping the
    # equity metrics non-trivial
    "VULNERABILITY_COVERAGE_RADIUS_KM": "2",
}


class _DemoEnv:
    """Temporarily apply DEMO_ENV inside the with block (setdefault
    semantics), restoring prior state on exit so demo-only parameters
    (SVI radius, coverage radius, ...) never leak into later requests
    handled by the same process.
    """

    def __init__(
        self,
        env: dict[str, str | None],
        force_keys: set[str] | None = None,
    ):
        self._env = {k: v for k, v in env.items() if v is not None}
        self._force_keys = force_keys or set()
        self._added: list[str] = []
        self._replaced: dict[str, str] = {}

    def __enter__(self) -> "_DemoEnv":
        for key, value in self._env.items():
            if key in self._force_keys and key in os.environ:
                self._replaced[key] = os.environ[key]
                os.environ[key] = value
            elif key not in os.environ:
                os.environ[key] = value
                self._added.append(key)
        return self

    def __exit__(self, *_exc) -> None:
        for key in self._added:
            os.environ.pop(key, None)
        for key, value in self._replaced.items():
            os.environ[key] = value


async def run_demo_assessment(
    target: str = "Friendswood",
    station_id: str = "08077600",
    raw_task: str = "Assess flood around Friendswood station 08077600",
    event_date: str | None = None,
    vulnerability_weight: float | None = None,
    equity_threshold: float | None = None,
    community_requirements: list[str] | None = None,
    community_source: str | None = None,
    community_note: str | None = None,
    demo_scenario: str = "nominal",
) -> tuple[Any, Any]:
    """
    Run the full flood pipeline on FakeMCP; returns (SkillResult, RunState).

    Shared by Web Demo mode and tests; works on any date with zero
    network dependency. Pass a past event_date for historical replay
    (date-aware fixtures keep time alignment, so gates always pass).
    """
    if demo_scenario not in DEMO_SCENARIOS:
        raise ValueError(
            f"Unknown demo scenario {demo_scenario!r}; expected one of: "
            + ", ".join(sorted(DEMO_SCENARIOS))
        )

    # Fill the JSON strings in a copy; never mutate the module-level
    # DEMO_ENV constant.
    demo_env = dict(DEMO_ENV)
    fusion_sources = list(FUSION_SOURCES)
    if demo_scenario == "fusion_conflict":
        fusion_sources.append({
            "name": "backup_gauge_conflict_demo",
            "source_type": "hydrology",
            "tool": "get_flood_observation",
            "enabled": True,
            "arguments": {
                "station_id": "{station_id}",
                "start_dt": "{event_start}",
                "end_dt": "{event_end}",
                "demo_conflict": True,
            },
        })
    demo_env["FLOOD_FUSION_SOURCES_JSON"] = json.dumps(fusion_sources)
    demo_env["RESOURCE_DISCOVERY_ARGUMENTS_JSON"] = json.dumps(
        {"latitude": "{latitude}", "longitude": "{longitude}"}
    )
    demo_env["RESOURCE_ALLOCATION_OBJECTIVES_JSON"] = json.dumps([
        {"name": "risk_reduction", "direction": "maximize"},
        {"name": "coverage", "direction": "maximize"},
        {"name": "response_time", "direction": "minimize"},
        {"name": "cost", "direction": "minimize"},
        {"name": "unmet_demand", "direction": "minimize"},
    ])
    demo_env["RESOURCE_OBJECTIVE_WEIGHTS_JSON"] = json.dumps({
        "risk_reduction": 0.35, "coverage": 0.25,
        "response_time": 0.20, "cost": 0.10, "unmet_demand": 0.10,
    })

    from .hitl import AdaptiveHITL
    from .models import RunState
    from .skills import FloodSkill
    from .verification import Verifier
    from .experiment import ExperimentLogger

    overrides: dict[str, Any] = {}
    effective_event_date = event_date
    if demo_scenario == "stale_sar" and not effective_event_date:
        effective_event_date = "2017-08-27"
    if effective_event_date:
        overrides["event_date"] = effective_event_date
    if vulnerability_weight is not None:
        overrides["vulnerability_weight"] = vulnerability_weight
    if equity_threshold is not None:
        overrides["equity_threshold"] = equity_threshold
    if community_requirements:
        overrides["community_requirements"] = community_requirements
    if community_source:
        overrides["community_source"] = community_source
    if community_note:
        overrides["community_note"] = community_note

    state = RunState(run_id=f"demo-{datetime.now(timezone.utc).strftime('%H%M%S')}")
    if demo_scenario == "far_station":
        target = "Seattle"
        raw_task = "Assess flood around Seattle station 08077600"

    mcp = FakeMCP(scenario=demo_scenario)
    hitl = AdaptiveHITL(state)
    # Fault demos must be deterministic and unattended. Safety-critical
    # checkpoints therefore take their documented denial default rather
    # than waiting 120 seconds for a web response.
    hitl.enabled = False
    hitl.enable_web_mode()
    # Demo parameters apply only during the run (FloodSkill reads env
    # vars at construction, so construction must stay inside the
    # context too); the context restores state on exceptions as well
    with _DemoEnv(
        demo_env,
        # The conflict preset adds a controlled second hydrology source;
        # do not let a process-level fusion config hide that preset.
        force_keys={"FLOOD_FUSION_SOURCES_JSON"},
    ):
        skill = FloodSkill(
            state, Verifier(), hitl, mcp,
            ExperimentLogger(path=os.devnull),
        )
        result = await skill.run(
            target, station_id=station_id, raw_task=raw_task,
            overrides=overrides or None,
        )
    return result, state
