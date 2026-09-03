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
# with osmnx: directed drive network with street polylines): endpoints
# snap to the nearest nodes, flooded edges are removed via shapely
# intersection tests, Dijkstra finds the shortest path by edge length —
# same semantics as gis_vector_server.vec_shortest_path. A disconnected
# graph returns no_path_found (the caller retries without flood
# avoidance); travel time uses conservative speeds (drive 30 km/h).
# Falls back to a deterministic synthetic grid network when the cache
# is missing, so the demo always runs.
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
    """Fallback network: synthetic grid nodes covering the origin-dest bounding box (padded by 2 cells)."""
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
    """Load the real OSM road cache; return None if missing or corrupt
    (caller falls back to the synthetic grid).

    Returns (nodes, adj): nodes = {id: (lat, lon)}; adj =
    {u: [(v, length_m, geom)]} where geom is the street polyline of a
    simplified OSM edge ([[lon, lat], ...], oriented u -> v) or None
    (straight edge). Directed: one-way semantics match the real tool.
    """
    global _demo_road_cache, _demo_road_cache_loaded
    if _demo_road_cache_loaded:
        return _demo_road_cache
    _demo_road_cache_loaded = True
    try:
        with open(_DEMO_ROAD_CACHE_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        nodes = {k: (float(v[0]), float(v[1])) for k, v in payload["nodes"].items()}
        adj: dict = {}
        for u, v, length_m, geom in payload["edges"]:
            w = float(length_m)
            adj.setdefault(str(u), []).append((str(v), w, geom))
        _demo_road_cache = (nodes, adj)
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

    def __init__(self):
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
        # requested window (time alignment is verified by the pipeline)
        obs_time = NOW
        if args.get("start_dt"):
            try:
                _start = datetime.fromisoformat(
                    str(args["start_dt"]).replace("Z", "+00:00")
                )
                obs_time = (_start + timedelta(hours=6)).isoformat()
            except ValueError:
                pass
        return {
            "status": "ok",
            "query_window": (
                {"start": args.get("start_dt"), "end": args.get("end_dt")}
                if args.get("start_dt")
                else None
            ),
            "observation": {
                "station_id": "08077600",
                "water_level": 8.0,
                "unit": "ft",
                "observation_time": obs_time,
                "source": "USGS",
                "station_name": "Clear Ck nr Friendswood, TX",
                "latitude": 29.5175,
                "longitude": -95.1785,
                "metadata_verified": True,
                "location_verified": True,
            },
        }

    def _fx_get_nwps_gauge(self, args):
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
            "source_type": "population_exposure",
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
        return {
            "status": "ok",
            "source": "OSM",
            "measurements": {
                "facility_count": {"value": 12, "unit": "facilities"},
            },
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
        return {
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
        # Computed over the real OSM road network (synthetic grid
        # fallback when the cache is missing): snap origin/destination
        # → optionally remove flooded edges via the flood polygon →
        # Dijkstra → street-polyline GeoJSON with consistent
        # length/travel time. The hospital sits inside the demo flood
        # polygon, so flood-avoiding requests always yield
        # no_path_found; the pipeline then retries without flood
        # avoidance per the real contract and discloses
        # route_avoids_flood.
        origin = (float(args["origin_lat"]), float(args["origin_lon"]))
        dest = (float(args["dest_lat"]), float(args["dest_lon"]))
        travel_mode = str(args.get("travel_mode", "drive") or "drive")
        speed = _TRAVEL_SPEED_MPS.get(
            travel_mode, _TRAVEL_SPEED_MPS["drive"]
        )

        blocked = _demo_flood_predicate(
            str(args.get("avoid_polygon_path", "") or "")
        )

        cache = _load_demo_road_cache()
        if cache is not None:
            nodes, adj_full = cache

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
            for u, outs in adj_full.items():
                for v, w, geom in outs:
                    coords = geom or [
                        [nodes[u][1], nodes[u][0]],
                        [nodes[v][1], nodes[v][0]],
                    ]
                    if blocked is not None and blocked(coords):
                        removed += 1
                        continue
                    adj.setdefault(u, []).append((v, w, geom))
        else:
            # Fallback: deterministic synthetic grid network (demo runs
            # even without the real cache)
            nodes = _demo_street_grid(origin, dest)

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
            for node in nodes:
                i, j = node
                for nb in ((i + 1, j), (i, j + 1)):
                    if nb not in nodes:
                        continue
                    w = geodesic_km(
                        nodes[node][0], nodes[node][1],
                        nodes[nb][0], nodes[nb][1],
                    ) * 1000.0
                    coords = [
                        [nodes[node][1], nodes[node][0]],
                        [nodes[nb][1], nodes[nb][0]],
                    ]
                    if blocked is not None and blocked(coords):
                        removed += 1
                        continue
                    adj.setdefault(node, []).append((nb, w, coords))

        result = _demo_dijkstra(adj, src, goal)
        if result is None:
            # Same contract as the real tool: disconnected after flood
            # avoidance → caller retries without flood avoidance.
            return json.dumps({
                "error": "no_path_found",
                "reason": (
                    "No connected route remains after flood-edge removal"
                ),
                "flooded_edges_removed": removed,
            }, ensure_ascii=False)
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
    "NWS_WARNING_RADIUS_KM": "25",
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

    def __init__(self, env: dict[str, str | None]):
        self._env = {k: v for k, v in env.items() if v is not None}
        self._added: list[str] = []

    def __enter__(self) -> "_DemoEnv":
        for key, value in self._env.items():
            if key not in os.environ:
                os.environ[key] = value
                self._added.append(key)
        return self

    def __exit__(self, *_exc) -> None:
        for key in self._added:
            os.environ.pop(key, None)


async def run_demo_assessment(
    target: str = "Friendswood",
    station_id: str = "08077600",
    raw_task: str = "Assess flood around Friendswood station 08077600",
    event_date: str | None = None,
    vulnerability_weight: float | None = None,
    equity_threshold: float | None = None,
) -> tuple[Any, Any]:
    """
    Run the full flood pipeline on FakeMCP; returns (SkillResult, RunState).

    Shared by Web Demo mode and tests; works on any date with zero
    network dependency. Pass a past event_date for historical replay
    (date-aware fixtures keep time alignment, so gates always pass).
    """
    # Fill the JSON strings in a copy; never mutate the module-level
    # DEMO_ENV constant.
    demo_env = dict(DEMO_ENV)
    demo_env["FLOOD_FUSION_SOURCES_JSON"] = json.dumps(FUSION_SOURCES)
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
    if event_date:
        overrides["event_date"] = event_date
    if vulnerability_weight is not None:
        overrides["vulnerability_weight"] = vulnerability_weight
    if equity_threshold is not None:
        overrides["equity_threshold"] = equity_threshold

    state = RunState(run_id=f"demo-{datetime.now(timezone.utc).strftime('%H%M%S')}")
    mcp = FakeMCP()
    hitl = AdaptiveHITL(state)
    hitl.enable_web_mode()  # parameter checkpoints auto-accept defaults so the demo never blocks
    # Demo parameters apply only during the run (FloodSkill reads env
    # vars at construction, so construction must stay inside the
    # context too); the context restores state on exceptions as well
    with _DemoEnv(demo_env):
        skill = FloodSkill(
            state, Verifier(), hitl, mcp,
            ExperimentLogger(path=os.devnull),
        )
        result = await skill.run(
            target, station_id=station_id, raw_task=raw_task,
            overrides=overrides or None,
        )
    return result, state
