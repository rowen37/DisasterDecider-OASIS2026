#!/usr/bin/env python3
"""
GIS Vector Operations MCP Server.

Provides deterministic vector-GIS tools for disaster-response workflows:
buffer, intersection, spatial counting, flood-avoiding shortest path,
OpenStreetMap POI search, difference, and centroid calculation.

All GeoJSON outputs are written in EPSG:4326 unless otherwise stated.
Metric geometry operations use a local UTM CRS selected from the input
geometry to provide meter-based calculations with better local accuracy.
"""

import json
import logging
import math
import os
import sys
from pathlib import Path

from gis_io import out_path as _out
from typing import Any

import geopandas as gpd
from pyproj import CRS, Geod
from shapely.geometry import LineString, Point
from mcp.server.fastmcp import FastMCP

from overpass_client import (
    OverpassError,
    configured_endpoints,
    post_overpass_sync,
)
from osm_labels import load_label_overrides, osm_display_label


logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("gis-vector")

mcp = FastMCP("gis-vector-tools")

WGS84 = "EPSG:4326"
MAX_ROUTE_CORRIDOR_KM = 8.0
GEOD = Geod(ellps="WGS84")
SUPPORTED_TRAVEL_MODES = {"drive", "walk", "bike"}
# Conservative disaster-condition speeds. These are explicitly reported
# as assumptions rather than measurements from a live traffic service.
TRAVEL_SPEED_MPS = {
    "drive": 30.0 / 3.6,
    "walk": 4.0 / 3.6,
    "bike": 12.0 / 3.6,
}


def _osmnx_overpass_base(endpoint: str) -> str:
    """Convert the shared client's interpreter URL to OSMnx's base URL."""

    value = endpoint.rstrip("/")
    suffix = "/interpreter"
    return value[:-len(suffix)] if value.endswith(suffix) else value

def _json(payload: dict[str, Any]) -> str:
    """Serialize tool output consistently."""
    return json.dumps(payload, ensure_ascii=False)


def _read_with_crs(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf

def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of a GeoDataFrame transformed to WGS84."""
    if gdf.crs is None:
        raise ValueError("Cannot transform a GeoDataFrame with undefined CRS")
    return gdf.to_crs(WGS84)

def _metric_crs(gdf: gpd.GeoDataFrame, layer_path: str, tool: str) -> CRS:
    """
    Select a local projected CRS suitable for meter-based geometry operations.

    Disaster-response layers are normally local/regional, so a local UTM CRS
    is preferable to Web Mercator for buffering, centroid, and area work.
    """
    if gdf.crs is None:
        raise ValueError(
            f"Input layer has no CRS defined: {layer_path} "
            f"(tool={tool})"
        )

    estimated = gdf.estimate_utm_crs()
    if estimated is None:
        raise ValueError(
            f"Unable to determine a local projected CRS for {layer_path} "
            f"(tool={tool})"
        )
    return estimated


@mcp.tool()
def vec_buffer(
    input_path: str,
    buffer_meters: float,
    output_path: str = "",
    dissolve: bool = True,
) -> str:
    """
    Create a buffer around vector features.

    Args:
        input_path: Input vector layer.
        buffer_meters: Buffer distance in meters; negative values shrink.
        output_path: Output GeoJSON; auto-generated if empty.
        dissolve: Merge all buffered geometries into one feature.

    Returns:
        JSON with output path, feature count, and metric area.
    """
    try:
        if not isinstance(buffer_meters, (int, float)):
            return _json({"error": "buffer_meters must be numeric", "tool": "vec_buffer"})

        output_path = _out(output_path, ".geojson")
        gdf = _read_with_crs(input_path)

        if gdf.empty:
            return _json({"error": "Input vector layer is empty", "tool": "vec_buffer"})

        # Buffering is performed in a local projected CRS with meter units.
        metric_crs = _metric_crs(gdf, input_path, "vec_buffer")
        gdf_m = gdf.to_crs(metric_crs)
        gdf_m["geometry"] = gdf_m.geometry.buffer(buffer_meters)

        if dissolve:
            gdf_m = gdf_m.dissolve().reset_index(drop=True)

        total_area_m2 = float(gdf_m.geometry.area.sum())
        gdf_out = _to_wgs84(gdf_m)
        gdf_out.to_file(output_path, driver="GeoJSON")

        return _json({
            "output_path": output_path,
            "buffer_meters": buffer_meters,
            "feature_count": len(gdf_out),
            "total_area_km2": round(total_area_m2 / 1e6, 4),
            "total_area_m2": round(total_area_m2, 1),
            "dissolved": dissolve,
            "_meta": {
                "crs": WGS84,
                "metric_crs": str(metric_crs),
                "type": "polygon",
                "tool": "vec_buffer",
                "source": input_path,
            },
        })
    except Exception as exc:
        log.exception("vec_buffer failed")
        return _json({"error": str(exc), "tool": "vec_buffer"})


@mcp.tool()
def vec_intersect(
    layer1_path: str,
    layer2_path: str,
    output_path: str = "",
    keep_fields: str = "",
) -> str:
    """
    Compute the spatial intersection of two vector layers.

    The operation is performed in a local UTM CRS (estimate_utm_crs) for
    meter-based geometry/area calculations, and the output is written as
    EPSG:4326.
    """
    try:
        output_path = _out(output_path, ".geojson")
        gdf1 = _read_with_crs(layer1_path)
        gdf2 = _read_with_crs(layer2_path)

        if gdf1.empty or gdf2.empty:
            empty = gpd.GeoDataFrame(geometry=[], crs=WGS84)
            empty.to_file(output_path, driver="GeoJSON")
            return _json({
                "output_path": output_path,
                "intersected_count": 0,
                "total_area_km2": 0.0,
                "warning": "One or both input layers are empty",
                "_meta": {"crs": WGS84, "tool": "vec_intersect"},
            })

        metric_crs = _metric_crs(gdf1, layer1_path, "vec_intersect")
        gdf1_m = gdf1.to_crs(metric_crs)
        gdf2_m = gdf2.to_crs(metric_crs)

        result = gpd.overlay(
            gdf1_m,
            gdf2_m,
            how="intersection",
            keep_geom_type=False,
        )
        result = result[
            ~result.geometry.is_empty & result.geometry.notna()
        ].copy()

        if result.empty:
            empty = gpd.GeoDataFrame(geometry=[], crs=WGS84)
            empty.to_file(output_path, driver="GeoJSON")
            return _json({
                "output_path": output_path,
                "intersected_count": 0,
                "total_area_km2": 0.0,
                "_meta": {"crs": WGS84, "tool": "vec_intersect"},
            })

        if keep_fields:
            fields = [
                field.strip()
                for field in keep_fields.split(",")
                if field.strip() in result.columns
            ]
            result = result[fields + ["geometry"]]

        total_area_m2 = float(result.geometry.area.sum())
        result_wgs = _to_wgs84(result)
        result_wgs.to_file(output_path, driver="GeoJSON")

        feature_list = [
            {
                col: str(row[col])
                for col in result_wgs.columns
                if col != "geometry" and row[col] is not None
            }
            for _, row in result_wgs.head(50).iterrows()
        ]

        return _json({
            "output_path": output_path,
            "intersected_count": len(result_wgs),
            "total_area_km2": round(total_area_m2 / 1e6, 4),
            "total_area_m2": round(total_area_m2, 1),
            "feature_list": feature_list,
            "_meta": {
                "crs": WGS84,
                "metric_crs": str(metric_crs),
                "type": "vector",
                "tool": "vec_intersect",
            },
        })
    except Exception as exc:
        log.exception("vec_intersect failed")
        return _json({"error": str(exc), "tool": "vec_intersect"})


@mcp.tool()
def vec_count_within(
    polygon_path: str,
    features_path: str,
    category_field: str = "",
) -> str:
    """
    Count features inside or intersecting a polygon zone.

    Polygon features are represented by their centroids to avoid counting a
    partially overlapping polygon as fully inside the zone.
    """
    try:
        poly = _read_with_crs(polygon_path)
        feat = _read_with_crs(features_path)

        if poly.empty:
            return _json({
                "total_count": 0,
                "warning": "Polygon layer empty",
                "_meta": {"type": "scalar", "tool": "vec_count_within"},
            })
        if feat.empty:
            return _json({
                "total_count": 0,
                "warning": "Features layer empty",
                "_meta": {"type": "scalar", "tool": "vec_count_within"},
            })

        metric_crs = _metric_crs(poly, polygon_path, "vec_count_within")
        poly_m = poly.to_crs(metric_crs)
        feat_m = feat.to_crs(metric_crs)
        poly_union = poly_m.geometry.union_all()

        if feat_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            check_geom = feat_m.geometry.centroid
        else:
            check_geom = feat_m.geometry

        inside = feat_m[
            check_geom.within(poly_union) | check_geom.intersects(poly_union)
        ].copy()

        result: dict[str, Any] = {
            "total_count": len(inside),
            "_meta": {
                "type": "scalar",
                "tool": "vec_count_within",
                "metric_crs": str(metric_crs),
            },
        }

        if category_field and category_field in inside.columns:
            result["category_counts"] = {
                str(k): int(v)
                for k, v in inside[category_field].value_counts().items()
            }

        result["feature_list"] = [
            {
                col: str(row[col])
                for col in inside.columns
                if col != "geometry" and row[col] is not None
            }
            for _, row in inside.head(50).iterrows()
        ]

        return _json(result)
    except Exception as exc:
        log.exception("vec_count_within failed")
        return _json({"error": str(exc), "tool": "vec_count_within"})


@mcp.tool()
def vec_shortest_path(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    avoid_polygon_path: str = "",
    travel_mode: str = "drive",
    search_radius_km: float = 2.0,
    expanded_search_radius_km: float = 4.0,
    output_path: str = "",
) -> str:
    """
    Find a shortest OSM road/path route, optionally removing flood-intersecting edges.

    Args:
        origin_lat/lon: Origin in WGS84.
        dest_lat/lon: Destination in WGS84.
        avoid_polygon_path: Optional flood polygon GeoJSON.
        travel_mode: drive, walk, or bike.
        search_radius_km: Initial half-width of the origin-to-destination
            road-download corridor, in km.
        expanded_search_radius_km: One wider corridor used only when the
            initial downloaded graph has no connected base route.
        output_path: Output GeoJSON; auto-generated if empty.

    Notes:
        Path length uses the OSMnx network's edge length attributes.
        Travel time is an explicit conservative speed assumption, not live traffic.
    """
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError:
        return _json({
            "error": "osmnx and networkx are required. Install with: pip install osmnx networkx",
            "tool": "vec_shortest_path",
        })

    try:
        if travel_mode not in SUPPORTED_TRAVEL_MODES:
            return _json({
                "error": f"Unsupported travel_mode: {travel_mode}",
                "allowed": sorted(SUPPORTED_TRAVEL_MODES),
                "tool": "vec_shortest_path",
            })

        if search_radius_km <= 0 or expanded_search_radius_km <= 0:
            return _json({
                "error": "route corridor widths must be greater than 0",
                "tool": "vec_shortest_path",
            })
        if expanded_search_radius_km < search_radius_km:
            return _json({
                "error": (
                    "expanded_search_radius_km must be greater than or "
                    "equal to search_radius_km"
                ),
                "tool": "vec_shortest_path",
            })

        for name, value in (
            ("origin_lat", origin_lat),
            ("origin_lon", origin_lon),
            ("dest_lat", dest_lat),
            ("dest_lon", dest_lon),
        ):
            if "lon" in name:
                valid = -180.0 <= value <= 180.0
            else:
                valid = -90.0 <= value <= 90.0
            if not valid:
                return _json({
                    "error": f"Invalid coordinate: {name}={value}",
                    "tool": "vec_shortest_path",
                })

        output_path = _out(output_path, ".geojson")

        # Geodesic distance in meters (not a degree-based approximation).
        _, _, straight_m = GEOD.inv(
            origin_lon,
            origin_lat,
            dest_lon,
            dest_lat,
        )
        # OSMnx otherwise waits up to 180 seconds inside its synchronous
        # Overpass request and defaults to one fixed host. Keep each attempt
        # bounded and rotate across the same configured mirrors used by the
        # project's other OSM tools.
        route_request_timeout = max(
            1.0,
            float(os.getenv("OSM_ROUTE_REQUEST_TIMEOUT_SECONDS", "20")),
        )
        # OSMnx embeds requests_timeout in Overpass QL as [timeout:N].
        # Overpass accepts an integer there, not a decimal such as 20.0.
        route_query_timeout = max(1, math.ceil(route_request_timeout))
        route_max_downloads = max(
            1,
            min(2, int(os.getenv("OSM_ROUTE_MAX_DOWNLOAD_ATTEMPTS", "2"))),
        )
        route_endpoints = configured_endpoints()[:route_max_downloads]

        route_line = gpd.GeoDataFrame(
            geometry=[LineString([
                (origin_lon, origin_lat),
                (dest_lon, dest_lat),
            ])],
            crs=WGS84,
        )
        route_crs = route_line.estimate_utm_crs()
        if route_crs is None:
            return _json({
                "error": "Unable to determine a projected CRS for route corridor",
                "tool": "vec_shortest_path",
            })

        corridor_widths = []
        for value in (search_radius_km, expanded_search_radius_km):
            width = min(float(value), MAX_ROUTE_CORRIDOR_KM)
            if width not in corridor_widths:
                corridor_widths.append(width)

        G = None
        orig_node = None
        dest_node = None
        used_corridor_km = None
        route_overpass_attempts: list[dict[str, Any]] = []
        download_count = 0
        successful_endpoint: str | None = None
        downloaded_graph = False
        # OSMnx's private requester sleeps 55 seconds and recursively retries
        # the same host on every 429/504. Replace only that request function
        # for this bounded route call with the project's shared client. OSMnx
        # still builds/simplifies the graph; HTTP now obeys our per-attempt
        # budget and selected mirror.
        from osmnx import _overpass as ox_overpass

        active_route_endpoint: dict[str, str | None] = {"value": None}

        def _bounded_route_overpass(data: Any) -> dict[str, Any]:
            endpoint = active_route_endpoint["value"]
            if not endpoint:
                raise OverpassError("No active route Overpass endpoint")
            query = str(data.get("data") or "")
            payload, _diag = post_overpass_sync(
                query,
                server_timeout_s=route_request_timeout,
                total_budget_s=route_request_timeout,
                endpoints=[endpoint],
            )
            return payload

        previous_overpass_url = ox.settings.overpass_url
        previous_request_timeout = ox.settings.requests_timeout
        previous_rate_limit = ox.settings.overpass_rate_limit
        previous_overpass_request = ox_overpass._overpass_request
        try:
            ox.settings.requests_timeout = route_query_timeout
            # Avoid a separate /status wait per mirror. The route layer has a
            # strict two-download budget instead of waiting for public slots.
            ox.settings.overpass_rate_limit = False
            ox_overpass._overpass_request = _bounded_route_overpass

            for width_km in corridor_widths:
                if download_count >= route_max_downloads:
                    break
                corridor = (
                    route_line.to_crs(route_crs)
                    .geometry.buffer(width_km * 1_000.0)
                    .to_crs(WGS84)
                    .iloc[0]
                )

                # A transport/server failure rotates to the next mirror at
                # the same width. A successfully downloaded but disconnected
                # graph spends the one remaining attempt widening the same
                # corridor, because changing mirrors cannot change topology.
                if successful_endpoint:
                    endpoints_for_width = [successful_endpoint]
                else:
                    endpoints_for_width = route_endpoints[download_count:]

                candidate = None
                candidate_endpoint = None
                for endpoint in endpoints_for_width:
                    if download_count >= route_max_downloads:
                        break
                    download_count += 1
                    active_route_endpoint["value"] = endpoint
                    ox.settings.overpass_url = _osmnx_overpass_base(endpoint)
                    log.info(
                        "Downloading OSM route corridor length=%.1fkm "
                        "half_width=%.1fkm endpoint=%s attempt=%d/%d",
                        straight_m / 1_000.0,
                        width_km,
                        endpoint,
                        download_count,
                        route_max_downloads,
                    )
                    try:
                        candidate = ox.graph_from_polygon(
                            corridor,
                            network_type=travel_mode,
                            simplify=True,
                            retain_all=True,
                            truncate_by_edge=True,
                        )
                    except Exception as endpoint_exc:
                        route_overpass_attempts.append({
                            "endpoint": endpoint,
                            "corridor_half_width_km": width_km,
                            "status": "failed",
                            "error": str(endpoint_exc)[:240],
                        })
                        log.warning(
                            "Route Overpass mirror failed (%s): %s",
                            endpoint,
                            endpoint_exc,
                        )
                        candidate = None
                        # A 400 response means Overpass rejected the query
                        # itself. Sending the same invalid query to another
                        # mirror only adds delay and cannot recover it.
                        if str(endpoint_exc).startswith(
                            "Overpass rejected the query"
                        ):
                            break
                        continue

                    candidate_endpoint = endpoint
                    successful_endpoint = endpoint
                    downloaded_graph = True
                    route_overpass_attempts.append({
                        "endpoint": endpoint,
                        "corridor_half_width_km": width_km,
                        "status": "downloaded",
                    })
                    break

                if candidate is None:
                    break

                candidate_orig = ox.distance.nearest_nodes(
                    candidate, origin_lon, origin_lat
                )
                candidate_dest = ox.distance.nearest_nodes(
                    candidate, dest_lon, dest_lat
                )
                if nx.has_path(candidate, candidate_orig, candidate_dest):
                    G = candidate
                    orig_node = candidate_orig
                    dest_node = candidate_dest
                    used_corridor_km = width_km
                    successful_endpoint = candidate_endpoint
                    break
                log.warning(
                    "No connected base route in %.1fkm corridor; expanding once",
                    width_km,
                )
        finally:
            ox.settings.overpass_url = previous_overpass_url
            ox.settings.requests_timeout = previous_request_timeout
            ox.settings.overpass_rate_limit = previous_rate_limit
            ox_overpass._overpass_request = previous_overpass_request

        if G is None or orig_node is None or dest_node is None:
            if not downloaded_graph:
                return _json({
                    "error": "overpass_route_download_failed",
                    "reason": "All bounded route-network mirror attempts failed",
                    "overpass_attempts": route_overpass_attempts,
                    "_meta": {"tool": "vec_shortest_path"},
                })
            return _json({
                "error": "no_path_found",
                "reason": "No connected base route in the configured corridors",
                "corridor_widths_tried_km": corridor_widths,
                "overpass_attempts": route_overpass_attempts,
                "_meta": {"tool": "vec_shortest_path"},
            })

        # Preserve the downloaded base graph. Flood avoidance operates on a
        # copy, so a disconnected avoiding route can fall back to the same
        # graph without a second Overpass download.
        base_graph = G
        route_graph = G.copy() if avoid_polygon_path else G
        route_avoids_flood = bool(avoid_polygon_path)

        flooded_removed = 0

        if avoid_polygon_path:
            if not os.path.exists(avoid_polygon_path):
                return _json({
                    "error": "avoid_polygon_path does not exist",
                    "layer": avoid_polygon_path,
                    "tool": "vec_shortest_path",
                })

            flood_gdf = _read_with_crs(avoid_polygon_path)
            flood_gdf = flood_gdf.to_crs(WGS84)

            if not flood_gdf.empty:
                flood_union = flood_gdf.geometry.union_all()
                to_remove: list[tuple[Any, Any, Any]] = []

                for u, v, k, data in route_graph.edges(keys=True, data=True):
                    geom = data.get("geometry")
                    if geom is None:
                        geom = LineString([
                            (route_graph.nodes[u]["x"], route_graph.nodes[u]["y"]),
                            (route_graph.nodes[v]["x"], route_graph.nodes[v]["y"]),
                        ])
                    if geom.intersects(flood_union):
                        to_remove.append((u, v, k))

                route_graph.remove_edges_from(to_remove)
                flooded_removed = len(to_remove)
                log.info("Removed %d flood-intersecting edges", flooded_removed)

        if not nx.has_path(route_graph, orig_node, dest_node):
            route_graph = base_graph
            route_avoids_flood = False

        path_nodes = nx.shortest_path(
            route_graph,
            orig_node,
            dest_node,
            weight="length",
        )
        path_length = float(nx.shortest_path_length(
            route_graph,
            orig_node,
            dest_node,
            weight="length",
        ))

        coords = [
            (route_graph.nodes[n]["x"], route_graph.nodes[n]["y"])
            for n in path_nodes
        ]
        route_gdf = gpd.GeoDataFrame(
            [{
                "path_length_m": round(path_length, 1),
                "flooded_edges_avoided": flooded_removed,
                "travel_mode": travel_mode,
            }],
            geometry=[LineString(coords)],
            crs=WGS84,
        )
        route_gdf.to_file(output_path, driver="GeoJSON")

        speed_mps = TRAVEL_SPEED_MPS[travel_mode]
        travel_min = round(path_length / speed_mps / 60.0, 1)

        return _json({
            "output_path": output_path,
            "path_length_m": round(path_length, 1),
            "path_length_km": round(path_length / 1000.0, 2),
            "travel_time_min": travel_min,
            "travel_time_basis": {
                "type": "conservative_assumption",
                "speed_kmh": round(speed_mps * 3.6, 1),
            },
            "node_count": len(path_nodes),
            "flooded_edges_removed": flooded_removed,
            "route_avoids_flood": route_avoids_flood,
            "download_geometry": "origin_destination_corridor",
            "corridor_half_width_km": used_corridor_km,
            "corridor_widths_tried_km": [
                width for width in corridor_widths
                if used_corridor_km is None or width <= used_corridor_km
            ],
            "overpass_endpoint": successful_endpoint,
            "overpass_attempts": route_overpass_attempts,
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "destination": {"lat": dest_lat, "lon": dest_lon},
            "_meta": {
                "crs": WGS84,
                "type": "line",
                "tool": "vec_shortest_path",
            },
        })
    except Exception as exc:
        log.exception("vec_shortest_path failed")
        return _json({"error": str(exc), "tool": "vec_shortest_path"})


@mcp.tool()
def poi_search_osm(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    amenity_types: str,
    output_path: str = "",
    max_results: int = 100,
    unlimited_results: bool = False,
) -> str:
    """
    Search OpenStreetMap amenity POIs through the public Overpass API.

    The query requests amenity nodes and ways and converts returned ways to
    their Overpass-provided center point. The output GeoJSON is EPSG:4326.
    """
    try:
        if radius_m <= 0:
            return _json({
                "error": "radius_m must be greater than 0",
                "tool": "poi_search_osm",
            })
        if max_results <= 0:
            return _json({
                "error": "max_results must be greater than 0",
                "tool": "poi_search_osm",
            })

        output_path = _out(output_path, ".geojson")
        amenities = [
            amenity.strip()
            for amenity in amenity_types.split(",")
            if amenity.strip()
        ]

        if not amenities:
            return _json({
                "error": "amenity_types is empty",
                "tool": "poi_search_osm",
            })

        # "building" is a top-level OSM key, not an amenity value:
        # querying node["amenity"="building"] matches nothing. Map it
        # to way/relation["building"] selectors instead.
        selectors: list[str] = []
        for a in amenities:
            r = int(radius_m)
            if a == "building":
                selectors.append(
                    f'  way["building"](around:{r},{center_lat},{center_lon});'
                )
                selectors.append(
                    f'  relation["building"](around:{r},{center_lat},{center_lon});'
                )
            else:
                selectors.append(
                    f'  node["amenity"="{a}"](around:{r},{center_lat},{center_lon});'
                )
                selectors.append(
                    f'  way["amenity"="{a}"](around:{r},{center_lat},{center_lon});'
                )

        output_clause = (
            "out center;"
            if unlimited_results
            else f"out center {int(max_results)};"
        )
        query = (
            '[out:json][timeout:30];\n'
            '(\n'
            + "\n".join(selectors) +
            '\n);\n'
            f'{output_clause}\n'
        )

        # Shared Overpass client: mirror rotation + cooldown + TTL cache +
        # budget cap + stale-cache fallback when all mirrors fail. A 3 km
        # building scan is a heavy query: public mirrors commonly "fail
        # slowly" at peak (the server runs the full [timeout:30] before
        # returning 504), so one slow failure costs ~30s. The caller
        # (flood_skill GIS section) allows 90s for this tool: server
        # [timeout:30], per-attempt <=45s, total budget 85s (room for
        # 2-3 mirror rotations). With cross-process mirror health
        # sharing, calls usually start directly on a healthy mirror.
        data, diag = post_overpass_sync(
            query,
            server_timeout_s=30.0,
            total_budget_s=85.0,
        )
        if diag.get("attempts"):
            log.warning(
                "poi_search_osm: overpass served by %s after %d failed "
                "attempt(s): %s",
                diag.get("endpoint"), len(diag["attempts"]), diag["attempts"],
            )
        elements = data.get("elements", [])

        features: list[dict[str, Any]] = []
        by_type: dict[str, int] = {}
        label_overrides = load_label_overrides()

        for el in elements:
            tags = el.get("tags", {})
            amenity = tags.get("amenity") or (
                "building" if "building" in tags else "unknown"
            )
            if el.get("type") == "node":
                lat, lon = el["lat"], el["lon"]
            elif el.get("type") in ("way", "relation") and "center" in el:
                lat, lon = el["center"]["lat"], el["center"]["lon"]
            else:
                continue

            label = osm_display_label(
                tags,
                feature_type=amenity,
                osm_type=el["type"],
                osm_id=el["id"],
                lat=float(lat),
                lon=float(lon),
                overrides=label_overrides,
            )

            extra = {
                k: v for k, v in tags.items()
                if k not in ("name", "amenity")
            }
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "name": label["label"],
                    "address": label["address"],
                    "label_source": label["label_source"],
                    "amenity": amenity,
                    "osm_id": el["id"],
                    "osm_type": el["type"],
                    **extra,
                },
            })
            by_type[amenity] = by_type.get(amenity, 0) + 1

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        feature_list = [
            {
                "name": feature["properties"]["name"],
                "address": feature["properties"].get("address"),
                "label_source": feature["properties"].get("label_source"),
                "amenity": feature["properties"]["amenity"],
                "lat": feature["geometry"]["coordinates"][1],
                "lon": feature["geometry"]["coordinates"][0],
            }
            for feature in features[:50]
        ]

        return _json({
            "output_path": output_path,
            "total_found": len(features),
            "by_type": by_type,
            "feature_list": feature_list,
            "search_center": {"lat": center_lat, "lon": center_lon},
            "search_radius_m": radius_m,
            "result_limit": None if unlimited_results else int(max_results),
            "results_truncated": (
                False
                if unlimited_results
                else len(features) >= int(max_results)
            ),
            "_meta": {
                "crs": WGS84,
                "type": "point",
                "tool": "poi_search_osm",
                "source": "OpenStreetMap Overpass API",
                # Data-source disclosure: healthy mirror / cache / stale cache fallback
                "overpass_endpoint": diag.get("endpoint"),
                "overpass_stale": bool(diag.get("stale")),
            },
        })
    except OverpassError as exc:
        # All mirrors failed (timeout/5xx) - suggest a smaller radius to lighten the query
        return _json({
            "error": (
                f"Overpass API unavailable after trying all mirrors: {exc}. "
                "Try a smaller radius_m."
            ),
            "tool": "poi_search_osm",
        })
    except Exception as exc:
        log.exception("poi_search_osm failed")
        return _json({"error": str(exc), "tool": "poi_search_osm"})


@mcp.tool()
def vec_difference(
    base_path: str,
    subtract_path: str,
    output_path: str = "",
) -> str:
    """
    Subtract the second vector layer from the first.

    The overlay is performed in a local UTM CRS and the output is written as
    EPSG:4326. Remaining area is reported in square meters/km².
    """
    try:
        output_path = _out(output_path, ".geojson")
        base = _read_with_crs(base_path)
        sub = _read_with_crs(subtract_path)

        if base.empty:
            empty = gpd.GeoDataFrame(geometry=[], crs=WGS84)
            empty.to_file(output_path, driver="GeoJSON")
            return _json({
                "output_path": output_path,
                "remaining_count": 0,
                "removed_count": 0,
                "_meta": {"crs": WGS84, "tool": "vec_difference"},
            })

        if sub.empty:
            result_wgs = _to_wgs84(base)
            result_wgs.to_file(output_path, driver="GeoJSON")
            metric_crs = _metric_crs(base, base_path, "vec_difference")
            total_area_m2 = float(
                base.to_crs(metric_crs).geometry.area.sum()
            )
            return _json({
                "output_path": output_path,
                "remaining_count": len(base),
                "removed_count": 0,
                "remaining_area_km2": round(total_area_m2 / 1e6, 4),
                "_meta": {"crs": WGS84, "tool": "vec_difference"},
            })

        metric_crs = _metric_crs(base, base_path, "vec_difference")
        base_m = base.to_crs(metric_crs)
        sub_m = sub.to_crs(metric_crs)

        original_count = len(base_m)
        result = gpd.overlay(
            base_m,
            sub_m,
            how="difference",
            keep_geom_type=False,
        )
        result = result[
            ~result.geometry.is_empty & result.geometry.notna()
        ].copy()

        total_area_m2 = (
            float(result.geometry.area.sum())
            if not result.empty
            else 0.0
        )
        result_wgs = _to_wgs84(result)
        result_wgs.to_file(output_path, driver="GeoJSON")

        return _json({
            "output_path": output_path,
            "remaining_count": len(result_wgs),
            "removed_count": max(original_count - len(result_wgs), 0),
            "remaining_area_km2": round(total_area_m2 / 1e6, 4),
            "remaining_area_m2": round(total_area_m2, 1),
            "_meta": {
                "crs": WGS84,
                "metric_crs": str(metric_crs),
                "type": "vector",
                "tool": "vec_difference",
            },
        })
    except Exception as exc:
        log.exception("vec_difference failed")
        return _json({"error": str(exc), "tool": "vec_difference"})


@mcp.tool()
def vec_centroid(
    input_path: str,
    output_path: str = "",
) -> str:
    """
    Compute polygon centroid point(s).

    CRS must be defined by the input data. The function does not guess a CRS.
    Centroids are calculated in a local UTM CRS and returned as EPSG:4326.
    """
    try:
        output_path = _out(output_path, ".geojson")
        gdf = _read_with_crs(input_path)

        if gdf.empty:
            return _json({
                "output_path": output_path,
                "centroids": [],
                "_meta": {"crs": WGS84, "tool": "vec_centroid"},
            })

        metric_crs = _metric_crs(gdf, input_path, "vec_centroid")
        gdf_m = gdf.to_crs(metric_crs)
        cents = gdf_m.copy()
        cents["geometry"] = gdf_m.geometry.centroid
        cents = cents.to_crs(WGS84)
        cents.to_file(output_path, driver="GeoJSON")

        centroids = [
            {
                "feature_index": str(i),
                "lat": round(row.geometry.y, 6),
                "lon": round(row.geometry.x, 6),
            }
            for i, row in cents.iterrows()
        ]

        return _json({
            "output_path": output_path,
            "centroid_count": len(centroids),
            "centroids": centroids,
            "primary_centroid": centroids[0] if centroids else None,
            "_meta": {
                "crs": WGS84,
                "metric_crs": str(metric_crs),
                "type": "point",
                "tool": "vec_centroid",
            },
        })
    except Exception as exc:
        log.exception("vec_centroid failed")
        return _json({"error": str(exc), "tool": "vec_centroid"})


if __name__ == "__main__":
    mcp.run()
