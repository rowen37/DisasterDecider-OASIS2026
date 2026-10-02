import json
import sys
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_servers"))
from mcp_servers.gis_vector_server import vec_shortest_path
from mcp_servers.overpass_client import OverpassError


def test_route_corridor_reuses_download_for_non_avoiding_fallback(
    tmp_path, monkeypatch
):
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(0, x=-95.200, y=29.500)
    graph.add_node(1, x=-95.190, y=29.500)
    geometry = LineString([(-95.200, 29.500), (-95.190, 29.500)])
    graph.add_edge(0, 1, length=970.0, geometry=geometry)
    graph.add_edge(1, 0, length=970.0, geometry=geometry)

    downloads = []

    def fake_graph_from_polygon(polygon, **kwargs):
        downloads.append((polygon, kwargs))
        return graph.copy()

    def fake_nearest_nodes(candidate, lon, lat):
        return min(
            candidate.nodes,
            key=lambda node: abs(candidate.nodes[node]["x"] - lon),
        )

    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)
    monkeypatch.setattr(ox.distance, "nearest_nodes", fake_nearest_nodes)

    flood_path = tmp_path / "flood.geojson"
    flood_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-95.201, 29.499], [-95.189, 29.499],
                    [-95.189, 29.501], [-95.201, 29.501],
                    [-95.201, 29.499],
                ]],
            },
        }],
    }), encoding="utf-8")

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        avoid_polygon_path=str(flood_path),
        search_radius_km=2.0,
        expanded_search_radius_km=4.0,
        output_path=str(tmp_path / "route.geojson"),
    ))

    assert len(downloads) == 1
    assert downloads[0][1]["network_type"] == "drive"
    assert result.get("error") is None
    assert result["route_avoids_flood"] is False
    assert result["corridor_half_width_km"] == 2.0
    assert result["download_geometry"] == "origin_destination_corridor"


def test_route_corridor_switches_mirror_after_transport_failure(
    tmp_path, monkeypatch
):
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(0, x=-95.200, y=29.500)
    graph.add_node(1, x=-95.190, y=29.500)
    graph.add_edge(0, 1, length=970.0)
    graph.add_edge(1, 0, length=970.0)

    endpoints_seen = []

    def fake_graph_from_polygon(polygon, **kwargs):
        endpoints_seen.append(ox.settings.overpass_url)
        if len(endpoints_seen) == 1:
            raise ConnectionError("first mirror refused connection")
        return graph.copy()

    def fake_nearest_nodes(candidate, lon, lat):
        return min(
            candidate.nodes,
            key=lambda node: abs(candidate.nodes[node]["x"] - lon),
        )

    monkeypatch.setenv(
        "OSM_OVERPASS_ENDPOINTS",
        "https://mirror-a.test/api/interpreter,"
        "https://mirror-b.test/api/interpreter",
    )
    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)
    monkeypatch.setattr(ox.distance, "nearest_nodes", fake_nearest_nodes)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        search_radius_km=2.0,
        expanded_search_radius_km=4.0,
        output_path=str(tmp_path / "mirror-route.geojson"),
    ))

    assert endpoints_seen == [
        "https://mirror-a.test/api",
        "https://mirror-b.test/api",
    ]
    assert result.get("error") is None
    assert result["overpass_endpoint"] == (
        "https://mirror-b.test/api/interpreter"
    )
    assert [item["status"] for item in result["overpass_attempts"]] == [
        "failed", "downloaded"
    ]


def test_route_corridor_uses_integer_overpass_query_timeout(
    tmp_path, monkeypatch
):
    timeouts_seen = []

    def fake_graph_from_polygon(polygon, **kwargs):
        timeouts_seen.append(ox.settings.requests_timeout)
        raise ConnectionError("stop after observing settings")

    monkeypatch.setenv("OSM_ROUTE_REQUEST_TIMEOUT_SECONDS", "20.0")
    monkeypatch.setenv("OSM_ROUTE_MAX_DOWNLOAD_ATTEMPTS", "1")
    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        output_path=str(tmp_path / "timeout-route.geojson"),
    ))

    assert result["error"] == "overpass_route_download_failed"
    assert timeouts_seen == [20]
    assert isinstance(timeouts_seen[0], int)


def test_route_corridor_does_not_rotate_after_query_rejection(
    tmp_path, monkeypatch
):
    endpoints_seen = []

    def fake_graph_from_polygon(polygon, **kwargs):
        endpoints_seen.append(ox.settings.overpass_url)
        raise OverpassError("Overpass rejected the query: HTTP 400")

    monkeypatch.setenv(
        "OSM_OVERPASS_ENDPOINTS",
        "https://mirror-a.test/api/interpreter,"
        "https://mirror-b.test/api/interpreter",
    )
    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        output_path=str(tmp_path / "rejected-route.geojson"),
    ))

    assert result["error"] == "overpass_route_download_failed"
    assert endpoints_seen == ["https://mirror-a.test/api"]
