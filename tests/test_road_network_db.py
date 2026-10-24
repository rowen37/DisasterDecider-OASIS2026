import json
import sys
from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_servers"))

from mcp_servers import road_network_store
from mcp_servers.gis_vector_server import vec_shortest_path

# Covers the OD pair used below with room for a 4km half-width corridor.
REGION_BBOX = [-95.30, 29.40, -95.10, 29.60]
REGION_NAME = "test_region"


@pytest.fixture(autouse=True)
def _clear_graph_caches():
    """Drop in-process graph caches so each test sees its own tmp store.

    gis_vector_server imports the store module bare (script-style) while
    this file imports it as a package member, so both instances exist;
    clear whichever are loaded.
    """
    road_network_store.clear_graph_cache()
    try:
        import road_network_store as bare_store

        bare_store.clear_graph_cache()
    except ImportError:
        pass
    yield
    road_network_store.clear_graph_cache()
    try:
        import road_network_store as bare_store

        bare_store.clear_graph_cache()
    except ImportError:
        pass


def _write_store(db_dir: Path, graph: nx.MultiDiGraph, mode: str = "drive") -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    graphml = db_dir / f"{REGION_NAME}__{mode}.graphml"
    ox.save_graphml(graph, graphml)
    (db_dir / f"{REGION_NAME}.json").write_text(
        json.dumps({
            "schema_version": 1,
            "region": REGION_NAME,
            "bbox": REGION_BBOX,
            "modes": {
                mode: {
                    "graphml": graphml.name,
                    "node_count": graph.number_of_nodes(),
                    "edge_count": graph.number_of_edges(),
                    "snapshot_date": "2026-10-22",
                }
            },
        }),
        encoding="utf-8",
    )
    return db_dir


def _chain_graph() -> nx.MultiDiGraph:
    """Three connected nodes spanning the OD pair (-95.200 -> -95.190)."""
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(0, x=-95.200, y=29.500)
    graph.add_node(1, x=-95.195, y=29.500)
    graph.add_node(2, x=-95.190, y=29.500)
    for u, v in ((0, 1), (1, 2), (2, 1), (1, 0)):
        graph.add_edge(
            u,
            v,
            length=970.0 / 2,
            geometry=LineString([
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]),
        )
    return graph


def test_covering_region_requires_full_coverage_and_mode(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "road_db", _chain_graph())

    hit = road_network_store.covering_region(
        (-95.21, 29.49, -95.19, 29.51), "drive", db_dir=db_dir
    )
    assert hit is not None
    assert hit["region"] == REGION_NAME
    assert hit["graphml_path"].name == f"{REGION_NAME}__drive.graphml"

    # mode not stored in the region
    assert road_network_store.covering_region(
        (-95.21, 29.49, -95.19, 29.51), "walk", db_dir=db_dir
    ) is None
    # bbox sticking out of the region
    assert road_network_store.covering_region(
        (-95.40, 29.49, -95.19, 29.51), "drive", db_dir=db_dir
    ) is None
    # missing directory
    assert road_network_store.covering_region(
        (-95.21, 29.49, -95.19, 29.51), "drive", db_dir=tmp_path / "nope"
    ) is None
    # kill switch
    monkeypatch.setenv("OSM_ROAD_DB_DISABLED", "1")
    assert road_network_store.covering_region(
        (-95.21, 29.49, -95.19, 29.51), "drive", db_dir=db_dir
    ) is None


def test_shortest_path_served_from_local_db(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "road_db", _chain_graph())
    monkeypatch.setenv("OSM_ROAD_DB_DIR", str(db_dir))

    downloads = []

    def fail_download(polygon, **kwargs):
        downloads.append(polygon)
        raise AssertionError("live download must not run when store covers route")

    monkeypatch.setattr(ox, "graph_from_polygon", fail_download)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        search_radius_km=2.0,
        expanded_search_radius_km=4.0,
        output_path=str(tmp_path / "local-route.geojson"),
    ))

    assert downloads == []
    assert result.get("error") is None
    assert result["graph_source"] == "local_road_db"
    assert result["road_db_region"] == REGION_NAME
    assert result["overpass_endpoint"] is None
    assert result["overpass_attempts"] == []
    assert result["route_avoids_flood"] is False
    assert 500 < result["path_length_m"] < 1500
    assert Path(result["output_path"]).is_file()


def test_shortest_path_falls_back_to_live_when_uncovered(tmp_path, monkeypatch):
    monkeypatch.setenv("OSM_ROAD_DB_DIR", str(tmp_path / "empty_road_db"))

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(0, x=-95.200, y=29.500)
    graph.add_node(1, x=-95.190, y=29.500)
    graph.add_edge(0, 1, length=970.0)
    graph.add_edge(1, 0, length=970.0)

    downloads = []

    def fake_graph_from_polygon(polygon, **kwargs):
        downloads.append(polygon)
        return graph.copy()

    def fake_nearest_nodes(candidate, lon, lat):
        return min(
            candidate.nodes,
            key=lambda node: abs(candidate.nodes[node]["x"] - lon),
        )

    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)
    monkeypatch.setattr(ox.distance, "nearest_nodes", fake_nearest_nodes)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        output_path=str(tmp_path / "live-route.geojson"),
    ))

    assert len(downloads) == 1
    assert result.get("error") is None
    assert result["graph_source"] == "overpass_live"
    assert result["road_db_region"] is None
    assert result["overpass_endpoint"] is not None


def test_shortest_path_falls_back_when_local_db_has_no_route(
    tmp_path, monkeypatch
):
    # The stored region covers the corridor but its graph has no edge
    # between the OD pair, so the local attempt must yield to live download.
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    graph.add_node(0, x=-95.200, y=29.500)
    graph.add_node(2, x=-95.190, y=29.500)
    db_dir = _write_store(tmp_path / "road_db", graph)
    monkeypatch.setenv("OSM_ROAD_DB_DIR", str(db_dir))

    live = nx.MultiDiGraph()
    live.graph["crs"] = "EPSG:4326"
    live.add_node(0, x=-95.200, y=29.500)
    live.add_node(1, x=-95.190, y=29.500)
    live.add_edge(0, 1, length=970.0)
    live.add_edge(1, 0, length=970.0)

    downloads = []

    def fake_graph_from_polygon(polygon, **kwargs):
        downloads.append(polygon)
        return live.copy()

    def fake_nearest_nodes(candidate, lon, lat):
        return min(
            candidate.nodes,
            key=lambda node: abs(candidate.nodes[node]["x"] - lon),
        )

    monkeypatch.setattr(ox, "graph_from_polygon", fake_graph_from_polygon)
    monkeypatch.setattr(ox.distance, "nearest_nodes", fake_nearest_nodes)

    result = json.loads(vec_shortest_path(
        origin_lat=29.500,
        origin_lon=-95.200,
        dest_lat=29.500,
        dest_lon=-95.190,
        output_path=str(tmp_path / "fallback-route.geojson"),
    ))

    assert len(downloads) == 1
    assert result.get("error") is None
    assert result["graph_source"] == "overpass_live"
