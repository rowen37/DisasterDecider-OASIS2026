import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcp_servers",
))

import gis_vector_server as gis  # noqa: E402


def _building(osm_id: int) -> dict:
    return {
        "type": "way",
        "id": osm_id,
        "center": {"lat": 40.54, "lon": -74.58},
        "tags": {"building": "yes"},
    }


def test_unlimited_poi_search_does_not_emit_overpass_result_cap(
    tmp_path, monkeypatch,
):
    captured = {}
    elements = [_building(osm_id) for osm_id in range(501)]

    def fake_overpass(query, **kwargs):
        captured["query"] = query
        return {"elements": elements}, {"endpoint": "test", "attempts": []}

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)
    output_path = tmp_path / "buildings.geojson"

    payload = json.loads(gis.poi_search_osm(
        center_lat=40.54,
        center_lon=-74.58,
        radius_m=3000,
        amenity_types="building",
        output_path=str(output_path),
        unlimited_results=True,
    ))

    assert "out center;" in captured["query"]
    assert "out center 500;" not in captured["query"]
    assert payload["total_found"] == 501
    assert payload["result_limit"] is None
    assert payload["results_truncated"] is False
    assert len(json.loads(output_path.read_text())["features"]) == 501


def test_bounded_poi_search_keeps_explicit_limit(tmp_path, monkeypatch):
    captured = {}

    def fake_overpass(query, **kwargs):
        captured["query"] = query
        return {"elements": [_building(1)]}, {
            "endpoint": "test",
            "attempts": [],
        }

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)

    payload = json.loads(gis.poi_search_osm(
        center_lat=40.54,
        center_lon=-74.58,
        radius_m=1000,
        amenity_types="hospital",
        output_path=str(tmp_path / "poi.geojson"),
        max_results=10,
    ))

    assert "out center 10;" in captured["query"]
    assert payload["result_limit"] == 10
    assert payload["results_truncated"] is False


def test_polygon_tiled_building_search_deduplicates_osm_ids(
    tmp_path, monkeypatch,
):
    flood_path = tmp_path / "flood.geojson"
    flood_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-74.10, 40.87], [-74.06, 40.87],
                    [-74.06, 40.90], [-74.10, 40.90],
                    [-74.10, 40.87],
                ]],
            },
        }],
    }), encoding="utf-8")
    queries = []

    def fake_overpass(query, **kwargs):
        queries.append(query)
        unique_id = len(queries) + 100
        return {
            "elements": [_building(1), _building(unique_id)],
        }, {
            "endpoint": "test",
            "attempts": [],
            "cache_hit": False,
        }

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)
    output_path = tmp_path / "tiled-buildings.geojson"
    payload = json.loads(gis.poi_search_osm(
        center_lat=40.8823215,
        center_lon=-74.0831971,
        radius_m=3000,
        amenity_types="building",
        output_path=str(output_path),
        unlimited_results=True,
        query_polygon_path=str(flood_path),
        tile_size_km=1.0,
        max_tiles=24,
    ))

    assert len(queries) > 1
    assert all('(around:' not in query for query in queries)
    assert all('way["building"](' in query for query in queries)
    assert payload["query_mode"] == "flood_polygon_tiles"
    assert payload["tile_count"] == len(queries)
    assert payload["total_found"] == len(queries) + 1
    features = json.loads(output_path.read_text())["features"]
    assert len(features) == len(queries) + 1


def test_polygon_tiled_building_search_rejects_partial_results(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        gis,
        "_polygon_query_tiles",
        lambda *_args: ([
            (40.87, -74.10, 40.88, -74.09),
            (40.88, -74.09, 40.89, -74.08),
        ], 1.0),
    )
    calls = 0

    def fake_overpass(query, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise gis.OverpassError("all mirrors failed")
        return {"elements": [_building(1)]}, {
            "endpoint": "test",
            "attempts": [],
        }

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)
    output_path = tmp_path / "partial.geojson"
    payload = json.loads(gis.poi_search_osm(
        center_lat=40.88,
        center_lon=-74.09,
        radius_m=3000,
        amenity_types="building",
        output_path=str(output_path),
        unlimited_results=True,
        query_polygon_path=str(tmp_path / "ignored.geojson"),
    ))

    assert "no partial building count was accepted" in payload["error"]
    assert payload["completed_tiles"] == 1
    assert payload["tile_count"] == 2
    assert payload["failed_tiles"][0]["tile_index"] == 1
    assert not output_path.exists()


def test_polygon_tiled_building_search_subdivides_failed_tile(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        gis,
        "_polygon_query_tiles",
        lambda *_args: ([
            (40.87, -74.10, 40.88, -74.09),
        ], 1.0),
    )
    calls = 0
    stored = []

    def fake_overpass(query, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise gis.OverpassError("parent tile timed out")
        return {
            "elements": [_building(1), _building(100 + calls)],
        }, {
            "endpoint": "test",
            "attempts": [],
            "cache_hit": False,
        }

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)
    monkeypatch.setattr(
        gis,
        "store_overpass_cache",
        lambda query, data: stored.append((query, data)),
    )
    output_path = tmp_path / "subdivided.geojson"
    payload = json.loads(gis.poi_search_osm(
        center_lat=40.88,
        center_lon=-74.09,
        radius_m=3000,
        amenity_types="building",
        output_path=str(output_path),
        unlimited_results=True,
        query_polygon_path=str(tmp_path / "ignored.geojson"),
    ))

    assert payload.get("error") is None
    assert calls == 5
    assert payload["total_found"] == 5
    assert payload["tile_diagnostics"][0]["endpoint"] == (
        "subdivided-tiles"
    )
    assert len(stored) == 1
    assert len(stored[0][1]["elements"]) == 5
