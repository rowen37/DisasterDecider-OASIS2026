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
