import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_servers"))

from mcp_servers import poi_store
from mcp_servers import osm_infrastructure
from mcp_servers import resource_manager

# Region big enough to contain a 50km-radius query circle at the test
# point (resource discovery defaults to radius_km=50; a query is served
# only when the whole circle fits inside the stored bbox).
REGION_BBOX = [-98.40, 29.70, -97.10, 30.85]
REGION_NAME = "test_poi_region"
CENTER = (30.25, -97.75)

NEAR_HOSPITAL = {
    "type": "node", "id": 101, "lat": 30.25, "lon": -97.74,
    "tags": {"amenity": "hospital", "name": "Near Hospital", "beds": "100"},
}
NEAR_SHELTER = {
    "type": "node", "id": 102, "lat": 30.26, "lon": -97.73,
    "tags": {"amenity": "shelter", "name": "Near Shelter"},
}
FAR_HOSPITAL = {
    "type": "node", "id": 103, "lat": 30.80, "lon": -97.20,
    "tags": {"amenity": "hospital", "name": "Far Hospital"},
}
BRIDGE_WAY = {
    "type": "way", "id": 555, "lat": 30.25, "lon": -97.75,
    "tags": {"highway": "primary", "name": "Test Rd", "bridge": "yes"},
}
# Crosses the query center: first vertex far outside a small radius, a
# middle vertex inside - kept only by geometry-intersection filtering.
CROSSING_CREEK = {
    "type": "way", "id": 777, "lat": 30.40, "lon": -98.00,
    "geom": [[-98.00, 30.40], [-97.85, 30.30], [-97.755, 30.2505], [-97.60, 30.20]],
    "tags": {"waterway": "creek", "name": "Test Creek"},
}
FAR_RIVER = {
    "type": "way", "id": 778, "lat": 30.80, "lon": -97.20,
    "geom": [[-97.20, 30.80], [-97.19, 30.79]],
    "tags": {"waterway": "river", "name": "Far River"},
}
# Building centers: one inside the test flood polygon, one outside it but
# inside the stored region.
BLDG_IN = {
    "type": "way", "id": 900, "lat": 30.25, "lon": -97.75,
    "tags": {"building": "yes"},
}
BLDG_OUT = {
    "type": "way", "id": 901, "lat": 30.25, "lon": -97.70,
    "tags": {"building": "yes"},
}


@pytest.fixture(autouse=True)
def _clear_element_caches():
    """Drop in-process element caches so each test sees its own tmp store.

    The MCP modules import the store module bare (script-style) while this
    file imports it as a package member; clear both instances.
    """
    poi_store.clear_element_cache()
    try:
        import poi_store as bare_store

        bare_store.clear_element_cache()
    except ImportError:
        pass
    yield
    poi_store.clear_element_cache()
    try:
        import poi_store as bare_store

        bare_store.clear_element_cache()
    except ImportError:
        pass


def _write_store(db_dir: Path, layers: list[str]) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    meta_layers = {}
    for layer, elements in (
        (poi_store.FACILITIES_LAYER, [NEAR_HOSPITAL, NEAR_SHELTER, FAR_HOSPITAL]),
        (poi_store.ROADS_LAYER, [BRIDGE_WAY]),
        (poi_store.WATERWAYS_LAYER, [CROSSING_CREEK, FAR_RIVER]),
        (poi_store.BUILDINGS_LAYER, [BLDG_IN, BLDG_OUT]),
    ):
        if layer not in layers:
            continue
        file_name = f"{REGION_NAME}__{layer}.json"
        (db_dir / file_name).write_text(
            json.dumps({"elements": elements}), encoding="utf-8"
        )
        meta_layers[layer] = {
            "file": file_name,
            "count": len(elements),
            "snapshot_date": "2026-10-22",
        }
    (db_dir / f"{REGION_NAME}.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "poi",
            "region": REGION_NAME,
            "bbox": REGION_BBOX,
            "layers": meta_layers,
        }),
        encoding="utf-8",
    )
    return db_dir


def test_covering_region_layer_and_circle_bbox():
    # 50km circle around the center must sit inside the region bbox
    w, s, e, n = poi_store.circle_bbox(CENTER[0], CENTER[1], 50_000)
    assert s > REGION_BBOX[1] and n < REGION_BBOX[3]
    assert w > REGION_BBOX[0] and e < REGION_BBOX[2]

    db_dir = _write_store(Path("/tmp/test_poi_store_coverage"), ["facilities"])
    try:
        hit = poi_store.covering_region(
            poi_store.circle_bbox(CENTER[0], CENTER[1], 50_000),
            poi_store.FACILITIES_LAYER,
            db_dir=db_dir,
        )
        assert hit is not None
        assert hit["region"] == REGION_NAME

        # layer not stored
        assert poi_store.covering_region(
            poi_store.circle_bbox(CENTER[0], CENTER[1], 50_000),
            poi_store.ROADS_LAYER,
            db_dir=db_dir,
        ) is None
        # circle sticking out of the region is not covered
        outside = poi_store.circle_bbox(30.80, -97.15, 50_000)
        assert poi_store.covering_region(
            outside, poi_store.FACILITIES_LAYER, db_dir=db_dir
        ) is None
    finally:
        import shutil

        shutil.rmtree(db_dir, ignore_errors=True)


def test_elements_within_radius_filters_far_elements(tmp_path):
    _write_store(tmp_path / "poi_db", ["facilities"])
    entry = poi_store.covering_region(
        poi_store.circle_bbox(CENTER[0], CENTER[1], 50_000),
        poi_store.FACILITIES_LAYER,
        db_dir=tmp_path / "poi_db",
    )
    elements = poi_store.elements_within_radius(
        poi_store.load_elements(entry), CENTER[0], CENTER[1], 50_000
    )
    ids = {el["id"] for el in elements}
    assert ids == {NEAR_HOSPITAL["id"], NEAR_SHELTER["id"]}


def test_resolve_center_prefers_node_coords_and_center_fallback():
    node = poi_store.resolve_center({"type": "node", "id": 1, "lat": 1.0, "lon": 2.0})
    assert node["lat"] == 1.0 and node["lon"] == 2.0
    way = poi_store.resolve_center({
        "type": "way", "id": 2,
        "center": {"lat": 3.0, "lon": 4.0}, "tags": {},
    })
    assert way["lat"] == 3.0 and way["lon"] == 4.0
    assert poi_store.resolve_center({"type": "way", "id": 3}) is None


@pytest.mark.asyncio
async def test_get_available_resources_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["facilities"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    async def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers circle")

    monkeypatch.setattr(resource_manager, "post_overpass_async", fail_overpass)

    result = json.loads(await resource_manager.get_available_resources(
        latitude=CENTER[0], longitude=CENTER[1], radius_km=50
    ))

    assert result["status"] == "ok"
    assert result["facility_count"] == 2  # far hospital filtered out
    names = " ".join(
        f.get("name", "") for f in result["facilities"]
    ) if "facilities" in result else ""
    endpoint = result["metadata"]["overpass_endpoint"]
    assert endpoint.startswith("local-poi-store")
    assert names == "" or "Near Hospital" in json.dumps(result)


@pytest.mark.asyncio
async def test_critical_infrastructure_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["facilities"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    async def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers circle")

    monkeypatch.setattr(osm_infrastructure, "post_overpass_async", fail_overpass)

    result = json.loads(await osm_infrastructure.get_critical_infrastructure(
        latitude=CENTER[0], longitude=CENTER[1], radius_km=50
    ))

    assert result["status"] == "ok"
    facilities = result["metadata"]["facilities"]
    assert {f["type"] for f in facilities} == {"hospital", "shelter"}
    assert result["metadata"]["overpass_endpoint"].startswith("local-poi-store")


@pytest.mark.asyncio
async def test_road_status_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["roads"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    async def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers circle")

    monkeypatch.setattr(osm_infrastructure, "post_overpass_async", fail_overpass)

    result = json.loads(await osm_infrastructure.get_road_status(
        latitude=CENTER[0], longitude=CENTER[1], radius_km=50
    ))

    assert result["status"] == "ok"
    assert result["measurements"]["road_count"]["value"] == 1
    assert result["measurements"]["bridge_count"]["value"] == 1
    assert result["metadata"]["overpass_endpoint"].startswith("local-poi-store")


@pytest.mark.asyncio
async def test_critical_infrastructure_falls_back_when_uncovered(
    tmp_path, monkeypatch
):
    # Store exists but the 50km circle around this edge point escapes the
    # region bbox -> live path must run (fake client returns one element).
    db_dir = _write_store(tmp_path / "poi_db", ["facilities"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    async def fake_overpass(query, **kwargs):
        assert "around:" in query
        return (
            {"elements": [{
                "type": "node", "id": 9,
                "lat": 30.80, "lon": -97.15,
                "tags": {"amenity": "hospital", "name": "Edge Hospital"},
            }]},
            {"endpoint": "fake-mirror", "cache_hit": False, "stale": False},
        )

    monkeypatch.setattr(osm_infrastructure, "post_overpass_async", fake_overpass)

    result = json.loads(await osm_infrastructure.get_critical_infrastructure(
        latitude=30.80, longitude=-97.15, radius_km=50
    ))

    assert result["status"] == "ok"
    assert result["metadata"]["overpass_endpoint"] == "fake-mirror"
    assert result["measurements"]["facility_count"]["value"] == 1


def test_waterways_within_radius_matches_geometry_intersection(tmp_path):
    # The creek's representative point (first vertex) is ~19km away, but a
    # middle vertex sits ~0.5km from the center: geometry-intersection
    # filtering must keep it, a center-point filter would drop it.
    kept = poi_store.waterways_within_radius(
        [CROSSING_CREEK, FAR_RIVER], CENTER[0], CENTER[1], 2000
    )
    assert [el["id"] for el in kept] == [CROSSING_CREEK["id"]]


@pytest.mark.asyncio
async def test_get_waterway_network_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["waterways"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    async def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers circle")

    monkeypatch.setattr(osm_infrastructure, "post_overpass_async", fail_overpass)

    result = json.loads(await osm_infrastructure.get_waterway_network(
        latitude=CENTER[0], longitude=CENTER[1], radius_km=5
    ))

    assert result["status"] == "ok"
    assert result["waterway_count"] == 1
    creek = result["waterways"][0]
    assert creek["name"] == "Test Creek"
    assert creek["waterway"] == "creek"
    assert len(creek["coordinates"]) == 4
    assert result["metadata"]["overpass_endpoint"].startswith("local-poi-store")


@pytest.mark.asyncio
async def test_poi_search_osm_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["facilities"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    import gis_vector_server as gis

    def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers circle")

    monkeypatch.setattr(gis, "post_overpass_sync", fail_overpass)

    payload = json.loads(gis.poi_search_osm(
        center_lat=CENTER[0],
        center_lon=CENTER[1],
        radius_m=3000,
        amenity_types="hospital,shelter",
        output_path=str(tmp_path / "poi.geojson"),
    ))

    assert payload.get("error") is None
    assert payload["total_found"] == 2  # far hospital outside the radius
    assert payload["by_type"] == {"hospital": 1, "shelter": 1}
    names = [f["name"] for f in payload["feature_list"]]
    assert "Near Hospital" in names and "Near Shelter" in names
    assert payload["_meta"]["overpass_endpoint"].startswith("local-poi-store")
    assert Path(payload["output_path"]).is_file()


@pytest.mark.asyncio
async def test_poi_search_osm_unstored_amenity_falls_back(tmp_path, monkeypatch):
    # "building" is not in the facilities layer: even with a covering
    # store present, the query must go to live Overpass.
    db_dir = _write_store(tmp_path / "poi_db", ["facilities"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    import gis_vector_server as gis

    captured = {}

    def fake_overpass(query, **kwargs):
        captured["query"] = query
        return {"elements": [{
            "type": "way", "id": 42,
            "center": {"lat": CENTER[0], "lon": CENTER[1]},
            "tags": {"building": "yes"},
        }]}, {"endpoint": "fake-mirror", "attempts": [], "cache_hit": False}

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)

    payload = json.loads(gis.poi_search_osm(
        center_lat=CENTER[0],
        center_lon=CENTER[1],
        radius_m=1000,
        amenity_types="building",
        output_path=str(tmp_path / "buildings.geojson"),
    ))

    assert payload.get("error") is None
    assert 'way["building"]' in captured["query"]
    assert payload["_meta"]["overpass_endpoint"] == "fake-mirror"
    assert payload["total_found"] == 1


def _flood_polygon_path(tmp_path: Path) -> str:
    """Small square around (30.25, -97.75): contains BLDG_IN, not BLDG_OUT."""
    path = tmp_path / "flood.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-97.76, 30.24], [-97.74, 30.24],
                    [-97.74, 30.26], [-97.76, 30.26],
                    [-97.76, 30.24],
                ]],
            },
        }],
    }), encoding="utf-8")
    return str(path)


def test_polygon_building_search_served_from_store(tmp_path, monkeypatch):
    db_dir = _write_store(tmp_path / "poi_db", ["buildings"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    import gis_vector_server as gis

    def fail_overpass(*args, **kwargs):
        raise AssertionError("live Overpass must not run when store covers polygon")

    monkeypatch.setattr(gis, "post_overpass_sync", fail_overpass)

    payload = json.loads(gis.poi_search_osm(
        center_lat=CENTER[0],
        center_lon=CENTER[1],
        radius_m=3000,
        amenity_types="building",
        output_path=str(tmp_path / "buildings.geojson"),
        unlimited_results=True,
        query_polygon_path=_flood_polygon_path(tmp_path),
    ))

    assert payload.get("error") is None
    assert payload["query_mode"] == "flood_polygon_tiles"
    # only the building whose center lies inside the flood polygon counts
    assert payload["total_found"] == 1
    assert payload["by_type"] == {"building": 1}
    assert payload["_meta"]["overpass_endpoint"].startswith("local-poi-store")
    assert Path(payload["output_path"]).is_file()


def test_polygon_building_search_falls_back_when_uncovered(
    tmp_path, monkeypatch
):
    # Polygon bbox escapes the stored region -> live tiled path (fake
    # client returns one building per tile).
    db_dir = _write_store(tmp_path / "poi_db", ["buildings"])
    monkeypatch.setenv("OSM_POI_DB_DIR", str(db_dir))

    import gis_vector_server as gis

    queries = []

    def fake_overpass(query, **kwargs):
        queries.append(query)
        return {"elements": [{
            "type": "way", "id": 42,
            "center": {"lat": 30.25, "lon": -97.75},
            "tags": {"building": "yes"},
        }]}, {"endpoint": "fake-mirror", "attempts": [], "cache_hit": False}

    monkeypatch.setattr(gis, "post_overpass_sync", fake_overpass)

    flood = tmp_path / "edge_flood.geojson"
    flood.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-97.30, 30.86], [-97.28, 30.86],
                    [-97.28, 30.88], [-97.30, 30.88],
                    [-97.30, 30.86],
                ]],
            },
        }],
    }), encoding="utf-8")

    payload = json.loads(gis.poi_search_osm(
        center_lat=30.87,
        center_lon=-97.29,
        radius_m=3000,
        amenity_types="building",
        output_path=str(tmp_path / "buildings.geojson"),
        unlimited_results=True,
        query_polygon_path=str(flood),
    ))

    assert payload.get("error") is None
    assert queries, "live Overpass must run when the polygon is not covered"
    assert payload["_meta"]["overpass_endpoint"] in {"fake-mirror", "polygon-tiles"}
