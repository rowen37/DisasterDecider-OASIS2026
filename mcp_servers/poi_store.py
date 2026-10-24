"""Local OSM POI / infrastructure store for offline facility & road-tag
queries.

Companion to ``road_network_store`` (routing): this store answers the
point/attribute OSM queries that the flood pipeline otherwise issues live
to Overpass on every run —

  - ``facilities`` layer: emergency POIs (hospital / fire_station /
    police / school / shelter / ambulance_station, plus
    emergency=assembly_point) with resolved coordinates. Serves
    ``resource_manager.get_available_resources`` and
    ``osm_infrastructure.get_critical_infrastructure``.
  - ``roads`` layer: highway ways with tags + way center (no geometry).
    Serves ``osm_infrastructure.get_road_status``.

Store layout (built by ``scripts/build_poi_db.py``)::

    <dir>/<region>.json
          {"schema_version": 1, "kind": "poi", "region": ...,
           "bbox": [w, s, e, n],
           "layers": {"facilities": {"file": "<region>__facilities.json",
                                     "count": ..., "snapshot_date": ...},
                      "roads": {...}}}
    <dir>/<region>__facilities.json   {"elements": [{type,id,lat,lon,tags}]}
    <dir>/<region>__roads.json        {"elements": [...]}

Serving semantics: an ``around:{r},{lat},{lon}`` query is answered from
the store when the stored bbox fully contains the query circle; elements
are then filtered by geodesic distance of their resolved center point.
For ways this is an approximation of Overpass's geometry-intersection
match (a way whose center lies outside the radius but crosses the circle
is not returned) - close enough for the counting/evidence use of these
tools, and disclosed via the diagnostics the tools already report.

Environment variables:
  OSM_POI_DB_DIR       store directory, default <project>/cache/poi_store
  OSM_POI_DB_DISABLED  "1"/"true" disables the store entirely
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical selector set shared by the builder and every consumer, so a
# store built by the default command always covers what the tools ask for.
FACILITY_SELECTORS = [
    'nwr["amenity"="hospital"]',
    'nwr["amenity"="fire_station"]',
    'nwr["amenity"="police"]',
    'nwr["amenity"="school"]',
    'nwr["amenity"="shelter"]',
    'nwr["amenity"="ambulance_station"]',
    'nwr["emergency"="assembly_point"]',
]
ROAD_SELECTOR = 'way["highway"]'
WATERWAY_SELECTOR = 'way["waterway"~"^(river|canal|stream)$"]'
BUILDING_SELECTOR = 'way["building"]'

FACILITIES_LAYER = "facilities"
ROADS_LAYER = "roads"
WATERWAYS_LAYER = "waterways"
BUILDINGS_LAYER = "buildings"

EARTH_RADIUS_M = 6371000.0


def poi_db_dir() -> Path:
    override = os.environ.get("OSM_POI_DB_DIR")
    if override:
        return Path(override)
    return PROJECT_ROOT / "cache" / "poi_store"


def poi_db_enabled() -> bool:
    return os.environ.get("OSM_POI_DB_DISABLED", "").strip().lower() not in {
        "1", "true", "yes",
    }


def circle_bbox(
    latitude: float, longitude: float, radius_m: float
) -> tuple[float, float, float, float]:
    """Bounding box (west, south, east, north) of an around-radius query."""
    dlat = math.degrees(radius_m / EARTH_RADIUS_M)
    dlon = math.degrees(
        radius_m / (EARTH_RADIUS_M * math.cos(math.radians(latitude)))
    )
    return (
        max(-180.0, longitude - dlon),
        max(-90.0, latitude - dlat),
        min(180.0, longitude + dlon),
        min(90.0, latitude + dlat),
    )


def _iter_metas(db_dir: Path) -> list[dict[str, Any]]:
    try:
        paths = sorted(db_dir.glob("*.json"))
    except OSError:
        return []
    metas: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        layers = raw.get("layers")
        if raw.get("schema_version") != 1 or raw.get("kind") != "poi":
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not (isinstance(layers, dict) and layers):
            continue
        metas.append(raw)
    return metas


def covering_region(
    bbox: tuple[float, float, float, float],
    layer: str,
    db_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return the store entry whose region fully covers ``bbox`` and has
    ``layer``, or None. Result carries ``region`` and ``file_path``."""
    if not poi_db_enabled():
        return None
    directory = db_dir if db_dir is not None else poi_db_dir()
    query = box(*bbox)
    for meta in _iter_metas(directory):
        entry = meta["layers"].get(layer)
        if not isinstance(entry, dict):
            continue
        if not box(*meta["bbox"]).covers(query):
            continue
        file = directory / str(entry.get("file", ""))
        if not file.is_file():
            continue
        result = dict(entry)
        result["region"] = meta.get("region") or file.stem
        result["file_path"] = file
        return result
    return None


_loaded_lock = threading.Lock()
_loaded_elements: dict[str, list[dict[str, Any]]] = {}  # path -> elements


def clear_element_cache() -> None:
    """Drop the in-process element cache (for tests)."""
    with _loaded_lock:
        _loaded_elements.clear()


def load_elements(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Load a layer's element list once per process (thread-safe LRU of 2)."""
    key = str(Path(entry["file_path"]).resolve())
    with _loaded_lock:
        if key in _loaded_elements:
            elements = _loaded_elements.pop(key)
            _loaded_elements[key] = elements  # refresh recency
            return elements

    raw = json.loads(Path(key).read_text(encoding="utf-8"))
    elements = raw.get("elements")
    if not isinstance(elements, list):
        elements = []
    with _loaded_lock:
        _loaded_elements[key] = elements
        while len(_loaded_elements) > 2:
            _loaded_elements.pop(next(iter(_loaded_elements)))
    return elements


def resolve_center(element: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an Overpass element to ``{type,id,lat,lon,tags}``.

    Nodes carry coordinates at the top level; ways/relations rely on the
    ``out center`` annotation. Elements without any location return None.
    Shared by the builder and the serving tools so live and stored
    elements always have the same shape.
    """
    center = element.get("center") or {}
    lat = element.get("lat")
    lon = element.get("lon")
    if lat is None:
        lat = center.get("lat")
    if lon is None:
        lon = center.get("lon")
    if lat is None or lon is None:
        return None
    return {
        "type": element.get("type"),
        "id": element.get("id"),
        "lat": float(lat),
        "lon": float(lon),
        "tags": element.get("tags") or {},
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def elements_within_radius(
    elements: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    radius_m: float,
) -> list[dict[str, Any]]:
    """Elements whose resolved center lies within radius of the query
    point. Elements without usable coordinates are dropped."""
    keep: list[dict[str, Any]] = []
    for el in elements:
        lat = el.get("lat")
        lon = el.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        if haversine_m(latitude, longitude, float(lat), float(lon)) <= radius_m:
            keep.append(el)
    return keep


def waterways_within_radius(
    elements: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    radius_m: float,
) -> list[dict[str, Any]]:
    """Waterway ways with ANY geometry vertex inside the radius - the same
    geometry-intersection semantics as an Overpass ``around:`` filter, so
    local results match live results (a center-point filter would drop
    ways that merely cross the circle). Records carry ``geom`` as
    ``[[lon, lat], ...]`` from the builder."""
    keep: list[dict[str, Any]] = []
    for el in elements:
        geom = el.get("geom")
        if not isinstance(geom, list):
            continue
        for point in geom:
            if not (isinstance(point, (list, tuple)) and len(point) >= 2):
                continue
            lon, lat = point[0], point[1]
            if not (
                isinstance(lat, (int, float)) and isinstance(lon, (int, float))
            ):
                continue
            if haversine_m(
                latitude, longitude, float(lat), float(lon)
            ) <= radius_m:
                keep.append(el)
                break
    return keep
