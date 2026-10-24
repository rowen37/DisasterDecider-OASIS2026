#!/usr/bin/env python3
"""Build the local OSM POI/infrastructure store (facilities + road tags).

Companion to build_road_network_db.py: that one stores routable graphs for
vec_shortest_path; this one stores the point/attribute data that
resource_manager.get_available_resources and osm_infrastructure's
get_critical_infrastructure / get_road_status otherwise query live on
every run. After building, those tools answer from disk whenever the
query circle lies inside the region.

Usage (network required):
    uv run python scripts/build_poi_db.py \
        --bbox -98.05 30.10 -97.50 30.60 --region-name austin_metro

Layers:
  - facilities (default on): one cheap query for the whole bbox
    (hospital / fire_station / police / school / shelter /
    ambulance_station / emergency=assembly_point with `out center`).
  - roads (default on): highway way tags + way center. Tag-only output is
    a few tens of MB even for a metro, but a single whole-region query is
    still mirror-fragile, so this layer is tiled like the road-graph
    builder and resumable tile-by-tile.

Re-run with the same --region-name to add missing layers or refresh;
delete the region's files to force a rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_servers"))

from overpass_client import post_overpass_sync  # noqa: E402
from poi_store import (  # noqa: E402
    BUILDINGS_LAYER,
    BUILDING_SELECTOR,
    FACILITIES_LAYER,
    FACILITY_SELECTORS,
    ROADS_LAYER,
    ROAD_SELECTOR,
    WATERWAYS_LAYER,
    WATERWAY_SELECTOR,
    resolve_center as _resolve_center,
)
from build_road_network_db import _tile_grid  # noqa: E402

DEFAULT_LAYERS = (
    f"{FACILITIES_LAYER},{ROADS_LAYER},{WATERWAYS_LAYER},{BUILDINGS_LAYER}"
)
TILE_RETRIES = 4
TILE_RETRY_WAIT_S = 90.0
ROAD_TILE_KM = 25.0
WATERWAY_TILE_KM = 25.0
BUILDING_TILE_KM = 25.0


def _post(query: str, server_timeout_s: float) -> dict[str, Any]:
    payload, _diag = post_overpass_sync(
        query,
        server_timeout_s=server_timeout_s,
        total_budget_s=server_timeout_s + 60.0,
        ttl_s=0,
    )
    return payload


def _download_with_retries(label: str, query: str, timeout: float) -> dict:
    for attempt in range(1, TILE_RETRIES + 1):
        try:
            return _post(query, timeout)
        except Exception as exc:
            if attempt >= TILE_RETRIES:
                raise SystemExit(
                    f"{label} failed after {TILE_RETRIES} rounds: {exc}. "
                    f"Re-run the same command to resume."
                ) from exc
            print(
                f"{label} failed ({exc}); all mirrors likely overloaded - "
                f"retry {attempt + 1}/{TILE_RETRIES} in "
                f"{TILE_RETRY_WAIT_S:.0f}s",
                flush=True,
            )
            time.sleep(TILE_RETRY_WAIT_S)
    raise AssertionError("unreachable")


def _build_facilities(
    bbox: tuple[float, float, float, float],
    region_name: str,
    out_dir: Path,
    server_timeout_s: float,
) -> dict:
    west, south, east, north = bbox
    selectors = "\n".join(
        f"{sel}({south:.5f},{west:.5f},{north:.5f},{east:.5f});"
        for sel in FACILITY_SELECTORS
    )
    query = f"[out:json][timeout:{int(server_timeout_s)}];\n(\n{selectors}\n);\nout center;"
    print(f"[facilities] querying region POIs ({len(FACILITY_SELECTORS)} selectors)", flush=True)
    data = _download_with_retries("[facilities]", query, server_timeout_s)
    elements = [
        record
        for el in data.get("elements", [])
        if (record := _resolve_center(el)) is not None
    ]
    file_path = out_dir / f"{region_name}__{FACILITIES_LAYER}.json"
    file_path.write_text(
        json.dumps({"elements": elements}, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[facilities] saved {file_path.name}: {len(elements)} elements, "
        f"{file_path.stat().st_size / 1e6:.1f} MB",
        flush=True,
    )
    return {
        "file": file_path.name,
        "count": len(elements),
        "snapshot_date": date.today().isoformat(),
    }


def _build_tiled_center_layer(
    layer: str,
    selector: str,
    bbox: tuple[float, float, float, float],
    region_name: str,
    out_dir: Path,
    server_timeout_s: float,
    tile_km: float,
) -> dict:
    """Download a way layer as tags + center points, tiled and resumable.

    Shared by the roads and buildings layers; center-point semantics match
    what the serving tools emit live (`out center`)."""
    tiles = list(_tile_grid(bbox, tile_km, overlap_km=0.0))
    print(
        f"[{layer}] downloading {selector} in {len(tiles)} tiles "
        f"of {tile_km:.0f}km (resumable)",
        flush=True,
    )
    staging_dir = out_dir / "staging" / f"{region_name}__{layer}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict[tuple[Any, Any], dict[str, Any]] = {}
    for index, (col, row, tile_bbox) in enumerate(tiles, start=1):
        tile_path = staging_dir / f"{col:03d}_{row:03d}.json"
        if tile_path.is_file():
            tile_elements = json.loads(
                tile_path.read_text(encoding="utf-8")
            ).get("elements", [])
        else:
            west, south, east, north = tile_bbox
            query = (
                f"[out:json][timeout:{int(server_timeout_s)}];\n"
                f"(\n  {selector}({south:.5f},{west:.5f},"
                f"{north:.5f},{east:.5f});\n);\nout tags center;"
            )
            data = _download_with_retries(
                f"[{layer} {index}/{len(tiles)}]", query, server_timeout_s
            )
            tile_elements = [
                record
                for el in data.get("elements", [])
                if (record := _resolve_center(el)) is not None
            ]
            tile_path.write_text(
                json.dumps({"elements": tile_elements}, ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(1.0)
        for record in tile_elements:
            by_id[(record["type"], record["id"])] = record  # dedupe overlap
        if index % 5 == 0:
            print(
                f"[{layer}] {index}/{len(tiles)} tiles ({len(by_id):,} ways)",
                flush=True,
            )

    elements = sorted(by_id.values(), key=lambda r: (r["type"], r["id"]))
    file_path = out_dir / f"{region_name}__{layer}.json"
    file_path.write_text(
        json.dumps({"elements": elements}, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{layer}] saved {file_path.name}: {len(elements):,} ways, "
        f"{file_path.stat().st_size / 1e6:.1f} MB",
        flush=True,
    )
    return {
        "file": file_path.name,
        "count": len(elements),
        "snapshot_date": date.today().isoformat(),
    }


def _waterway_record(element: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize an `out geom` waterway way to a stored record.

    Keeps the full polyline as ``geom`` ([[lon, lat], ...]; the first
    vertex doubles as the representative point) so serving can apply
    geometry-intersection radius matching.
    """
    geometry = element.get("geometry")
    if not (isinstance(geometry, list) and len(geometry) >= 2):
        return None
    coords: list[list[float]] = []
    for point in geometry:
        if not isinstance(point, dict):
            continue
        lat, lon = point.get("lat"), point.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            coords.append([float(lon), float(lat)])
    if len(coords) < 2:
        return None
    return {
        "type": "way",
        "id": element.get("id"),
        "lat": coords[0][1],
        "lon": coords[0][0],
        "geom": coords,
        "tags": element.get("tags") or {},
    }


def _build_waterways(
    bbox: tuple[float, float, float, float],
    region_name: str,
    out_dir: Path,
    server_timeout_s: float,
    waterway_tile_km: float,
) -> dict:
    tiles = list(_tile_grid(bbox, waterway_tile_km, overlap_km=0.0))
    print(
        f"[waterways] downloading waterway polylines in {len(tiles)} tiles "
        f"of {waterway_tile_km:.0f}km (resumable)",
        flush=True,
    )
    staging_dir = out_dir / "staging" / f"{region_name}__{WATERWAYS_LAYER}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict[Any, dict[str, Any]] = {}
    for index, (col, row, tile_bbox) in enumerate(tiles, start=1):
        tile_path = staging_dir / f"{col:03d}_{row:03d}.json"
        if tile_path.is_file():
            tile_elements = json.loads(
                tile_path.read_text(encoding="utf-8")
            ).get("elements", [])
        else:
            west, south, east, north = tile_bbox
            query = (
                f"[out:json][timeout:{int(server_timeout_s)}];\n"
                f"(\n  {WATERWAY_SELECTOR}({south:.5f},{west:.5f},"
                f"{north:.5f},{east:.5f});\n);\nout geom;"
            )
            data = _download_with_retries(
                f"[waterways {index}/{len(tiles)}]", query, server_timeout_s
            )
            tile_elements = [
                record
                for el in data.get("elements", [])
                if (record := _waterway_record(el)) is not None
            ]
            tile_path.write_text(
                json.dumps({"elements": tile_elements}, ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(1.0)
        for record in tile_elements:
            by_id[record["id"]] = record  # dedupe tile overlap
        if index % 5 == 0:
            print(
                f"[waterways] {index}/{len(tiles)} tiles "
                f"({len(by_id):,} ways)",
                flush=True,
            )

    elements = sorted(by_id.values(), key=lambda r: r["id"])
    file_path = out_dir / f"{region_name}__{WATERWAYS_LAYER}.json"
    file_path.write_text(
        json.dumps({"elements": elements}, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[waterways] saved {file_path.name}: {len(elements):,} ways, "
        f"{file_path.stat().st_size / 1e6:.1f} MB",
        flush=True,
    )
    return {
        "file": file_path.name,
        "count": len(elements),
        "snapshot_date": date.today().isoformat(),
    }


def build(
    bbox: tuple[float, float, float, float],
    region_name: str,
    layers: list[str],
    out_dir: Path,
    server_timeout_s: float,
    road_tile_km: float,
    waterway_tile_km: float,
    building_tile_km: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{region_name}.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta["schema_version"] = 1
    meta["kind"] = "poi"
    meta["region"] = region_name
    meta["bbox"] = list(bbox)
    stored_layers = meta.setdefault("layers", {})

    if FACILITIES_LAYER in layers:
        stored_layers[FACILITIES_LAYER] = _build_facilities(
            bbox, region_name, out_dir, server_timeout_s
        )
    if ROADS_LAYER in layers:
        stored_layers[ROADS_LAYER] = _build_tiled_center_layer(
            ROADS_LAYER, ROAD_SELECTOR,
            bbox, region_name, out_dir, server_timeout_s, road_tile_km,
        )
    if WATERWAYS_LAYER in layers:
        stored_layers[WATERWAYS_LAYER] = _build_waterways(
            bbox, region_name, out_dir, server_timeout_s, waterway_tile_km
        )
    if BUILDINGS_LAYER in layers:
        stored_layers[BUILDINGS_LAYER] = _build_tiled_center_layer(
            BUILDINGS_LAYER, BUILDING_SELECTOR,
            bbox, region_name, out_dir, server_timeout_s, building_tile_km,
        )

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"index written: {meta_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local OSM POI/infrastructure store."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="region bounding box in WGS84 degrees (add margin: queries are "
             "served only when their whole radius circle fits inside)",
    )
    parser.add_argument("--region-name", required=True)
    parser.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help=f"comma-separated layers from {{{DEFAULT_LAYERS}}} (default: both)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "cache" / "poi_store",
        help="store directory (default: cache/poi_store)",
    )
    parser.add_argument(
        "--server-timeout", type=float, default=120.0,
        help="per-query Overpass server timeout in seconds (default 120)",
    )
    parser.add_argument(
        "--road-tile-km", type=float, default=ROAD_TILE_KM,
        help=f"road-layer tile size in km (default {ROAD_TILE_KM:.0f})",
    )
    parser.add_argument(
        "--waterway-tile-km", type=float, default=WATERWAY_TILE_KM,
        help=f"waterway-layer tile size in km (default {WATERWAY_TILE_KM:.0f})",
    )
    parser.add_argument(
        "--building-tile-km", type=float, default=BUILDING_TILE_KM,
        help=f"buildings-layer tile size in km (default {BUILDING_TILE_KM:.0f})",
    )
    args = parser.parse_args()

    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    unknown = set(layers) - {
        FACILITIES_LAYER, ROADS_LAYER, WATERWAYS_LAYER, BUILDINGS_LAYER,
    }
    if unknown:
        raise SystemExit(f"unknown layers: {sorted(unknown)}")

    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise SystemExit(f"invalid bbox (west south east north): {args.bbox}")

    build(
        bbox=tuple(args.bbox),
        region_name=args.region_name,
        layers=layers,
        out_dir=args.out_dir.resolve(),
        server_timeout_s=args.server_timeout,
        road_tile_km=args.road_tile_km,
        waterway_tile_km=args.waterway_tile_km,
        building_tile_km=args.building_tile_km,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
