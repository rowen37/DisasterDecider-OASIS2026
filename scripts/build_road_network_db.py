#!/usr/bin/env python3
"""Build the local OSM road-network store used by vec_shortest_path.

Downloads a region's road network once, tiled into small Overpass queries
(a full-region query would time out on public mirrors), merges the tiles,
and writes the GraphML store consumed by mcp_servers/road_network_store.py.
After this, routing inside the region needs no network at all.

Usage (network required):
    uv run python scripts/build_road_network_db.py \
        --bbox -95.85 29.45 -94.90 30.05 --region-name houston_metro

    # add another mode to an existing region later:
    uv run python scripts/build_road_network_db.py \
        --bbox -95.85 29.45 -94.90 30.05 --region-name houston_metro --modes walk

The build is resumable: each tile is staged as its own GraphML under
<out-dir>/staging/ and completed tiles are skipped on re-run, so a mirror
outage mid-build only costs the unfinished tiles. Delete a region's
.graphml/.json to force a full rebuild.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_servers"))

from overpass_client import post_overpass_sync  # noqa: E402

# Politeness pause between tile downloads; the shared client's mirror
# rotation/cooldown handles the actual back-off on failures.
INTER_TILE_PAUSE_S = 1.0
# Public mirrors regularly have all-hands overload windows (every mirror
# returning 504/502 at once). Waiting out the shared client's cooldown
# and retrying a tile in a later round recovers without operator input.
DEFAULT_TILE_RETRIES = 4
TILE_RETRY_WAIT_S = 90.0
DEFAULT_MODES = "drive"
VALID_MODES = {"drive", "walk", "bike"}


def _tile_grid(
    bbox: tuple[float, float, float, float],
    tile_km: float,
    overlap_km: float,
) -> Iterator[tuple[int, int, tuple[float, float, float, float]]]:
    """Cover a bbox (west, south, east, north) with metric tiles.

    Yields ``(col, row, tile_bbox)`` in Overpass order (west, south,
    east, north). Neighboring tiles overlap by ``overlap_km`` so ways
    crossing a tile border are fully contained in at least one tile and
    the merged graph keeps a consistent border topology.
    """
    west, south, east, north = bbox
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians((north + south) / 2.0))
    cols = max(1, math.ceil((east - west) * km_per_deg_lon / tile_km))
    rows = max(1, math.ceil((north - south) * km_per_deg_lat / tile_km))
    dlon = (east - west) / cols
    dlat = (north - south) / rows
    ovl_lon = overlap_km / km_per_deg_lon
    ovl_lat = overlap_km / km_per_deg_lat
    for col in range(cols):
        for row in range(rows):
            tile = (
                max(-180.0, west + col * dlon - ovl_lon),
                max(-90.0, south + row * dlat - ovl_lat),
                min(180.0, west + (col + 1) * dlon + ovl_lon),
                min(90.0, south + (row + 1) * dlat + ovl_lat),
            )
            yield col, row, tile


def _download_tile(
    tile_bbox: tuple[float, float, float, float],
    mode: str,
    server_timeout_s: float,
) -> object:
    """Download one tile's graph via the project's rotating Overpass client
    (same bounded-mirror substitution as scripts/build_demo_road_cache.py)."""
    import osmnx as ox
    from osmnx import _overpass as ox_overpass

    def _rotating_request(data):
        payload, _diagnostics = post_overpass_sync(
            str(data.get("data") or ""),
            server_timeout_s=server_timeout_s,
            total_budget_s=server_timeout_s + 60.0,
            ttl_s=0,
        )
        return payload

    previous = ox_overpass._overpass_request
    previous_timeout = ox.settings.requests_timeout
    previous_rate_limit = ox.settings.overpass_rate_limit
    ox_overpass._overpass_request = _rotating_request
    try:
        ox.settings.requests_timeout = math.ceil(server_timeout_s)
        ox.settings.overpass_rate_limit = False
        try:
            # osmnx 2.x: bbox = (left, bottom, right, top)
            return ox.graph_from_bbox(
                bbox=tile_bbox,
                network_type=mode,
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,
            )
        except TypeError:  # osmnx 1.x: bbox = (north, south, east, west)
            west, south, east, north = tile_bbox
            return ox.graph_from_bbox(
                north, south, east, west,
                network_type=mode,
                simplify=True,
                retain_all=True,
                truncate_by_edge=True,
            )
    finally:
        ox_overpass._overpass_request = previous
        ox.settings.requests_timeout = previous_timeout
        ox.settings.overpass_rate_limit = previous_rate_limit


def build(
    bbox: tuple[float, float, float, float],
    region_name: str,
    modes: list[str],
    out_dir: Path,
    tile_km: float,
    overlap_km: float,
    server_timeout_s: float,
    tile_retries: int = DEFAULT_TILE_RETRIES,
) -> None:
    import networkx as nx
    import osmnx as ox

    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise SystemExit(f"invalid bbox (west south east north): {bbox}")

    tiles = list(_tile_grid(bbox, tile_km, overlap_km))
    print(
        f"region={region_name} modes={','.join(modes)} "
        f"bbox={bbox} tiles={len(tiles)} (tile {tile_km}km, "
        f"overlap {overlap_km}km)",
        flush=True,
    )

    staging_dir = out_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{region_name}.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta.setdefault("schema_version", 1)
    meta["region"] = region_name
    meta["bbox"] = list(bbox)
    stored_modes = meta.setdefault("modes", {})

    for mode in modes:
        staging_prefix = staging_dir / f"{region_name}__{mode}"
        merged = nx.MultiDiGraph()
        merged.graph["crs"] = "EPSG:4326"
        done = 0
        for index, (col, row, tile_bbox) in enumerate(tiles, start=1):
            tile_path = staging_prefix / f"{col:03d}_{row:03d}.graphml"
            if tile_path.is_file():
                tile_graph = ox.load_graphml(tile_path)
            else:
                print(
                    f"[{mode} {index}/{len(tiles)}] downloading tile "
                    f"col={col} row={row} bbox={tuple(round(v, 4) for v in tile_bbox)}",
                    flush=True,
                )
                for attempt in range(1, tile_retries + 1):
                    try:
                        tile_graph = _download_tile(
                            tile_bbox, mode, server_timeout_s
                        )
                        break
                    except Exception as exc:
                        if attempt >= tile_retries:
                            raise SystemExit(
                                f"tile col={col} row={row} failed after "
                                f"{tile_retries} rounds: {exc}. Re-run the "
                                f"same command to resume from finished tiles."
                            ) from exc
                        print(
                            f"[{mode} {index}/{len(tiles)}] tile col={col} "
                            f"row={row} failed ({exc}); all mirrors likely "
                            f"overloaded - retry {attempt + 1}/{tile_retries} "
                            f"in {TILE_RETRY_WAIT_S:.0f}s",
                            flush=True,
                        )
                        time.sleep(TILE_RETRY_WAIT_S)
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(tile_graph, tile_path)
                time.sleep(INTER_TILE_PAUSE_S)
            merged = nx.compose(merged, tile_graph)
            done = index
            if index % 10 == 0:
                print(
                    f"[{mode}] {index}/{len(tiles)} tiles merged "
                    f"({len(merged.nodes):,} nodes, {len(merged.edges):,} edges)",
                    flush=True,
                )

        # Tiles are simplified per tile, so a way crossing a border may
        # survive with slightly different endpoints in both tiles; compose
        # then keeps parallel duplicates. Harmless for shortest-path work
        # (the cheaper parallel edge wins), so no dedup pass here.
        graphml_path = out_dir / f"{region_name}__{mode}.graphml"
        ox.save_graphml(merged, graphml_path)
        stored_modes[mode] = {
            "graphml": graphml_path.name,
            "node_count": len(merged.nodes),
            "edge_count": len(merged.edges),
            "snapshot_date": date.today().isoformat(),
        }
        print(
            f"[{mode}] saved {graphml_path.name}: "
            f"{len(merged.nodes):,} nodes, {len(merged.edges):,} edges, "
            f"{graphml_path.stat().st_size / 1e6:.1f} MB ({done} tiles)",
            flush=True,
        )

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"index written: {meta_path}")
    print(
        "vec_shortest_path now serves this region from disk; "
        "re-run with the same --region-name to add modes or refresh tiles."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local OSM road-network GraphML store."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="region bounding box in WGS84 degrees",
    )
    parser.add_argument(
        "--region-name",
        default=None,
        help="store region name (default: road_<lat>_<lon> of the SW corner)",
    )
    parser.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help=f"comma-separated travel modes from {sorted(VALID_MODES)} "
             f"(default: {DEFAULT_MODES})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "cache" / "road_network",
        help="store directory (default: cache/road_network)",
    )
    parser.add_argument(
        "--tile-km", type=float, default=10.0, help="tile size in km (default 10)"
    )
    parser.add_argument(
        "--overlap-km",
        type=float,
        default=0.5,
        help="neighbor-tile overlap in km for border topology (default 0.5)",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=90.0,
        help="per-tile Overpass server timeout in seconds (default 90)",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = set(modes) - VALID_MODES
    if invalid:
        raise SystemExit(f"unsupported modes: {sorted(invalid)}")
    if args.tile_km <= 0 or args.overlap_km < 0:
        raise SystemExit("tile-km must be > 0 and overlap-km >= 0")

    bbox = tuple(args.bbox)
    if args.region_name is None:
        args.region_name = (
            f"road_{round(bbox[1], 2)}_{round(bbox[0], 2)}".replace("-", "m")
        )

    build(
        bbox=bbox,
        region_name=args.region_name,
        modes=modes,
        out_dir=args.out_dir.resolve(),
        tile_km=args.tile_km,
        overlap_km=args.overlap_km,
        server_timeout_s=args.server_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
