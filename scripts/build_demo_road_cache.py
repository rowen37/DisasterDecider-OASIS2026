#!/usr/bin/env python3
"""Build the local OSM road cache used by the offline demo.

Usage (network required):
    uv run python scripts/build_demo_road_cache.py

The generated cache is ignored by Git and can be refreshed when needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "mcp_servers"))

from app.utils import geodesic_km  # noqa: E402
from overpass_client import post_overpass_sync  # noqa: E402


ORIGIN = (29.5294, -95.2010)
DESTINATION = (29.506, -95.102)
DEFAULT_OUTPUT = PROJECT_ROOT / "cache" / "demo_road_network.json"

# Friendswood demo corridor, padded by approximately 1.5 km.
MARGIN_LAT = 1.5 / 110.54
MARGIN_LON = 1.5 / 96.9
NORTH = max(ORIGIN[0], DESTINATION[0]) + MARGIN_LAT
SOUTH = min(ORIGIN[0], DESTINATION[0]) - MARGIN_LAT
EAST = max(ORIGIN[1], DESTINATION[1]) + MARGIN_LON
WEST = min(ORIGIN[1], DESTINATION[1]) - MARGIN_LON


def _edge_geometry(data, source, destination):
    geometry = data.get("geometry")
    if geometry is None:
        return None
    coordinates = [
        [round(float(lon), 6), round(float(lat), 6)]
        for lon, lat in geometry.coords
    ]
    source_xy = [round(source[1], 6), round(source[0], 6)]
    destination_xy = [round(destination[1], 6), round(destination[0], 6)]
    if len(coordinates) > 1 and coordinates[0] != source_xy:
        if coordinates[0] == destination_xy:
            coordinates.reverse()
    return coordinates


def build(output: Path) -> None:
    import osmnx as ox
    from osmnx import _overpass as ox_overpass

    # Use the same bounded, rotating Overpass client as the live MCP instead
    # of OSMnx's single-host requester. This makes the refresh command robust
    # to a public mirror being unavailable and avoids persistent workspace
    # cache state.
    previous_request = ox_overpass._overpass_request

    def _rotating_request(data):
        payload, _diagnostics = post_overpass_sync(
            str(data.get("data") or ""),
            server_timeout_s=60,
            total_budget_s=180,
            cache_dir=Path("/tmp/disaster-demo-road-overpass"),
            ttl_s=0,
        )
        return payload

    ox_overpass._overpass_request = _rotating_request
    try:
        try:
            graph = ox.graph_from_bbox(
                bbox=(WEST, SOUTH, EAST, NORTH),
                network_type="drive",
                simplify=True,
            )
        except TypeError:  # osmnx 1.x compatibility
            graph = ox.graph_from_bbox(
                NORTH,
                SOUTH,
                EAST,
                WEST,
                network_type="drive",
                simplify=True,
            )
    finally:
        ox_overpass._overpass_request = previous_request

    nodes = {
        str(node_id): [float(data["y"]), float(data["x"])]
        for node_id, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    }
    edges = []
    for source_id, destination_id, key, data in graph.edges(keys=True, data=True):
        source = nodes[str(source_id)]
        destination = nodes[str(destination_id)]
        length_m = float(data.get(
            "length",
            geodesic_km(source[0], source[1], destination[0], destination[1])
            * 1000.0,
        ))
        edges.append([
            str(source_id),
            str(destination_id),
            round(length_m, 1),
            _edge_geometry(data, source, destination),
            str(key),
        ])
    edges.sort(key=lambda item: (item[0], item[1], item[4]))

    payload = {
        "schema_version": 1,
        "meta": {
            "source": "OpenStreetMap (ODbL) via OSMnx",
            "network_type": "drive",
            "simplified": True,
            "snapshot_date": date.today().isoformat(),
            "bbox": [WEST, SOUTH, EAST, NORTH],
            "origin": list(ORIGIN),
            "destination": list(DESTINATION),
        },
        "nodes": nodes,
        "edges": edges,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"saved {output}: {len(nodes)} nodes, {len(edges)} edges, "
        f"{output.stat().st_size / 1e6:.1f} MB"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output fixture path",
    )
    args = parser.parse_args()
    build(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
