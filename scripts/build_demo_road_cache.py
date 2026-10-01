#!/usr/bin/env python
"""One-shot builder for the real-OSM road-network cache behind the demo rescue route.

Usage (needs network; build once, then the demo reproduces offline):
    .venv/bin/python scripts/build_demo_road_cache.py

Output: cache/demo_road_network.json
    {"nodes": {id: [lat, lon]}, "edges": [[u, v, length_m, geom_or_null], ...]}
where geom is the polyline shape of a simplified OSM edge
([[lon, lat], ...], oriented u->v).

app/demo_fixtures.py's _fx_vec_shortest_path runs Dijkstra on this cached
network when present (same semantics as the real
gis_vector_server.vec_shortest_path: endpoint snapping, flood-edge
cutting, no_path_found fallback, conservative-speed travel time).
Without the cache it falls back to the synthetic grid network, so the
demo runs in every case.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp_servers",
    ),
)

from app.utils import geodesic_km  # noqa: E402
from gis_vector_server import _evacuation_edge_attributes  # noqa: E402

# Demo scenario endpoints: Friendswood assessment target -> Demo Hospital (POI fixture)
ORIGIN = (29.5294, -95.2010)
DEST = (29.506, -95.102)
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "demo_road_network.json",
)

# Road-network corridor covering the endpoints' bounding box, padded ~1.5 km
MARGIN_LAT = 1.5 / 110.54          # deg
MARGIN_LON = 1.5 / 96.9            # deg (near lat 29.5)
NORTH = max(ORIGIN[0], DEST[0]) + MARGIN_LAT
SOUTH = min(ORIGIN[0], DEST[0]) - MARGIN_LAT
EAST = max(ORIGIN[1], DEST[1]) + MARGIN_LON
WEST = min(ORIGIN[1], DEST[1]) - MARGIN_LON


def main() -> None:
    import osmnx as ox

    try:  # osmnx 2.x
        G = ox.graph_from_bbox(
            bbox=(WEST, SOUTH, EAST, NORTH),
            network_type="drive",
            simplify=True,
        )
    except TypeError:  # osmnx 1.x
        G = ox.graph_from_bbox(
            NORTH, SOUTH, EAST, WEST,
            network_type="drive",
            simplify=True,
        )
    # Keep the directed graph: matches the real vec_shortest_path, preserving one-way semantics

    nodes = {
        str(n): [float(d["y"]), float(d["x"])]
        for n, d in G.nodes(data=True)
    }
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is not None:
            coords = [[round(p[0], 6), round(p[1], 6)] for p in geom.coords]
            # Ensure the geometry is oriented u->v
            if [round(coords[0][0], 6), round(coords[0][1], 6)] != [
                round(nodes[str(u)][1], 6), round(nodes[str(u)][0], 6)
            ] and len(coords) > 1:
                if (abs(coords[0][0] - nodes[str(v)][1]) < 1e-6
                        and abs(coords[0][1] - nodes[str(v)][0]) < 1e-6):
                    coords = coords[::-1]
            length_m = float(data.get(
                "length",
                geodesic_km(
                    nodes[str(u)][0], nodes[str(u)][1],
                    nodes[str(v)][0], nodes[str(v)][1],
                ) * 1000.0,
            ))
        else:
            coords = None
            length_m = float(data.get(
                "length",
                geodesic_km(
                    nodes[str(u)][0], nodes[str(u)][1],
                    nodes[str(v)][0], nodes[str(v)][1],
                ) * 1000.0,
            ))
        attrs = _evacuation_edge_attributes(data)
        attrs.update({
            "blocked": False,
            "osm_key": str(key),
            "capacity_assumed": data.get("lanes") is None,
        })
        edges.append([str(u), str(v), round(length_m, 1), coords, attrs])

    payload = {
        "meta": {
            "source": "OpenStreetMap (ODbL) via osmnx, drive network, simplified",
            "bbox": [WEST, SOUTH, EAST, NORTH],
            "origin": ORIGIN,
            "destination": DEST,
        },
        "nodes": nodes,
        "edges": edges,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"saved {OUT}: {len(nodes)} nodes, {len(edges)} edges, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
