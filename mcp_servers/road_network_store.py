"""Local OSM road-network store for offline, instant corridor routing.

The store is a directory (default ``<project>/cache/road_network``) holding
one GraphML per (region, travel-mode) plus a JSON metadata index per
region, produced by ``scripts/build_road_network_db.py``::

    <dir>/houston_metro.json
          {"schema_version": 1, "region": "houston_metro",
           "bbox": [west, south, east, north],
           "modes": {"drive": {"graphml": "houston_metro__drive.graphml",
                               "node_count": ..., "edge_count": ...,
                               "snapshot_date": "2026-10-22"}}}
    <dir>/houston_metro__drive.graphml

``vec_shortest_path`` consults the store before any live Overpass
download: a corridor fully inside a stored region is answered by
truncating the stored graph to the corridor polygon - millisecond-scale
and independent of public-mirror health. Uncovered corridors fall back
to the existing bounded live download unchanged.

Environment variables:
  OSM_ROAD_DB_DIR        store directory, default <project>/cache/road_network
  OSM_ROAD_DB_DISABLED   "1"/"true" disables the store entirely
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Graphs are memory-heavy (a metro drive network is hundreds of MB as a
# networkx object). Keep only the most recent few loaded per process.
_MAX_LOADED_GRAPHS = 2


def road_db_dir() -> Path:
    override = os.environ.get("OSM_ROAD_DB_DIR")
    if override:
        return Path(override)
    return PROJECT_ROOT / "cache" / "road_network"


def road_db_enabled() -> bool:
    return os.environ.get("OSM_ROAD_DB_DISABLED", "").strip().lower() not in {
        "1", "true", "yes",
    }


def _iter_region_metas(db_dir: Path) -> list[dict[str, Any]]:
    """Load every valid region index in the store dir; tolerate junk files."""
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
        modes = raw.get("modes")
        if raw.get("schema_version") != 1:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if not (isinstance(modes, dict) and modes):
            continue
        metas.append(raw)
    return metas


def covering_region(
    bbox: tuple[float, float, float, float],
    mode: str,
    db_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return the store entry covering ``bbox`` (west, south, east, north)
    for ``mode``, or None when no stored region fully covers it.

    The returned dict has ``region`` and ``graphml_path`` keys plus the
    per-mode metadata (node_count, edge_count, snapshot_date).
    """
    if not road_db_enabled():
        return None
    directory = db_dir if db_dir is not None else road_db_dir()
    query = box(*bbox)
    for meta in _iter_region_metas(directory):
        entry = meta["modes"].get(mode)
        if not isinstance(entry, dict):
            continue
        if not box(*meta["bbox"]).covers(query):
            continue
        graphml = directory / str(entry.get("graphml", ""))
        if not graphml.is_file():
            continue
        result = dict(entry)
        result["region"] = meta.get("region") or graphml.stem
        result["graphml_path"] = graphml
        return result
    return None


_loaded_lock = threading.Lock()
_loaded_graphs: dict[str, Any] = {}   # resolved path -> graph, insertion order = age


def clear_graph_cache() -> None:
    """Drop in-process graph cache (for tests)."""
    with _loaded_lock:
        _loaded_graphs.clear()


def load_graph(graphml_path: str | Path) -> Any:
    """Load a stored GraphML once per process (thread-safe LRU of 1-2 graphs).

    Raises on IO/parse errors; callers fall back to live download.
    """
    import osmnx as ox

    key = str(Path(graphml_path).resolve())
    with _loaded_lock:
        if key in _loaded_graphs:
            graph = _loaded_graphs.pop(key)
            _loaded_graphs[key] = graph  # refresh recency
            return graph

    graph = ox.load_graphml(key)
    with _loaded_lock:
        _loaded_graphs[key] = graph
        while len(_loaded_graphs) > _MAX_LOADED_GRAPHS:
            _loaded_graphs.pop(next(iter(_loaded_graphs)))
    return graph


def corridor_subgraph(graph: Any, polygon: Any) -> Any | None:
    """Clip a stored graph to a corridor polygon (keeping boundary-crossing
    edges), or None when the graph has no node inside the polygon."""
    from osmnx import truncate as ox_truncate

    try:
        return ox_truncate.truncate_graph_polygon(
            graph, polygon, truncate_by_edge=True
        )
    except ValueError:
        return None
