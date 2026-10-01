# tests/test_demo_route.py
#
# The demo rescue route is computed over the street network:
#   1. It starts at the assessment target and ends at Demo Hospital
#      (POI fixture feature_list[0]).
#   2. It follows the street network: many vertices, single segments
#      bounded by grid spacing, a realistic detour factor (> the
#      straight-line distance), vertices clearly off the straight line
#      between endpoints.
#   3. Length / travel time are consistent with the polyline geometry
#      (conservative 30 km/h), not hardcoded constants.
#   4. Flood-avoidance contract: Demo Hospital lies inside the demo
#      flood polygon, so the preferred flood-avoiding route yields
#      no_path_found; the pipeline falls back to a non-avoiding route
#      and reports route_avoids_flood=False in stats.
# Run: python -m pytest tests/test_demo_route.py -v

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.demo_fixtures import run_demo_assessment  # noqa: E402

ORIGIN = (29.5294, -95.2010)    # Friendswood assessment target (geocode fixture)
HOSPITAL = (29.506, -95.102)    # Demo Hospital (POI fixture feature_list[0])


def _m(lat1, lon1, lat2, lon2):
    """Equirectangular distance approximation in meters; the error is negligible at this ~10 km scale."""
    x = (lon2 - lon1) * 111_320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    y = (lat2 - lat1) * 110_540.0
    return math.hypot(x, y)


def _perp_dev(v, a, b):
    """Perpendicular distance from vertex v to the a-b line (meters, local planar approximation)."""
    def xy(p):
        return (p[1] * 96_900.0, p[0] * 110_540.0)

    ax, ay = xy(a)
    bx, by = xy(b)
    vx, vy = xy(v)
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((vx - ax) * dx + (vy - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(vx - (ax + t * dx), vy - (ay + t * dy))


@pytest.mark.asyncio
async def test_demo_rescue_route_follows_street_network():
    result, state = await run_demo_assessment()
    gis = state.gis_results
    route_path = gis.get("rescue_route_path")
    assert route_path, "demo 管线应产出救援路线产物"
    assert os.path.exists(route_path)
    with open(route_path, encoding="utf-8") as fh:
        gj = json.load(fh)
    lines = [
        f["geometry"]["coordinates"]
        for f in gj.get("features", [])
        if f.get("geometry", {}).get("type") == "LineString"
    ]
    assert lines, "救援路线产物应为 LineString GeoJSON"
    coords = lines[0]

    # 1) Endpoint alignment: start near the target, end near Demo Hospital
    #    (snap tolerance = 1.2 km > grid half-diagonal ~0.6 km + jitter)
    assert _m(coords[0][1], coords[0][0], *ORIGIN) < 1200, (
        f"路线起点 {coords[0]} 应在评估目标点 {ORIGIN} 附近（沿路吸附）"
    )
    assert _m(coords[-1][1], coords[-1][0], *HOSPITAL) < 1200, (
        f"路线终点 {coords[-1]} 应在 Demo Hospital {HOSPITAL} 附近"
    )

    # 2) Network properties: multi-vertex polyline, bounded segments,
    #    detour factor > 1, clearly off the straight line
    assert len(coords) >= 8, (
        f"路线顶点数 {len(coords)} 过少 —— 疑似仍是直线而非街道折线"
    )
    segs = [
        _m(coords[k][1], coords[k][0], coords[k + 1][1], coords[k + 1][0])
        for k in range(len(coords) - 1)
    ]
    total = sum(segs)
    straight = _m(ORIGIN[0], ORIGIN[1], HOSPITAL[0], HOSPITAL[1])
    assert straight < 12000, "场景尺度应约 10 km（防 fixture 漂移）"
    assert total >= 1.05 * straight, (
        f"路网路线总长 {total:.0f} m 应 > 直线 {straight:.0f} m（绕行系数 ≥ 1.05）"
    )
    assert max(segs) <= 3000, (
        f"最长单段 {max(segs):.0f} m 超过街道网间距量级 —— 存在长直跳段"
    )
    deviations = [_perp_dev(c, ORIGIN, HOSPITAL) for c in coords[1:-1]]
    assert max(deviations) >= 50, "路线应显著偏离起终直线（街道折线特征）"

    # 3) Length / travel time consistent with the polyline geometry
    #    (conservative 30 km/h, same as the real tool)
    stats = gis["stats"]
    assert abs(stats["route_length_km"] * 1000.0 - total) <= max(50.0, 0.01 * total), (
        f"stats.route_length_km={stats['route_length_km']} 与折线长 {total:.0f} m 不一致"
    )
    expect_min = total / (30.0 / 3.6) / 60.0
    assert stats["travel_time_min"] > 0, "行程时间不应为 0（旧 fixture 缺失该字段）"
    assert abs(stats["travel_time_min"] - expect_min) <= 0.5

    # 4) Flood-avoidance contract: hospital inside the flood polygon,
    #    so the preferred route has no path; fall back and disclose
    assert stats.get("route_avoids_flood") is False, (
        "医院位于 demo 洪水多边形内：首选避洪路径应 no_path_found，"
        "回退路线必须以 route_avoids_flood=False 披露"
    )


def test_demo_road_cache_reader_accepts_both_edge_layouts(tmp_path, monkeypatch):
    """The cache builder writes 5-element edges [u, v, length, coords,
    attrs]; older caches have 4. The reader must accept both so a
    rebuilt cache cannot silently degrade the demo rescue route to the
    synthetic grid."""
    from app import demo_fixtures

    payload = {
        "meta": {"source": "test cache"},
        "nodes": {"a": [29.5, -95.20], "b": [29.5, -95.19]},
        "edges": [
            [
                "a",
                "b",
                770.0,
                None,
                {"speed_kph": 50, "capacity_vph": 900, "blocked": False},
            ],
            ["b", "a", 770.0],
        ],
    }
    path = tmp_path / "roads.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(demo_fixtures, "_DEMO_ROAD_CACHE_PATH", str(path))
    monkeypatch.setattr(demo_fixtures, "_demo_road_cache", None)
    monkeypatch.setattr(demo_fixtures, "_demo_road_cache_loaded", False)

    loaded = demo_fixtures._load_demo_road_cache()
    assert loaded is not None
    nodes, adj = loaded
    assert set(nodes) == {"a", "b"}
    assert adj["a"] == [("b", 770.0, None)]
    assert adj["b"] == [("a", 770.0, None)]
