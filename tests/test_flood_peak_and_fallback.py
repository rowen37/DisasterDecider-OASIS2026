# tests/test_flood_peak_and_fallback.py
#
# Offline unit tests for the 2026-09-26 "peak-stage + fail-loud
# degradation" rework (Lodi NJ 2026-09-13 review):
#
#   1. Historical window queries select the window PEAK as the event
#      stage (not the window-end snapshot) and ship a full hydrograph
#      summary; realtime queries keep the latest value.
#   2. The modeled stage-buffer extent builder produces a disclosed,
#      exceedance-scaled waterway buffer and refuses to model when the
#      gauge never crossed action stage.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.flood_alert import select_stage_observation  # noqa: E402
from app.engine.geometry_engine import GeometryEngine          # noqa: E402


# ------------------------------------------------------------------
# 1. Stage selection: window peak vs latest instantaneous
# ------------------------------------------------------------------

def _values():
    # Hydrograph shaped like the Lodi 2026-09-13 event: rises to a
    # mid-window peak, recedes below action stage by window end.
    return [
        {"value": "2.12", "dateTime": "2026-09-12T20:00:00.000-04:00"},
        {"value": "3.50", "dateTime": "2026-09-13T04:00:00.000-04:00"},
        {"value": "6.79", "dateTime": "2026-09-13T14:15:00.000-04:00"},
        {"value": "5.20", "dateTime": "2026-09-13T17:00:00.000-04:00"},
        {"value": "4.67", "dateTime": "2026-09-13T20:00:00.000-04:00"},
    ]


def test_windowed_query_returns_peak_not_end_snapshot():
    primary, summary, semantics = select_stage_observation(
        _values(), windowed=True
    )
    # The event's stage is its PEAK: 6.79 ft at 14:15, not the
    # receded 4.67 ft window-end value.
    assert float(primary["value"]) == 6.79
    assert semantics == "window_peak"
    assert summary is not None
    assert summary["peak_stage_ft"] == 6.79
    assert summary["peak_time"].startswith("2026-09-13T14:15")
    assert summary["end_stage_ft"] == 4.67
    assert summary["min_stage_ft"] == 2.12
    assert summary["value_count"] == 5


def test_realtime_query_returns_latest_instantaneous():
    primary, summary, semantics = select_stage_observation(
        _values(), windowed=False
    )
    assert float(primary["value"]) == 4.67
    assert semantics == "latest_instantaneous"
    assert summary is None


def test_missing_sentinels_never_win_the_peak():
    values = _values() + [
        {"value": "-999999", "dateTime": "2026-09-13T21:00:00.000-04:00"},
    ]
    primary, summary, _ = select_stage_observation(values, windowed=True)
    assert float(primary["value"]) == 6.79
    assert summary["value_count"] == 5
    # All-sentinel window is an error, not a fabricated stage
    assert select_stage_observation(
        [{"value": "-999999", "dateTime": "x"}], windowed=True
    ) is None


# ------------------------------------------------------------------
# 2. Modeled stage-buffer extent
# ------------------------------------------------------------------

def _waterways():
    return [
        {
            "osm_id": 1,
            "name": "Saddle River",
            "waterway": "river",
            "coordinates": [
                [-74.10, 40.85], [-74.08, 40.88],
                [-74.06, 40.91], [-74.04, 40.94],
            ],
        },
        {
            "osm_id": 2,
            "name": "Tributary",
            "waterway": "stream",
            "coordinates": [
                [-74.09, 40.95], [-74.075, 40.90],
            ],
        },
    ]


def test_stage_buffer_builds_disclosed_extent():
    geom = GeometryEngine()
    result = geom.build_stage_buffer_extent(
        waterways=_waterways(),
        peak_stage_ft=6.79,
        action_stage_ft=5.0,
        minor_stage_ft=5.5,
        moderate_stage_ft=7.0,
    )
    assert result is not None
    assert result.geojson.get("features")
    assert result.area_km2 > 0.2  # 2 buffered lines, radius > 0.5 km
    # Exceedance 1.79 ft anchors between minor (0.5 ft -> 0.4 km) and
    # moderate (2.0 ft -> 0.8 km): ~0.75 km radius
    model = result.model
    assert model["radius_basis"] == "nwps_category_anchored"
    assert 0.5 < model["radius_km"] < 0.9
    assert model["waterway_count"] == 2
    assert model["peak_stage_ft"] == 6.79
    assert "NOT observed inundation" in model["semantics"]


def test_stage_buffer_refuses_without_positive_exceedance():
    geom = GeometryEngine()
    # Peak below action: nothing to model (no trigger, no map)
    assert geom.build_stage_buffer_extent(
        waterways=_waterways(),
        peak_stage_ft=4.67,
        action_stage_ft=5.0,
    ) is None
    # No waterway geometry: nothing to buffer
    assert geom.build_stage_buffer_extent(
        waterways=[{"coordinates": [[-74.1, 40.9]]}],
        peak_stage_ft=6.79,
        action_stage_ft=5.0,
    ) is None
    # Radius cap is honored even for extreme exceedance
    capped = geom.build_stage_buffer_extent(
        waterways=_waterways(),
        peak_stage_ft=25.0,
        action_stage_ft=5.0,
        minor_stage_ft=5.5,
        moderate_stage_ft=7.0,
        max_km=2.5,
    )
    assert capped.model["radius_km"] == 2.5


def test_stage_buffer_anchor_selects_nearest_waterways_only():
    """The gauge measures ONE river: with an anchor, only the nearest
    waterways are buffered -- a Saddle River stage reading must not
    flood every brook in a 15 km urban radius (Lodi NJ live-data
    regression: 1067 waterways -> 733 km2 absurd extent)."""
    geom = GeometryEngine()
    # Ten far-away tributaries plus one main stem through the station
    far = [
        {
            "osm_id": 100 + i,
            "name": f"Far Brook {i}",
            "waterway": "stream",
            "coordinates": [[-74.50, 41.20], [-74.49, 41.21]],
        }
        for i in range(10)
    ]
    main_stem = {
        "osm_id": 1,
        "name": "Saddle River",
        "waterway": "river",
        "coordinates": [[-74.10, 40.85], [-74.06, 40.91]],
    }
    result = geom.build_stage_buffer_extent(
        waterways=far + [main_stem],
        peak_stage_ft=6.79,
        action_stage_ft=5.0,
        minor_stage_ft=5.5,
        moderate_stage_ft=7.0,
        anchor=(40.89027778, -74.0805556),  # the Lodi gauge
        max_waterways=3,
    )
    model = result.model
    assert model["waterway_selection"] == "nearest_to_gauge"
    # Only the main stem is near enough to matter
    selected = {w["name"] for w in model["selected_waterways"]}
    assert "Saddle River" in selected
    assert len(model["selected_waterways"]) == 1
    # A sane corridor size (the pre-anchor regression produced 733 km2)
    assert result.area_km2 < 30.0


def test_stage_buffer_stream_radius_reduced():
    geom = GeometryEngine()
    river = {
        "name": "River", "waterway": "river",
        "coordinates": [[-74.10, 40.85], [-74.06, 40.91]],
    }
    stream = {
        "name": "Brook", "waterway": "stream",
        "coordinates": [[-74.10, 40.85], [-74.06, 40.91]],
    }
    r = GeometryEngine().build_stage_buffer_extent(
        waterways=[river], peak_stage_ft=6.79, action_stage_ft=5.0,
        minor_stage_ft=5.5, moderate_stage_ft=7.0,
    )
    s = geom.build_stage_buffer_extent(
        waterways=[stream], peak_stage_ft=6.79, action_stage_ft=5.0,
        minor_stage_ft=5.5, moderate_stage_ft=7.0,
        stream_fraction=0.4,
    )
    # Same geometry, stream class -> smaller buffered area
    assert s.area_km2 < r.area_km2
    assert s.model["stream_radius_fraction"] == 0.4
