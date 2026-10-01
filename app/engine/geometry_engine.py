"""Geometry Engine — spatial clipping and affected-population estimation.

The Skill layer orchestrates data flow only.  All Shapely / GeoPandas
imports stay lazy (method-level) so the module degrades gracefully when
the optional geospatial stack is unavailable.
"""

from __future__ import annotations

import json
import math

from dataclasses import dataclass
from typing import Any

from ..utils import safe_float as _safe_float


@dataclass
class ClipResult:
    """Outcome of clipping a satellite flood extent to a city boundary."""

    clipped_geojson: dict[str, Any]
    flooded_area_km2: float
    city_area_km2: float
    flood_union_geom: Any
    polygon_count: int
    city_boundary_geom: Any = None


@dataclass
class StageBufferResult:
    """Outcome of the modeled stage-buffer flood extent."""

    geojson: dict[str, Any]
    area_km2: float
    union_geom: Any
    # Full audit of the model inputs and radius policy so the map can
    # disclose exactly how the extent was produced.
    model: dict[str, Any]


class GeometryEngine:
    """Pure spatial computation: geometries in, geometries and numbers out."""

    @staticmethod
    def _as_geometry(boundary: Any) -> Any:
        """
        Accept either a bare geometry GeoJSON dict or a FeatureCollection
        (Nominatim-style boundary responses may be either).  A
        FeatureCollection fed straight to shapely ``shape()`` raises,
        so it is unpacked here.
        """
        from shapely.geometry import shape as _shp_shape
        from shapely.ops import unary_union as _shp_union

        if not isinstance(boundary, dict):
            return None
        if boundary.get("type") == "FeatureCollection":
            geoms = []
            for feat in boundary.get("features", []):
                try:
                    geom = _shp_shape(feat.get("geometry", {}))
                    if not geom.is_empty:
                        geoms.append(geom)
                except Exception:
                    continue
            return _shp_union(geoms) if geoms else None
        try:
            return _shp_shape(boundary)
        except Exception:
            return None

    def clip_flood_extent_to_city(
        self,
        city_boundary_geojson: dict[str, Any],
        flood_geojson: dict[str, Any],
    ) -> ClipResult | None:
        """
        Clip flood polygons to the city administrative boundary and recompute
        the flooded area on an equal-area projection (EPSG:6933).

        Returns ``None`` when no flooded pixels fall inside the city (the
        caller decides how to represent the empty case).
        """
        from shapely.geometry import shape as _shp_shape
        from shapely.ops import unary_union as _shp_union

        city_geom = self._as_geometry(city_boundary_geojson)
        if city_geom is None or city_geom.is_empty:
            return None

        clipped_geoms = []
        for feat in flood_geojson["features"]:
            try:
                geom = _shp_shape(feat.get("geometry", {}))
                inter = geom.intersection(city_geom)
                if not inter.is_empty:
                    clipped_geoms.append(inter)
            except Exception:
                continue

        if not clipped_geoms:
            return None

        clipped_union = _shp_union(clipped_geoms)
        import geopandas as gpd
        clipped_gdf = gpd.GeoDataFrame(
            geometry=[clipped_union], crs="EPSG:4326"
        )
        # Areas computed on the equal-area projection (km²)
        city_area_km2 = float(
            gpd.GeoDataFrame(
                geometry=[city_geom], crs="EPSG:4326"
            )
            .to_crs("EPSG:6933")
            .geometry.area.iloc[0]
            / 1e6
        )
        clipped_area_km2 = float(
            clipped_gdf.to_crs("EPSG:6933")
            .geometry.area.iloc[0]
            / 1e6
        )
        clipped_geojson = json.loads(
            clipped_gdf.to_json(drop_id=True)
        )

        return ClipResult(
            clipped_geojson=clipped_geojson,
            flooded_area_km2=round(clipped_area_km2, 4),
            city_area_km2=city_area_km2,
            flood_union_geom=clipped_union,
            polygon_count=len(clipped_geoms),
            city_boundary_geom=city_geom,
        )

    def build_stage_buffer_extent(
        self,
        *,
        waterways: list[dict[str, Any]],
        peak_stage_ft: float | None,
        action_stage_ft: float | None,
        minor_stage_ft: float | None = None,
        moderate_stage_ft: float | None = None,
        base_km: float = 0.15,
        minor_km: float = 0.4,
        moderate_km: float = 0.8,
        max_km: float = 1.5,
        anchor: tuple[float, float] | None = None,
        max_waterways: int = 5,
        stream_fraction: float = 0.4,
        max_corridor_km: float = 10.0,
    ) -> StageBufferResult | None:
        """First-order hydraulic-proximity flood extent.

        Buffers the OSM waterway network by a radius that scales with
        the gauge peak's exceedance above action stage, piecewise-anchored
        at the NWPS minor/moderate categories:

            exceedance 0       -> base_km
            minor - action     -> minor_km
            moderate - action  -> moderate_km   (linear between anchors,
                                                extrapolated beyond,
                                                capped at max_km)

        The gauge measures ONE river: with an ``anchor`` (station
        coordinates) only the few nearest waterways are buffered (the
        gauged main stem plus immediate confluences), and streams get a
        reduced radius (``stream_fraction``) versus rivers/canals --
        a stage reading on the Saddle River must not flood every brook
        within 15 km.

        This is a disclosed PLANNING HEURISTIC, not hydraulics: it exists
        so that a gauge-confirmed flood still gets a spatial picture
        (map, exposure, routes) when satellite imagery is missing or
        mistimed. It deliberately over-includes within the gauged
        corridor because the operational cost of a missed flood exceeds
        the cost of an over-wide advisory polygon. Without minor/moderate
        anchors the radius falls back to ``base_km * (1 + exceedance_ft)``
        capped at ``max_km``.

        Returns None when there is nothing to model (no positive
        exceedance or no usable waterway geometry).
        """
        peak = _safe_float(peak_stage_ft)
        action = _safe_float(action_stage_ft)
        minor = _safe_float(minor_stage_ft)
        moderate = _safe_float(moderate_stage_ft)
        if peak is None or action is None or peak <= action:
            return None

        exceedance_ft = peak - action

        def _interp(
            x: float, x0: float, y0: float,
            x1: float, y1: float, clamp: bool = True,
        ) -> float:
            if x1 <= x0:
                return y1
            t = (x - x0) / (x1 - x0)
            if clamp:
                t = min(1.0, max(0.0, t))
            return y0 + t * (y1 - y0)

        if minor is not None and minor > action:
            radius_km = _interp(exceedance_ft, 0.0, base_km,
                                minor - action, minor_km)
            if moderate is not None and moderate > minor:
                if exceedance_ft > minor - action:
                    # Beyond moderate, keep the minor->moderate slope
                    # (unclamped extrapolation); the max_km cap below is
                    # the only ceiling.
                    radius_km = _interp(
                        exceedance_ft, minor - action, minor_km,
                        moderate - action, moderate_km, clamp=False,
                    )
            radius_basis = "nwps_category_anchored"
        else:
            radius_km = base_km * (1.0 + exceedance_ft)
            radius_basis = "linear_fallback_no_minor_anchor"
        radius_km = round(min(max_km, radius_km), 3)

        from shapely.geometry import LineString as _ShpLineString

        def _way_distance_km(way: dict[str, Any]) -> float:
            # Min great-circle distance from the anchor to the way's
            # vertices (kilometers); infinity when unparseable.
            coords = way.get("coordinates") or []
            best = float("inf")
            for p in coords:
                try:
                    lon, lat = float(p[0]), float(p[1])
                except (TypeError, ValueError, IndexError):
                    continue
                dlon = math.radians(lon - anchor[1])
                dlat = math.radians(lat - anchor[0])
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(anchor[0]))
                    * math.cos(math.radians(lat))
                    * math.sin(dlon / 2) ** 2
                )
                best = min(best, 6371.0 * 2 * math.asin(math.sqrt(a)))
            return best

        candidates = [
            w for w in waterways or []
            if isinstance(w, dict)
            and isinstance(w.get("coordinates"), list)
            and len(w["coordinates"]) >= 2
        ]
        selected = candidates
        selection_basis = "all_waterways"
        if anchor is not None:
            # The gauge measures one river: keep only waterways within
            # the corridor around the station, nearest first (main stem
            # + immediate confluences). Everything else in a dense
            # urban search radius is un-gauged territory the stage
            # reading says nothing about.
            ranked = sorted(
                (
                    (_way_distance_km(w), w) for w in candidates
                ),
                key=lambda pair: pair[0],
            )
            in_corridor = [
                w for d, w in ranked
                if d <= max_corridor_km
            ][: max(1, max_waterways)]
            selected = in_corridor or [
                w for _, w in ranked[: max(1, max_waterways)]
            ]
            selection_basis = "nearest_to_gauge"

        lines = []
        for way in selected:
            coords = way["coordinates"]
            try:
                line = _ShpLineString(
                    [(float(p[0]), float(p[1])) for p in coords]
                )
                if not line.is_empty:
                    lines.append((way, line))
            except (TypeError, ValueError):
                continue
        if not lines:
            return None

        from shapely.ops import unary_union as _shp_union

        # Buffer in the equal-area CRS so the radius is true meters.
        # Rivers/canals carry the full exceedance-scaled radius; streams
        # a reduced fraction (a stage on the main stem propagates far
        # less up tiny tributaries).
        projected = []
        for way, line in lines:
            p = self._equal_area(line)
            if p is None or p.is_empty:
                continue
            way_radius = radius_km
            if str(way.get("waterway") or "").lower() == "stream":
                way_radius = radius_km * stream_fraction
            projected.append((way_radius, p))
        if not projected:
            return None
        buffered = [
            p.buffer(r_km * 1000.0) for r_km, p in projected
        ]
        union = _shp_union(buffered)
        if union is None or union.is_empty:
            return None

        # Back to WGS84 for GeoJSON; area from the equal-area union.
        try:
            from pyproj import Transformer as _Transformer
            from shapely.ops import transform as _shp_transform

            _inv = _Transformer.from_crs(
                "EPSG:6933", "EPSG:4326", always_xy=True
            ).transform
            wgs84_union = _shp_transform(
                lambda x, y, z=None: _inv(x, y), union
            )
        except Exception:
            return None

        import geopandas as gpd

        area_km2 = float(union.area) / 1e6
        gdf = gpd.GeoDataFrame(geometry=[wgs84_union], crs="EPSG:4326")
        geojson = json.loads(gdf.to_json(drop_id=True))

        model = {
            "method": "stage_buffer",
            "radius_km": radius_km,
            "radius_basis": radius_basis,
            "waterway_selection": selection_basis,
            "stream_radius_fraction": stream_fraction,
            "peak_stage_ft": peak,
            "action_stage_ft": action,
            "minor_stage_ft": minor,
            "moderate_stage_ft": moderate,
            "exceedance_ft": round(exceedance_ft, 3),
            "waterway_count": len(lines),
            "selected_waterways": [
                {
                    "name": way.get("name"),
                    "type": way.get("waterway"),
                }
                for way, _ in lines
            ],
            "buffer_params_km": {
                "base": base_km, "minor": minor_km,
                "moderate": moderate_km, "max": max_km,
            },
            "semantics": (
                "First-order hydraulic-proximity model: the waterways "
                "nearest the gauge buffered by a stage-exceedance-"
                "scaled radius (streams at a reduced fraction). NOT "
                "observed inundation and NOT a hydraulic simulation; "
                "deliberately over-inclusive within the gauged "
                "corridor (advisory posture)."
            ),
        }
        return StageBufferResult(
            geojson=geojson,
            area_km2=round(area_km2, 4),
            union_geom=wgs84_union,
            model=model,
        )

    def filter_tracts_to_boundary(
        self,
        tracts: list[dict[str, Any]] | None,
        boundary_geom: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Keep only census tracts intersecting the analysis boundary.

        Enforces one spatial scope: the exposure numerator (flood ∩ tract,
        flood already clipped to the city) and the population/SVI
        denominator (tract set) must come from the same extent; otherwise
        tracts outside the city pulled in by the 25 km circular search
        dilute population_factor and vulnerability_coverage.
        Returns (filtered tracts, excluded count).
        """
        if not tracts or boundary_geom is None:
            return list(tracts or []), 0
        kept, excluded = [], 0
        for tract in tracts:
            poly = self._tract_polygon(tract)
            try:
                if poly is not None and poly.intersects(boundary_geom):
                    kept.append(tract)
                elif poly is None:
                    # No geometry means undecidable: keep the tract and
                    # let downstream honesty rules handle it.
                    kept.append(tract)
                else:
                    excluded += 1
            except Exception:
                kept.append(tract)
        return kept, excluded

    def union_features(
        self,
        geojson: Any,
    ) -> Any | None:
        """
        Build a union geometry from a raw GEE FeatureCollection.

        Used as fallback when city clipping did not run, so affected
        population estimation remains possible.
        """
        if not (
            geojson
            and isinstance(geojson, dict)
            and geojson.get("features")
        ):
            return None

        from shapely.geometry import shape as _shp_shape
        from shapely.ops import unary_union as _shp_union

        _geoms = []
        for _f in geojson.get("features", []):
            try:
                _g = _shp_shape(_f.get("geometry", {}))
                if not _g.is_empty:
                    _geoms.append(_g)
            except Exception:
                continue
        if _geoms:
            return _shp_union(_geoms)
        return None

    @staticmethod
    def _tract_polygon(tract: dict[str, Any]) -> Any | None:
        """ESRI rings -> shapely Polygon (outer ring only; None on parse failure)."""
        geometry = tract.get("geometry") or {}
        rings = geometry.get("rings") if isinstance(geometry, dict) else None
        if not (isinstance(rings, list) and rings):
            return None
        from shapely.geometry import Polygon as _ShpPolygon

        ring = rings[0]
        if not (
            isinstance(ring, list)
            and len(ring) >= 3
            and all(
                isinstance(pt, (list, tuple)) and len(pt) >= 2
                for pt in ring[:3]
            )
        ):
            return None
        try:
            poly = _ShpPolygon(ring)
            return poly if not poly.is_empty else None
        except Exception:
            return None

    def flood_center_radius(self, geom: Any) -> tuple[float, float, float] | None:
        """Flood footprint -> (lat, lon, coverage radius in meters):
        equal-area centroid + farthest-vertex distance.

        Used to couple POI/facility search to the footprint: the search
        center follows the footprint centroid and the radius spans the
        whole footprint extent (equal-area CRS units are meters).
        Returns None on failure.
        """
        try:
            from pyproj import Transformer as _Transformer
            from shapely.ops import transform as _shp_transform

            ea = self._equal_area(geom)
            if ea is None or ea.is_empty:
                return None
            _inv = _Transformer.from_crs(
                "EPSG:6933", "EPSG:4326", always_xy=True
            ).transform
            c = ea.centroid
            clon, clat = _inv(c.x, c.y)
            r_m = 0.0
            polys = list(ea.geoms) if ea.geom_type == "MultiPolygon" else [ea]
            for poly in polys:
                for px, py in poly.exterior.coords:
                    r_m = max(r_m, math.hypot(px - c.x, py - c.y))
            return (clat, clon, r_m)
        except Exception:
            return None

    @staticmethod
    def _equal_area(geom: Any) -> Any | None:
        """Project to the equal-area CRS (EPSG:6933) for area ratios; None on failure."""
        try:
            from pyproj import Transformer as _Transformer
            from shapely.ops import transform as _shp_transform

            _project = _Transformer.from_crs(
                "EPSG:4326", "EPSG:6933", always_xy=True
            ).transform
            return _shp_transform(
                lambda x, y, z=None: _project(x, y), geom
            )
        except Exception:
            return None

    def tract_exposure(
        self,
        flood_union_geom: Any,
        social_vulnerability: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Per-tract area-weighted exposure (input to the VWUN fairness ledger).

        Each tract returns:
            flooded_fraction     flood area ∩ tract area / tract area (equal-area projection)
            exposed_population   population × flooded_fraction
            svi                  passed through as-is (None if missing)
        If a tract geometry cannot support an area ratio, that tract is
        marked unavailable. It is never promoted to whole-tract exposure
        merely because a centroid falls inside the flood polygon.
        """
        records: list[dict[str, Any]] = []
        for tract in social_vulnerability.get("tracts", []):
            pop = _safe_float(tract.get("population"))
            svi = _safe_float(tract.get("svi"))
            poly = self._tract_polygon(tract)
            method = "areal_weighted"
            if poly is not None:
                projected = self._equal_area(poly)
                inter = None
                if projected is not None:
                    try:
                        inter = self._equal_area(
                            poly.intersection(flood_union_geom)
                        )
                    except Exception:
                        inter = None
                if projected is not None and inter is not None:
                    denom = projected.area
                    fraction = (
                        inter.area / denom if denom > 0 else 0.0
                    )
                else:
                    method = "unavailable_area_ratio"
                    fraction = None
                # Centroid exposure recorded separately: the other end of
                # the affected-population method bracket.
                try:
                    centroid_exposed = bool(
                        flood_union_geom.contains(poly.centroid)
                    )
                except Exception:
                    centroid_exposed = None
            else:
                method = "unavailable_area_ratio"
                fraction = None
                centroid_exposed = None

            exposed = (
                pop * fraction
                if pop is not None and fraction is not None
                else None
            )
            records.append(
                {
                    "tract_id": tract.get("tract_id"),
                    "population": pop,
                    "svi": svi,
                    "flooded_fraction": (
                        round(fraction, 6)
                        if fraction is not None
                        else None
                    ),
                    "exposed_population": (
                        round(exposed, 2) if exposed is not None else None
                    ),
                    "centroid_exposed": centroid_exposed,
                    "method": method,
                }
            )
        return records

    @staticmethod
    def affected_population_bracket(
        records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Method bracket for affected population: areal-weighted estimate
        vs centroid whole-tract estimate.

        The true value lies between the two (it depends on the
        within-tract population distribution); reporting the point
        estimate (areal-weighted) alongside this bracket is more honest
        than a single number -- it makes the uncertainty of the
        uniform-within-tract assumption explicit.
        """
        areal = sum(
            r.get("exposed_population") or 0.0
            for r in records
            if r.get("exposed_population") is not None
        )
        centroid = sum(
            r.get("population") or 0.0
            for r in records
            if r.get("centroid_exposed")
        )
        if areal == 0.0 and centroid == 0.0:
            return None
        return {
            "point_estimate": round(areal, 0),
            "methods": {
                "areal_weighted": round(areal, 0),
                "centroid_whole_tract": round(centroid, 0),
            },
            "interval": [
                round(min(areal, centroid), 0),
                round(max(areal, centroid), 0),
            ],
        }

    def estimate_affected_population(
        self,
        flood_union_geom: Any,
        social_vulnerability: dict[str, Any],
    ) -> tuple[float | None, int, str]:
        """
        Area-weighted affected population:
        population × (flood ∩ tract area / tract area).

        The equal-area projection (EPSG:6933) keeps area ratios correct.
        Tracts without usable geometry make this tract-level estimate
        unavailable instead of triggering whole-tract centroid counting.

        Returns (affected population, affected tract count, method label).
        """
        records = self.tract_exposure(flood_union_geom, social_vulnerability)
        populated_records = [
            record
            for record in records
            if record.get("population") is not None
        ]
        if not populated_records:
            return None, 0, "unavailable_area_ratio"
        if any(
            record.get("exposed_population") is None
            for record in populated_records
        ):
            return None, 0, "unavailable_area_ratio"
        affected_pop = 0.0
        affected_tracts = 0
        for record in populated_records:
            exposed = record.get("exposed_population")
            if exposed is None:
                continue
            affected_pop += exposed
            if record.get("flooded_fraction", 0.0) > 0.0:
                affected_tracts += 1
        return affected_pop, affected_tracts, "areal_weighted"

    @staticmethod
    def estimate_uniform_density_population(
        total_population: Any,
        flooded_area_km2: Any,
        analysis_area_km2: Any,
    ) -> float | None:
        """Coarse fallback using one population density for the same scope."""
        population = _safe_float(total_population)
        flooded_area = _safe_float(flooded_area_km2)
        analysis_area = _safe_float(analysis_area_km2)
        if (
            population is None
            or population < 0
            or flooded_area is None
            or flooded_area < 0
            or analysis_area is None
            or analysis_area <= 0
        ):
            return None
        return population * min(flooded_area / analysis_area, 1.0)

    @staticmethod
    def count_facilities_in_extent(
        flood_union_geom: Any,
        facilities: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Count flooded and locatable facilities from the same inventory."""
        from shapely.geometry import Point

        affected = 0
        locatable = 0
        for facility in facilities:
            if not isinstance(facility, dict):
                continue
            lat = _safe_float(facility.get("latitude"))
            lon = _safe_float(facility.get("longitude"))
            if lat is None or lon is None:
                continue
            locatable += 1
            if flood_union_geom.covers(Point(lon, lat)):
                affected += 1
        return affected, locatable
