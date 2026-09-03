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
        Falls back to centroid containment (fraction ∈ {0, 1}) when the
        tract geometry is unusable, and says so in the method field --
        no fabricated intermediate precision.
        """
        from shapely.geometry import Point as _ShpPoint

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
                    # Projection failed: fall back to centroid containment and flag it
                    method = "centroid_fallback"
                    centroid = poly.centroid
                    fraction = (
                        1.0
                        if flood_union_geom.contains(centroid)
                        else 0.0
                    )
                # Centroid exposure recorded separately: the other end of
                # the affected-population method bracket.
                try:
                    centroid_exposed = bool(
                        flood_union_geom.contains(poly.centroid)
                    )
                except Exception:
                    centroid_exposed = None
            else:
                method = "centroid_fallback"
                geometry = tract.get("geometry") or {}
                rings = (
                    geometry.get("rings")
                    if isinstance(geometry, dict)
                    else None
                )
                centroid = None
                if isinstance(rings, list) and rings:
                    ring = rings[0]
                    if (
                        isinstance(ring, list)
                        and len(ring) >= 3
                        and all(
                            isinstance(pt, (list, tuple)) and len(pt) >= 2
                            for pt in ring[:3]
                        )
                    ):
                        xs = [pt[0] for pt in ring]
                        ys = [pt[1] for pt in ring]
                        centroid = _ShpPoint(
                            sum(xs) / len(xs), sum(ys) / len(ys)
                        )
                fraction = (
                    1.0
                    if centroid is not None
                    and flood_union_geom.contains(centroid)
                    else 0.0
                )
                centroid_exposed = fraction >= 1.0

            exposed = pop * fraction if pop is not None else None
            records.append(
                {
                    "tract_id": tract.get("tract_id"),
                    "population": pop,
                    "svi": svi,
                    "flooded_fraction": round(fraction, 6),
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
    ) -> tuple[float, int, str]:
        """
        Area-weighted affected population:
        population × (flood ∩ tract area / tract area).

        The equal-area projection (EPSG:6933) keeps area ratios correct;
        tracts without usable geometry fall back to centroid containment.
        Unlike centroid counting (whole tract counted when its centroid
        is flooded), this avoids systematic overestimation for small
        floods inside large tracts.

        Returns (affected population, affected tract count, method label).
        """
        records = self.tract_exposure(flood_union_geom, social_vulnerability)
        affected_pop = 0.0
        affected_tracts = 0
        methods = {r["method"] for r in records}
        for record in records:
            exposed = record.get("exposed_population")
            if exposed is None:
                continue
            affected_pop += exposed
            if record.get("flooded_fraction", 0.0) > 0.0:
                affected_tracts += 1
        method = (
            "areal_weighted"
            if methods == {"areal_weighted"}
            else "areal_weighted_with_centroid_fallback"
        )
        return affected_pop, affected_tracts, method
