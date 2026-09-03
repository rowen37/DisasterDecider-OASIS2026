#!/usr/bin/env python3
"""
GIS Raster Operations MCP Server
DORA-inspired raster analysis tools for flood disaster response.
"""
import json
import logging
import os
import sys

from gis_io import out_path as _out
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("gis-raster")

mcp = FastMCP("gis-raster-tools")


def _pixel_area_m2(src) -> float:
    from pyproj import CRS
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    if CRS(src.crs).is_geographic:
        bounds = src.bounds
        clat = (bounds.top + bounds.bottom) / 2.0
        mlon = 111_320.0 * abs(np.cos(np.radians(clat)))
        mlat = 111_320.0
        return res_x * mlon * res_y * mlat
    return res_x * res_y


@mcp.tool()
def ras_vectorize(
    raster_path: str,
    class_value: int = 1,
    output_path: str = "",
    min_area_m2: float = 500.0,
    simplify_tolerance: float = 0.00005,
) -> str:
    """Convert a raster class mask to vector polygon GeoJSON."""
    try:
        output_path = _out(output_path, ".geojson")
        with rasterio.open(raster_path) as src:
            band = src.read(1)
            transform = src.transform
            crs = src.crs
            pix_m2 = _pixel_area_m2(src)

        binary = (band == class_value).astype(np.uint8)
        if binary.sum() == 0:
            import geopandas as gpd
            gpd.GeoDataFrame(geometry=[], crs=crs).to_file(output_path, driver="GeoJSON")
            return json.dumps({
                "output_path": output_path,
                "polygon_count": 0,
                "total_area_km2": 0.0,
                "total_area_m2": 0.0,
                "warning": f"No pixels with class_value={class_value}",
                "_meta": {"crs": "EPSG:4326", "type": "polygon", "tool": "ras_vectorize"},
            })

        from shapely.geometry import shape as shp
        import geopandas as gpd
        polys = [shp(g) for g, v in shapes(binary, mask=binary, transform=transform) if v == 1]
        gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
        # Areas must be computed in a local/metric CRS: Web Mercator (3857)
        # inflates area by 1/cos^2(lat) at mid latitudes (~+30% for Houston),
        # so prefer estimate_utm_crs and fall back to equal-area EPSG:6933.
        try:
            metric_crs = gdf.estimate_utm_crs()
        except Exception:
            metric_crs = None
        if metric_crs is None:
            metric_crs = "EPSG:6933"
        gdf_m = gdf.to_crs(metric_crs)
        gdf_m["area_m2"] = gdf_m.geometry.area
        gdf_m = gdf_m[gdf_m["area_m2"] >= min_area_m2].copy()
        if gdf_m.empty:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(output_path, driver="GeoJSON")
            return json.dumps({
                "output_path": output_path,
                "polygon_count": 0,
                "total_area_km2": 0.0,
                "total_area_m2": 0.0,
                "warning": f"All polygons filtered out (min_area_m2={min_area_m2})",
                "_meta": {"crs": "EPSG:4326", "type": "polygon", "tool": "ras_vectorize"},
            })

        total_m2 = float(gdf_m["area_m2"].sum())
        gdf_out = gdf_m.to_crs("EPSG:4326")
        if simplify_tolerance > 0:
            gdf_out["geometry"] = gdf_out.geometry.simplify(simplify_tolerance, preserve_topology=True)
        gdf_out.to_file(output_path, driver="GeoJSON")
        log.info("ras_vectorize → %d polygons, %.4f km²", len(gdf_out), total_m2 / 1e6)

        return json.dumps({
            "output_path": output_path,
            "polygon_count": len(gdf_out),
            "total_area_km2": round(total_m2 / 1e6, 4),
            "total_area_m2": round(total_m2, 1),
            "class_extracted": class_value,
            "_meta": {"crs": "EPSG:4326", "type": "polygon", "tool": "ras_vectorize", "source_raster": raster_path},
        })
    except Exception as exc:
        log.exception("ras_vectorize failed")
        return json.dumps({"error": str(exc), "tool": "ras_vectorize"})


@mcp.tool()
def ras_area(
    raster_path: str,
    class_value: int = 1,
) -> str:
    """Calculate total area (m² and km²) of a pixel class in a raster."""
    try:
        with rasterio.open(raster_path) as src:
            band = src.read(1)
            pix_m2 = _pixel_area_m2(src)
        count = int((band == class_value).sum())
        return json.dumps({
            "class_value": class_value,
            "pixel_count": count,
            "total_area_m2": round(count * pix_m2, 2),
            "total_area_km2": round(count * pix_m2 / 1e6, 4),
            "pixel_area_m2": round(pix_m2, 4),
            "_meta": {"type": "scalar", "tool": "ras_area"},
        })
    except Exception as exc:
        log.exception("ras_area failed")
        return json.dumps({"error": str(exc), "tool": "ras_area"})


@mcp.tool()
def ras_threshold(
    raster_path: str,
    threshold: float,
    operator: str = "greater_equal",
    band_index: int = 1,
    output_path: str = "",
) -> str:
    """Apply a numeric threshold to a raster band → binary mask GeoTIFF."""
    ops = {
        "greater":       lambda a, t: a > t,
        "greater_equal": lambda a, t: a >= t,
        "less":          lambda a, t: a < t,
        "less_equal":    lambda a, t: a <= t,
        "equal":         lambda a, t: a == t,
        "not_equal":     lambda a, t: a != t,
    }
    try:
        if operator not in ops:
            return json.dumps({"error": f"operator must be one of {list(ops)}", "tool": "ras_threshold"})
        output_path = _out(output_path, ".tif")
        with rasterio.open(raster_path) as src:
            band = src.read(band_index).astype(np.float32)
            profile = src.profile.copy()
            pix_m2 = _pixel_area_m2(src)
        mask = ops[operator](band, threshold).astype(np.uint8)
        true_count = int(mask.sum())
        profile.update(dtype=rasterio.uint8, count=1, nodata=None)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)
        return json.dumps({
            "output_path": output_path,
            "true_pixel_count": true_count,
            "true_area_m2": round(true_count * pix_m2, 2),
            "true_area_km2": round(true_count * pix_m2 / 1e6, 4),
            "operator": operator,
            "threshold": threshold,
            "_meta": {"type": "raster", "tool": "ras_threshold", "source": raster_path},
        })
    except Exception as exc:
        log.exception("ras_threshold failed")
        return json.dumps({"error": str(exc), "tool": "ras_threshold"})


@mcp.tool()
def ras_zonal_stats(
    raster_path: str,
    zones_path: str,
    stats: str = "mean,sum,count,min,max",
    band_index: int = 1,
    zone_id_field: str = "",
) -> str:
    """Compute per-zone raster statistics within vector polygons."""
    funcs = {
        "mean":  np.nanmean,
        "sum":   np.nansum,
        "count": lambda x: float(np.sum(~np.isnan(x))),
        "min":   np.nanmin,
        "max":   np.nanmax,
        "std":   np.nanstd,
    }
    try:
        requested = [s.strip() for s in stats.split(",") if s.strip() in funcs]
        if not requested:
            return json.dumps({"error": "No valid stats requested", "tool": "ras_zonal_stats"})
        import geopandas as gpd
        from shapely.geometry import mapping
        from rasterio.mask import mask as rio_mask

        zones = gpd.read_file(zones_path)
        results: list[dict[str, Any]] = []
        with rasterio.open(raster_path) as src:
            if zones.crs != src.crs:
                zones = zones.to_crs(src.crs)
            for idx, row in zones.iterrows():
                zid = str(row[zone_id_field]) if zone_id_field in zones.columns else str(idx)
                try:
                    img, _ = rio_mask(
                        src, [mapping(row.geometry)],
                        crop=True, nodata=np.nan, all_touched=True,
                    )
                    vals = img[band_index - 1].astype(np.float32)
                    if src.nodata is not None:
                        vals[vals == src.nodata] = np.nan
                    rec: dict[str, Any] = {"zone_id": zid}
                    for s in requested:
                        try:
                            v = float(funcs[s](vals))
                            rec[s] = round(v, 4) if not np.isnan(v) else None
                        except Exception:
                            rec[s] = None
                    results.append(rec)
                except Exception as ze:
                    results.append({"zone_id": zid, "error": str(ze)})
        return json.dumps({
            "zone_stats": results,
            "zone_count": len(results),
            "stats_computed": requested,
            "_meta": {"type": "dict", "tool": "ras_zonal_stats"},
        })
    except Exception as exc:
        log.exception("ras_zonal_stats failed")
        return json.dumps({"error": str(exc), "tool": "ras_zonal_stats"})


@mcp.tool()
def ras_diff(
    pre_raster_path: str,
    post_raster_path: str,
    output_path: str = "",
    change_threshold: float = 0.5,
) -> str:
    """Bi-temporal change detection: compute (post - pre) difference raster."""
    try:
        output_path = _out(output_path, ".tif")
        with rasterio.open(pre_raster_path) as pre:
            pre_band = pre.read(1).astype(np.float32)
            profile = pre.profile.copy()
            pix_m2 = _pixel_area_m2(pre)
            nodata = pre.nodata
        with rasterio.open(post_raster_path) as post:
            post_band = post.read(1).astype(np.float32)

        h = min(pre_band.shape[0], post_band.shape[0])
        w = min(pre_band.shape[1], post_band.shape[1])
        pre_band = pre_band[:h, :w]
        post_band = post_band[:h, :w]
        if nodata is not None:
            pre_band[pre_band == nodata] = np.nan
            post_band[post_band == nodata] = np.nan

        diff = post_band - pre_band
        changed = (np.abs(diff) >= change_threshold)
        changed[np.isnan(diff)] = False
        changed_count = int(changed.sum())
        valid = diff[~np.isnan(diff)]
        profile.update(dtype=rasterio.float32, count=1, height=h, width=w)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(diff, 1)

        return json.dumps({
            "output_path": output_path,
            "changed_pixel_count": changed_count,
            "changed_area_m2": round(changed_count * pix_m2, 2),
            "changed_area_km2": round(changed_count * pix_m2 / 1e6, 4),
            "mean_diff": round(float(np.nanmean(valid)), 4) if len(valid) else 0.0,
            "max_diff": round(float(np.nanmax(valid)), 4) if len(valid) else 0.0,
            "min_diff": round(float(np.nanmin(valid)), 4) if len(valid) else 0.0,
            "change_threshold": change_threshold,
            "_meta": {"type": "raster", "tool": "ras_diff"},
        })
    except Exception as exc:
        log.exception("ras_diff failed")
        return json.dumps({"error": str(exc), "tool": "ras_diff"})


if __name__ == "__main__":
    mcp.run()