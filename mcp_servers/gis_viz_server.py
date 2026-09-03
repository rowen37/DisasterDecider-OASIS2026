#!/usr/bin/env python3
"""
GIS Visualization MCP Server
Produces interactive maps, time-series charts, and combined HTML reports.
"""
import base64
import html as _html
import json
import logging
import os
import sys
from pathlib import Path

from gis_io import is_allowed_path as _path_ok, out_path as _out
from datetime import datetime, timezone

import folium
import geopandas as gpd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import MiniMap
from mcp.server.fastmcp import FastMCP

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("gis-viz")

mcp = FastMCP("gis-viz-tools")


def _png_to_b64(png_path: str) -> str:
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _to_wgs84(gdf):
    """Normalize a loaded layer to EPSG:4326.

    A missing CRS is declared as 4326; a projected CRS (e.g. 3857 output
    from raster services) must be truly reprojected, otherwise Folium
    draws metric coordinates as lat/lon and the map is garbage.
    """
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() not in ("EPSG:4326", "4326", "WGS84"):
        return gdf.to_crs("EPSG:4326")
    return gdf


@mcp.tool()
def vis_flood_map(
    center_lat: float,
    center_lon: float,
    flood_boundary_path: str = "",
    affected_buildings_path: str = "",
    blocked_roads_path: str = "",
    rescue_route_path: str = "",
    poi_path: str = "",
    station_lat: float = 0.0,
    station_lon: float = 0.0,
    station_name: str = "",
    target_lat: float = 0.0,
    target_lon: float = 0.0,
    target_name: str = "",
    zoom_start: int = 13,
    output_path: str = "",
) -> str:
    """Generate a multi-layer interactive HTML flood map using Folium."""
    try:
        output_path = _out(output_path, ".html")
        # Standard OpenStreetMap tiles: CartoDB positron now requires an
        # API key and shows an "API KEY REQUIRED" watermark on free usage.
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")
        MiniMap(toggle_display=True).add_to(m)
        layers_added = []
        # Collect layer extents, then fit bounds to the data (incl.
        # station/target) so the initial view is not an unrelated wide area.
        fit_points: list[tuple[float, float]] = [(center_lat, center_lon)]

        if flood_boundary_path and os.path.exists(flood_boundary_path):
            gdf = _to_wgs84(gpd.read_file(flood_boundary_path))
            if not gdf.empty:
                folium.GeoJson(
                    gdf.__geo_interface__,
                    name="🟥 Flood Boundary",
                    style_function=lambda _: {"fillColor": "#FF4444", "color": "#CC0000", "weight": 2, "fillOpacity": 0.35},
                ).add_to(m)
                layers_added.append("flood_boundary")
                for _pt in gdf.geometry.representative_point():
                    if _pt is not None and not _pt.is_empty:
                        fit_points.append((_pt.y, _pt.x))

        if affected_buildings_path and os.path.exists(affected_buildings_path):
            gdf = _to_wgs84(gpd.read_file(affected_buildings_path))
            if not gdf.empty:
                gdf_wgs = gdf
                name_fields = [c for c in gdf_wgs.columns if c.lower() in ("name", "amenity", "type") and c != "geometry"]
                folium.GeoJson(
                    gdf_wgs.__geo_interface__,
                    name=f"🟨 Affected Buildings ({len(gdf_wgs)})",
                    style_function=lambda _: {"fillColor": "#FFD700", "color": "#FFA500", "weight": 1, "fillOpacity": 0.65},
                    tooltip=folium.GeoJsonTooltip(fields=name_fields) if name_fields else None,
                ).add_to(m)
                layers_added.append("affected_buildings")

        if blocked_roads_path and os.path.exists(blocked_roads_path):
            gdf = _to_wgs84(gpd.read_file(blocked_roads_path))
            if not gdf.empty:
                folium.GeoJson(
                    gdf.__geo_interface__,
                    name="🟧 Blocked Roads",
                    style_function=lambda _: {"color": "#FF8C00", "weight": 3, "dashArray": "8 4", "fillOpacity": 0},
                ).add_to(m)
                layers_added.append("blocked_roads")

        if rescue_route_path and os.path.exists(rescue_route_path):
            gdf = _to_wgs84(gpd.read_file(rescue_route_path))
            if not gdf.empty:
                props = gdf.iloc[0].drop("geometry").to_dict() if len(gdf) else {}
                dist = props.get("path_length_km", props.get("path_length_m", "?"))
                label = f"🟩 Rescue Route ({dist} {'km' if 'path_length_km' in props else 'm'})"
                folium.GeoJson(
                    gdf.__geo_interface__,
                    name=label,
                    style_function=lambda _: {"color": "#00AA00", "weight": 4, "fillOpacity": 0},
                ).add_to(m)
                layers_added.append("rescue_route")

        AMENITY_COLORS = {
            "hospital": "red", "clinic": "pink", "pharmacy": "lightred",
            "fire_station": "orange", "police": "blue", "school": "purple",
            "shelter": "green", "supermarket": "beige", "fuel": "gray",
        }
        if poi_path and os.path.exists(poi_path):
            gdf = _to_wgs84(gpd.read_file(poi_path))
            if not gdf.empty:
                poi_group = folium.FeatureGroup(name="🔵 Facilities / POIs")
                for _, row in gdf.iterrows():
                    if row.geometry is None:
                        continue
                    amenity = row.get("amenity", "unknown")
                    raw_name = row.get("name")
                    osm_id = row.get("osm_id", row.get("id", ""))
                    name = raw_name if raw_name and str(raw_name).lower() != "unnamed" else (
                        f"{str(amenity).replace('_', ' ').title()}" + (f" (OSM {osm_id})" if osm_id else "")
                    )
                    color = AMENITY_COLORS.get(amenity, "cadetblue")
                    pt = row.geometry if row.geometry.geom_type == "Point" else row.geometry.centroid
                    # OSM name/amenity are externally controlled strings:
                    # escape before inserting into popup/tooltip, else an OSM
                    # name containing <script> becomes a stored XSS.
                    _safe_name = _html.escape(str(name))
                    _safe_amenity = _html.escape(str(amenity))
                    folium.Marker(
                        location=[pt.y, pt.x],
                        popup=folium.Popup(
                            f"<b>{_safe_name}</b><br>{_safe_amenity}",
                            max_width=200,
                        ),
                        tooltip=_safe_name,
                        icon=folium.Icon(color=color, icon="info-sign"),
                    ).add_to(poi_group)
                poi_group.add_to(m)
                layers_added.append("pois")

        if station_lat and station_lon:
            _station_label = _html.escape(station_name or "USGS monitoring station")
            folium.Marker(
                location=[station_lat, station_lon],
                popup=_station_label,
                tooltip=_html.escape(station_name or "Gauge station"),
                icon=folium.Icon(color="blue", icon="tint"),
            ).add_to(m)
            layers_added.append("station")
            fit_points.append((station_lat, station_lon))

        if target_lat and target_lon:
            _target_label = _html.escape(target_name or "Requested target location")
            folium.Marker(
                location=[target_lat, target_lon],
                popup=_target_label,
                tooltip=_html.escape(target_name or "Target city"),
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(m)
            layers_added.append("target")
            fit_points.append((target_lat, target_lon))

        folium.LayerControl(collapsed=False).add_to(m)

        # Fit the view to the event extent (flood layers + station + target)
        # instead of staying at country scale.
        if len(fit_points) > 1:
            try:
                m.fit_bounds(fit_points, padding=(30, 30))
            except Exception:
                pass

        m.save(output_path)
        log.info("vis_flood_map → %s  layers=%s", output_path, layers_added)
        return json.dumps({
            "output_path": output_path,
            "layers_added": layers_added,
            "layer_count": len(layers_added),
            "_meta": {"type": "html_map", "tool": "vis_flood_map"},
        })
    except Exception as exc:
        log.exception("vis_flood_map failed")
        return json.dumps({"error": str(exc), "tool": "vis_flood_map"})


@mcp.tool()
def vis_time_series(
    series_json: str,
    title: str,
    y_label: str = "Value",
    x_label: str = "Time",
    thresholds_json: str = "",
    forecast_json: str = "",
    output_path: str = "",
) -> str:
    """Render a time-series chart (PNG) with optional threshold lines and forecast."""
    try:
        output_path = _out(output_path, ".png")
        series = json.loads(series_json)
        if not series:
            return json.dumps({"error": "series_json is empty", "tool": "vis_time_series"})
        df = pd.DataFrame(series)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").fillna(
            pd.to_datetime(df["timestamp"], errors="coerce")
        )
        df = df.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["value"], color="#1565C0", linewidth=2,
                label=df["label"].iloc[0] if "label" in df.columns else "Observed", zorder=3)
        ax.fill_between(df["timestamp"], df["value"], alpha=0.15, color="#1565C0")
        if forecast_json:
            fdf = pd.DataFrame(json.loads(forecast_json))
            fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], unit="s", errors="coerce").fillna(
                pd.to_datetime(fdf["timestamp"], errors="coerce")
            )
            ax.plot(fdf["timestamp"], fdf["value"], color="#F57C00", linewidth=2,
                    linestyle="--", label="Forecast", zorder=3)
        if thresholds_json:
            for thr in json.loads(thresholds_json):
                val = thr.get("value")
                label = thr.get("label", f"Threshold {val}")
                color = thr.get("color", "#D32F2F")
                ax.axhline(val, color=color, linewidth=1.5, linestyle=":", alpha=0.85, label=label)
                ax.annotate(label, xy=(df["timestamp"].iloc[-1], val), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color=color)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return json.dumps({
            "output_path": output_path,
            "data_points": len(df),
            "_meta": {"type": "png_chart", "tool": "vis_time_series"},
        })
    except Exception as exc:
        log.exception("vis_time_series failed")
        return json.dumps({"error": str(exc), "tool": "vis_time_series"})


@mcp.tool()
def vis_report_page(
    title: str,
    map_path: str = "",
    chart_paths_json: str = "",
    summary_text: str = "",
    statistics_json: str = "",
    alert_level: str = "LOW",
    output_path: str = "",
) -> str:
    """Combine map, charts, statistics and narrative into a single-file HTML report."""
    try:
        output_path = _out(output_path, ".html")
        ALERT_COLORS = {
            "LOW": ("#1B5E20", "#E8F5E9"),
            "MODERATE": ("#E65100", "#FFF3E0"),
            "HIGH": ("#B71C1C", "#FFEBEE"),
            "CRITICAL": ("#4A0000", "#FF1744"),
        }
        hdr_color, bg_color = ALERT_COLORS.get(alert_level.upper(), ALERT_COLORS["LOW"])
        chart_paths = json.loads(chart_paths_json) if chart_paths_json else []
        statistics = json.loads(statistics_json) if statistics_json else {}
        sections = []
        # The report embeds local file contents (chart base64 / map iframe):
        # only paths inside allowed roots (temp dir, static/maps) are
        # accepted, so arbitrary local files cannot end up in a distributable
        # HTML report. Rejected paths are listed in the return value.
        skipped_paths: list[str] = []

        if statistics:
            rows = "".join(
                f"<tr><td style='padding:6px 12px;font-weight:600;color:#555'>{_html.escape(str(k))}</td>"
                f"<td style='padding:6px 12px'>{_html.escape(str(v))}</td></tr>"
                for k, v in statistics.items()
            )
            sections.append(f"""
<section>
  <h2 style="color:{hdr_color}">Key Statistics</h2>
  <table style="border-collapse:collapse;width:100%;max-width:640px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.12)">
    {rows}
  </table>
</section>""")

        if summary_text:
            html_text = _html.escape(summary_text).replace("\n", "<br>")
            sections.append(f"""
<section>
  <h2 style="color:{hdr_color}">Situation Summary</h2>
  <div style="background:#fff;padding:16px 20px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.12);line-height:1.7">
    {html_text}
  </div>
</section>""")

        for cp in chart_paths:
            if not _path_ok(cp):
                skipped_paths.append(str(cp))
                log.warning("vis_report_page: chart path outside allowed roots, skipped: %s", cp)
                continue
            if os.path.exists(cp):
                b64 = _png_to_b64(cp)
                name = Path(cp).stem.replace("_", " ").title()
                sections.append(f"""
<section>
  <h2 style="color:{hdr_color}">{_html.escape(name)}</h2>
  <img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.12)">
</section>""")

        if map_path and os.path.exists(map_path):
            if not _path_ok(map_path):
                skipped_paths.append(str(map_path))
                log.warning("vis_report_page: map path outside allowed roots, skipped: %s", map_path)
            else:
                abs_map = str(Path(map_path).resolve())
                sections.append(f"""
<section>
  <h2 style="color:{hdr_color}">Situational Awareness Map</h2>
  <iframe src="file://{abs_map}" width="100%" height="520" style="border:none;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.12)"></iframe>
</section>""")

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        # title / alert_level are caller-controlled strings; escape before HTML
        title_esc = _html.escape(title)
        alert_esc = _html.escape(alert_level)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title_esc}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:{bg_color};padding:24px;color:#212121}}
    header{{background:{hdr_color};color:#fff;padding:20px 28px;border-radius:10px;
            margin-bottom:24px;display:flex;justify-content:space-between;align-items:center}}
    header h1{{font-size:1.55rem;font-weight:700}}
    header .meta{{font-size:.85rem;opacity:.85;text-align:right;line-height:1.5}}
    section{{margin-bottom:24px}}
    h2{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#333}}
    @media(max-width:600px){{header{{flex-direction:column;gap:8px}}}}
  </style>
</head>
<body>
  <header>
    <h1>🚨 {title_esc}</h1>
    <div class="meta">
      Alert Level: <strong>{alert_esc}</strong><br>
      Generated: {now_str}
    </div>
  </header>
  {''.join(sections)}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("vis_report_page → %s  sections=%d", output_path, len(sections))
        return json.dumps({
            "output_path": output_path,
            "sections_included": len(sections),
            "alert_level": alert_level,
            "skipped_paths": skipped_paths,
            "_meta": {"type": "html_report", "tool": "vis_report_page"},
        })
    except Exception as exc:
        log.exception("vis_report_page failed")
        return json.dumps({"error": str(exc), "tool": "vis_report_page"})


if __name__ == "__main__":
    mcp.run()
