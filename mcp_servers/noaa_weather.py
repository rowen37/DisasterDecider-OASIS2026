# noaa_weather.py
import json
import os
import httpx
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NOAA Weather")

# ------------------------- Constants -------------------------
EXPECTED_PRECIP_UNIT = "wmoUnit:mm"   # standard unit of NWS precipitation fields
VALID_PRECIP_HOURS = (1, 3, 6)       # the only windows NWS /observations/latest provides

async def _get_grid_endpoint(latitude: float, longitude: float) -> dict:
    points_url = f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            points_url,
            headers={"User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, test@example.com)")},
            timeout=float(os.environ.get("MCP_HTTP_TIMEOUT", 15.0)),
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("properties", {})


@mcp.tool()
async def get_weather_observations(latitude: float, longitude: float) -> str:
    try:
        properties = await _get_grid_endpoint(latitude, longitude)
        stations_url = properties.get("observationStations")
        if not stations_url:
            return json.dumps({"status": "error", "error": "Could not find observation stations"})

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                stations_url,
                headers={"User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, test@example.com)")},
                timeout=float(os.environ.get("MCP_HTTP_TIMEOUT", 15.0)),
            )
            resp.raise_for_status()
            stations = resp.json().get("features", [])
            if not stations:
                return json.dumps({"status": "error", "error": "No observation stations found"})

            station_id = stations[0].get("properties", {}).get("stationIdentifier")
            if not station_id:
                return json.dumps({"status": "error", "error": "No station identifier found"})

            obs_resp = await client.get(
                f"https://api.weather.gov/stations/{station_id}/observations/latest",
                headers={"User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, test@example.com)")},
                timeout=float(os.environ.get("MCP_HTTP_TIMEOUT", 15.0)),
            )
            obs_resp.raise_for_status()
            data = obs_resp.json()
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"NOAA API request failed: {str(exc)}"})

    try:
        obs = data.get("properties", {})
        return json.dumps({
            "status": "ok",
            "source": "NOAA",
            "source_type": "weather_observation",
            "timestamp": obs.get("timestamp"),
            "location": {"latitude": latitude, "longitude": longitude},
            "measurements": {
                "value": obs.get("temperature", {}).get("value"),
                "unit": "°C",
                "variable": "temperature"
            },
            "metadata": {
                "station_id": station_id,
                "text_description": obs.get("textDescription"),
                "humidity": obs.get("relativeHumidity", {}).get("value"),
                "wind_speed": obs.get("windSpeed", {}).get("value"),
                "pressure": obs.get("barometricPressure", {}).get("value")
            },
            "metadata_verified": True
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Failed to parse NOAA response: {str(exc)}"})


@mcp.tool()
async def get_precipitation(latitude: float, longitude: float, hours: int = 6) -> str:
    """
    Get precipitation. Iterates over the nearest observation stations until
    one reports a precipitation field.

    hours accepts only 1 / 3 / 6, matching the three windows provided by
    NWS /observations/latest. NWS does not support arbitrary-hour
    precipitation queries; for true 24h data use the
    /stations/{sid}/observations?start=...&end=... endpoint instead.

    Environment variables:
        NWS_PRECIP_MAX_STATIONS (default=15): max number of nearby stations to try
    """
    # hours input validation
    if hours not in VALID_PRECIP_HOURS:
        return json.dumps({
            "status": "error",
            "error": (
                f"hours={hours} is not supported. "
                "NWS /observations/latest only provides 1h, 3h, or 6h precipitation windows. "
                "For longer windows please use the observations time-range endpoint."
            ),
            "valid_hours": list(VALID_PRECIP_HOURS),
            "metadata_verified": False,
            "action_required": "Do NOT treat missing precipitation as 0.0 mm.",
        }, ensure_ascii=False)

    max_try = int(os.environ.get("NWS_PRECIP_MAX_STATIONS", "15"))

    try:
        properties = await _get_grid_endpoint(latitude, longitude)
        stations_url = properties.get("observationStations")
        if not stations_url:
            return json.dumps({
                "status": "error",
                "error": "Could not find observation stations for precipitation.",
                "metadata_verified": False,
                "action_required": "Do NOT treat missing precipitation as 0.0 mm.",
            }, ensure_ascii=False)

        headers = {"User-Agent": os.environ.get(
            "NWS_USER_AGENT", "(disaster-agent, contact@example.com)"
        )}
        timeout = float(os.environ.get("MCP_HTTP_TIMEOUT", 15.0))

        async with httpx.AsyncClient() as client:
            st_resp = await client.get(stations_url, headers=headers, timeout=timeout)
            st_resp.raise_for_status()
            stations = st_resp.json().get("features", [])
            if not stations:
                return json.dumps({
                    "status": "error",
                    "error": "No observation stations found.",
                    "metadata_verified": False,
                    "action_required": "Do NOT treat missing precipitation as 0.0 mm.",
                }, ensure_ascii=False)

            precip_fields = (
                "precipitationLastHour",
                "precipitationLast3Hours",
                "precipitationLast6Hours",
            )

            tried: list[dict] = []
            chosen_obs = None
            chosen_station = None
            # Last station that responded successfully, used to stamp
            # the "no precipitation observed" result.
            last_props: dict = {}
            last_station: str | None = None

            for st in stations[:max_try]:
                sid = st.get("properties", {}).get("stationIdentifier")
                if not sid:
                    continue
                try:
                    r = await client.get(
                        f"https://api.weather.gov/stations/{sid}/observations/latest",
                        headers=headers, timeout=timeout,
                    )
                    r.raise_for_status()
                    props = r.json().get("properties", {})
                except Exception as e:
                    tried.append({"station": sid, "error": str(e)})
                    continue

                last_props = props
                last_station = sid

                # Validate unitCode: accept a field only when its unit is wmoUnit:mm
                def _has_valid_precip(field: str, obs: dict) -> bool:
                    v = obs.get(field)
                    if not isinstance(v, dict):
                        return False
                    if v.get("value") is None:
                        return False
                    unit = v.get("unitCode", "")
                    if unit != EXPECTED_PRECIP_UNIT:
                        import sys
                        print(
                            f"[get_precipitation] Unexpected unitCode for {field} "
                            f"at station {sid}: {unit!r}",
                            file=sys.stderr,
                        )
                        return False
                    return True

                has_precip = any(_has_valid_precip(f, props) for f in precip_fields)
                tried.append({"station": sid, "has_precip": has_precip})

                if has_precip:
                    chosen_obs = props
                    chosen_station = sid
                    break

            if chosen_obs is None:
                # NWS returns null precipitation values (not 0.0) when no
                # precipitation is recorded. "No observed rainfall" is a
                # distinct, meaningful observation — it must not be conflated
                # with "precipitation data unavailable", and it must not be
                # fabricated as 0.0 mm either.
                return json.dumps({
                    "status": "ok",
                    "source": "NWS",
                    "source_type": "precipitation",
                    "observation_semantics": "no_precipitation_observed",
                    "note": (
                        "None of the nearest NWS stations reported measurable "
                        "precipitation; NWS returns null values (not 0.0) when "
                        "no precipitation is recorded, which indicates no "
                        "observed rainfall rather than missing data."
                    ),
                    "stations_checked": len(
                        [t for t in tried if t.get("has_precip") is False]
                    ),
                    "timestamp": last_props.get("timestamp"),
                    "location": {"latitude": latitude, "longitude": longitude},
                    "metadata": {
                        "last_station_checked": last_station,
                        "stations_tried": tried,
                    },
                    "metadata_verified": False,
                    "action_required": (
                        "Do NOT treat this as 0.0 mm and do NOT report it as "
                        "missing data; report it as 'no observed precipitation'."
                    ),
                }, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"NWS precipitation fetch failed: {str(exc)}",
            "metadata_verified": False,
            "action_required": "Do NOT treat missing precipitation as 0.0 mm.",
        }, ensure_ascii=False)

    # _extract applies the same unitCode screening
    def _extract(field: str) -> float | None:
        v = chosen_obs.get(field, {})
        if not isinstance(v, dict) or v.get("value") is None:
            return None
        unit = v.get("unitCode", "")
        if unit != EXPECTED_PRECIP_UNIT:
            import sys
            print(
                f"[get_precipitation] _extract: unexpected unitCode for {field}: {unit!r}",
                file=sys.stderr,
            )
            return None
        try:
            return float(v["value"])
        except (TypeError, ValueError):
            return None

    p1 = _extract("precipitationLastHour")
    p3 = _extract("precipitationLast3Hours")
    p6 = _extract("precipitationLast6Hours")

    # hours accepts only 1/3/6; pick the available window closest to the request
    if hours == 1:
        best_mm, period = (p1, 1) if p1 is not None else \
                         (p3, 3) if p3 is not None else \
                         (p6, 6) if p6 is not None else (None, None)
    elif hours == 3:
        best_mm, period = (p3, 3) if p3 is not None else \
                         (p6, 6) if p6 is not None else \
                         (p1, 1) if p1 is not None else (None, None)
    else:  # hours == 6
        best_mm, period = (p6, 6) if p6 is not None else \
                         (p3, 3) if p3 is not None else \
                         (p1, 1) if p1 is not None else (None, None)

    if best_mm is None:
        return json.dumps({
            "status": "error",
            "error": (
                f"Station {chosen_station} passed unit screening but all "
                "precipitation values are None after extraction."
            ),
            "stations_tried": tried,
            "metadata_verified": False,
            "action_required": "Do NOT treat missing precipitation as 0.0 mm.",
        }, ensure_ascii=False)

    best_in = best_mm / 25.4

    # Warn when the returned window differs from the requested one
    warning = None
    if period != hours:
        warning = (
            f"Requested {hours}h window but only {period}h data was available "
            f"at station {chosen_station}."
        )

    result = {
        "status": "ok",
        "source": "NWS",
        "source_type": "precipitation",
        "timestamp": chosen_obs.get("timestamp"),
        "location": {"latitude": latitude, "longitude": longitude},
        "measurements": {
            "precipitation":    {"value": round(best_in, 4), "unit": "in"},
            "precipitation_mm": {"value": round(best_mm, 2), "unit": "mm"},
            "period_hours": period,
            "requested_hours": hours,
            "variable": "precipitation",
            "value": round(best_mm, 2),
            "unit": "mm",
            "all_available": {
                "last_1h_mm": round(p1, 2) if p1 is not None else None,
                "last_3h_mm": round(p3, 2) if p3 is not None else None,
                "last_6h_mm": round(p6, 2) if p6 is not None else None,
            },
        },
        "metadata": {
            "station_id": chosen_station,
            "stations_tried": tried,
        },
        "metadata_verified": True,
    }

    if warning:
        result["warning"] = warning

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float, periods: int = 7) -> str:
    try:
        properties = await _get_grid_endpoint(latitude, longitude)
        forecast_url = properties.get("forecast")
        if not forecast_url:
            return json.dumps({"status": "error", "error": "Could not find forecast URL"})

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                forecast_url,
                headers={"User-Agent": os.environ.get("NWS_USER_AGENT", "(disaster-agent, test@example.com)")},
                timeout=float(os.environ.get("MCP_HTTP_TIMEOUT", 15.0)),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"NOAA forecast API failed: {str(exc)}"})

    try:
        forecasts = data.get("properties", {}).get("periods", [])[:periods]
        return json.dumps({
            "status": "ok",
            "source": "NOAA",
            "source_type": "forecast",
            "location": {"latitude": latitude, "longitude": longitude},
            "forecasts": [
                {
                    "period": f.get("name"),
                    "temperature": f.get("temperature"),
                    "temperature_unit": f.get("temperatureUnit"),
                    "wind_speed": f.get("windSpeed"),
                    "short_forecast": f.get("shortForecast"),
                    "probability_of_precipitation": f.get("probabilityOfPrecipitation", {}).get("value"),
                    "timestamp": f.get("startTime"),
                }
                for f in forecasts
            ],
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Failed to parse NOAA forecast: {str(exc)}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")