import sys
import os
_k = os.environ.get("CENSUS_API_KEY")
# Report key presence only, never length: key length is a credential
# fingerprint that would leak into collected stderr logs.
print(f"[population_exposure] CENSUS_API_KEY set: {bool(_k)}", file=sys.stderr)

import json
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Population Exposure")

@mcp.tool()
async def get_population_exposure(
    latitude: float,
    longitude: float,
    radius_km: float = 10,
) -> str:
    """
    Get a population exposure estimate near a given location.

    Data sources:
        1. Census Geocoder - converts lat/lon to FIPS codes (state, county, tract)
        2. Census ACS 5-Year Population API - total population for that tract

    Note:
        The Census API does not support radius queries by coordinate, so this
        tool returns the population of the census tract containing the point,
        not an aggregate within the radius (radius_km is kept for interface
        compatibility only).
    """
    # ---------- 1. Environment variable check ----------
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        return json.dumps({
            "status": "error",
            "error": "CENSUS_API_KEY is not set. Please obtain a key from https://api.census.gov/data/key_signup.html",
            "metadata_verified": False,
            "action_required": "Set CENSUS_API_KEY in environment variables."
        }, ensure_ascii=False)

    # ---------- 2. Get FIPS codes via Census Geocoder ----------
    geocode_url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    geo_params = {
        "x": longitude,
        "y": latitude,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo_resp = await client.get(geocode_url, params=geo_params)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"Census Geocoder request failed: {str(exc)}",
            "metadata_verified": False,
            "action_required": "Check network connectivity or Census Geocoder availability."
        }, ensure_ascii=False)

    # Extract census tract info
    try:
        tracts = geo_data.get("result", {}).get("geographies", {}).get("Census Tracts", [])
        if not tracts:
            raise ValueError("No Census Tract found for the given coordinates.")
        tract_info = tracts[0]
        state_fips = tract_info.get("STATE")
        county_fips = tract_info.get("COUNTY")
        tract_fips = tract_info.get("TRACT")
        if not (state_fips and county_fips and tract_fips):
            raise ValueError("Incomplete FIPS codes returned from Geocoder.")
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"Failed to parse Geocoder response: {str(exc)}",
            "metadata_verified": False,
            "action_required": "Verify coordinates are within the US and its territories."
        }, ensure_ascii=False)

    # ---------- 3. Call the Census Population API ----------
    # ACS 5-Year 2020 dataset
    pop_url = "https://api.census.gov/data/2020/acs/acs5"
    pop_params = {
        "get": "NAME,B01003_001E",           # B01003_001E = total population
        "for": f"tract:{tract_fips}",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": api_key
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            pop_resp = await client.get(pop_url, params=pop_params)
            pop_resp.raise_for_status()
            pop_data = pop_resp.json()
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": f"Census Population API request failed: {str(exc)}",
            "metadata_verified": False,
            "action_required": "Check CENSUS_API_KEY validity and network connectivity."
        }, ensure_ascii=False)

    # Parse the result (pop_data is a 2D array: first row is the header)
    if not pop_data or len(pop_data) < 2:
        return json.dumps({
            "status": "error",
            "error": "Census API returned empty or incomplete data.",
            "metadata_verified": False
        }, ensure_ascii=False)

    try:
        header = pop_data[0]
        row = pop_data[1]
        # Column order matches the requested fields: NAME, B01003_001E, state, county, tract
        name = row[0]
        population = int(row[1]) if row[1] else None
    except (IndexError, ValueError) as exc:
        return json.dumps({
            "status": "error",
            "error": f"Failed to parse Census population response: {str(exc)}",
            "metadata_verified": False
        }, ensure_ascii=False)

    # ---------- 4. Return result ----------
    return json.dumps({
        "status": "ok",
        "source": "Census ACS 5-Year 2020",
        "source_type": "population_exposure",
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "radius_km": radius_km,
        "census_geography": {
            "state_fips": state_fips,
            "county_fips": county_fips,
            "tract_fips": tract_fips,
            "full_fips": f"{state_fips}{county_fips}{tract_fips}",
            "tract_name": name
        },
        "population": {
            "total": population,
            "unit": "people",
            "data_year": 2020,
            "dataset": "ACS 5-Year"
        },
        "metadata_verified": True,
        "note": "Population returned for the census tract containing the point, not radius aggregate."
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")