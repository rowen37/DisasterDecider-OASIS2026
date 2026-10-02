# Historical validation results

Execution date: 2026-09-30.

Repository regression check: 2026-10-02. A cache-free offline run completed
with **130 passed** tests. This is separate from the dated real-service event
records below; those external-service runs were not repeated during workspace
cleanup.

This record separates two different claims:

1. **Hydrologic validation** checks the official USGS peak and the category
   produced from the station-specific NWPS thresholds.
2. **Full Agent acceptance** currently applies ten explicit checks: completed
   run; station identity; historical mode; peak stage; hydrologic category;
   allowed extent provenance; positive population exposure; completed
   optimization; a Pareto recommendation; and at least two operator-choice
   scenarios. It does not currently assert SVI content, affected-building
   count, POI availability, or rescue-route production, and it does not prove
   that any generated artifact is ground truth.

## Status after the facility-service-area change

The current working tree now screens OSM hospitals and shelters around the
assessment target using a disclosed 30 km/h planning speed:

- primary: 30 minutes = 15 km;
- extended: 60 minutes = 30 km, used only after a successful empty primary
  query;
- rescue-route time: OSM route length divided by 30 km/h, not live traffic;
- allocation p95 distance: straight-line demand-to-facility assignment
  distance, which is a separate metric.

The facility query currently retains at most ten returned POIs and routes to
the nearest among those returned. The implementation therefore does not yet
prove that the selected facility is reachable under actual flood traffic or
suitable for a particular service requirement. The road download uses a 2 km
half-width target-to-facility corridor and expands once to 4 km only if the
base graph is disconnected; flood-avoiding and reference routes reuse it.

The 15/30 km policy applies only to that GIS facility/map-route layer. Resource
optimization uses a separate `get_available_resources` query. The versioned
environment template supplies a 30 km radius; the server falls back to 50 km
only when no `radius_km` argument is supplied. Its
candidate-plan `response_time` objective still uses a 60 km/h straight-line
heuristic. Capacity-constrained assignment recomputes coverage and unmet
demand but does not replace that objective, so the two speed assumptions are
not yet unified.

It also requests the complete building inventory inside the configured
near-field survey radius instead of accepting a silently truncated Overpass
result. Repository regression status is recorded by the current
`uv run python -m pytest -q` output rather than a manually maintained test count. The
suite includes the 15 km to 30 km facility fallback and route-production
regression.

The real-service event records below were captured before these later
facility-search and building-query changes. They remain a dated historical
record, not a post-change rerun. The real-service command shown below must be
run again before publishing updated POI, building, or rescue-route claims.

## Balanced held-out sample

Harvey/Friendswood 2017 remains development-only and is excluded from all
held-out totals.

| Stratum | Held-out event | Peak | Result | Full Agent |
|---|---|---:|---|---|
| below_action | Lodi 2026-09-22 | 2.25 ft | passed | no |
| action | Imelda / Friendswood 2019-09-18 | 11.64 ft | passed | passed |
| minor | Lodi 2026-09-13 | 6.79 ft | passed | no |
| moderate | Richmond 2020-11-14 | 18.33 ft | passed | no |
| major | Ida / Manville 2021-09-02 | 27.66 ft | passed | passed |

The `--refresh-usgs` run re-downloaded all five official series. Every stored
peak and peak-time range matched on 2026-09-30.

## Full Agent acceptance

The following command ran both full cases against the configured USGS, NWPS,
Nominatim, Census, Earth Engine, OSM/Overpass, CDC SVI, and local GIS tools:

```bash
uv run python scripts/validate_full_agent_events.py \
  --output-dir /tmp/oasis-full-agent-validation
```

Recorded result at the time: **passed (2/2 events)** under the ten checks
listed above.

### Imelda / Friendswood

- Run: `historical-validation-98de8f44-c8b6-4994-89e5-6c207c266c77`
- Official peak/category: 11.64 ft, action — matched.
- Time mode: historical; no mismatched source was admitted.
- Operational extent: Sentinel-1 SAR, 7.8477 km² inside the city boundary.
- Estimated exposed population: 6,211 people across 14 intersecting tracts.
- Resource result: optimized; `shelter_top3`; four operator-facing plan
  scenarios were produced.
- Acceptance checks: 10/10 passed (these checks did not require a route or
  validate SVI content).
- Warnings: archived NWPS stage-flow is unavailable; a non-essential GIS POI
  request timed out after several Overpass 504 responses. Resource discovery
  recovered through another Overpass endpoint and the allocation still ran.

### Ida / Manville

- Run: `historical-validation-afb6af83-a63e-4dc6-a9ad-92bd8d6128af`
- Official peak/category: 27.66 ft, major — matched.
- Time mode: historical; no mismatched source was admitted.
- Operational extent: Sentinel-1 SAR, 0.8919 km² inside the borough boundary.
- Estimated exposed population: 1,424 people across 4 intersecting tracts.
- Resource result: optimized; `hospital_top3`; four operator-facing plan
  scenarios were produced.
- Acceptance checks: 10/10 passed (these checks did not require a route or
  validate SVI content).
- Warning: archived NWPS stage-flow is unavailable.

## Limits that remain

These runs do **not** independently validate:

- the accuracy of the SAR polygons against surveyed or high-water-mark ground
  truth;
- the CDRI Low/Moderate/High/Very High cut points;
- the accuracy of evacuation or transfer-time estimates;
- whether the 15/30 km service-radius proxy matches real flood-time road
  access, or whether a rescue route is produced for every event;
- nearest-facility selection and facility service suitability;
- consistency between the 30 km/h GIS route estimate and the resource
  candidate generator's 60 km/h response-time objective;
- SVI content, affected-building counts, or POI completeness as part of the
  ten scripted acceptance checks;
- community preferences or service-suitability claims.

The last item is intentionally not inferred from SVI. Community requirements
must enter as sourced, editable input, and plan labels must show which
requirements each facility or plan can and cannot satisfy.
