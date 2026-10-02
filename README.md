# Disaster Spatial Agent — extensible hazard decider (flood reference plugin)

Two constructed baselines — a naive catch-all dashboard and a
plan-then-execute scaffold (DORA-inspired) — plus a Skill / Verification /
Adaptive-HITL layer, restructured as a **hazard-plugin architecture**: the flood pipeline is the
reference implementation. New hazards (wildfire, typhoon, ...) register
without adding hazard branches to the shared router, agent, or serving loop;
they still require one package import and one API `HazardType` entry. See
`docs/EXTENDING.md`.

Architecture:

```
User → Master Router (registry) → Hazard Skill plugin → MCP tools
     → Spatial State → Verification → HITL → Equity ledger → Decision Support
```

Four decoupled layers:

- **Skills** orchestrate data flow, HITL and evidence (hazard plugins live here)
- **Engine** pure computation (`<hazard>_<role>_engine` + shared engines)
- **Ops** MCP lifecycle, timeout/retry policy, call metrics
- **Agent** evidence-grounded LLM synthesis with anti-hallucination output
  validation; hazard-specific rules are injected by the plugin
  (`OUTPUT_RULES` / `OUTPUT_VALIDATOR`)

Flood-plugin skill boundaries:

| unit | responsibility |
|---|---|
| `GeocodeSkill` / `UsgsSkill` | resolve place and verified gauge observations |
| `FusionSourcesSkill` | collect, time-align, and fuse weather/infrastructure evidence |
| `SviSkill` / `ResourceSkill` | collect vulnerability tracts and candidate resources |
| `FloodSkill` | orchestrate evidence gates, GIS calls, optimization, and audit output |
| `RiskEngine` / `AllocationEngine` | pure CDRI/EPS and resource-allocation computation |

## 1. Requirements

- Python 3.11+ (3.13/3.14 fine)
- uv
- API key for live LLM synthesis (not needed by offline demo/tests)

## 2. Install

```bash
uv sync --frozen
cp .env.example .env   # then edit .env
```

`uv sync --frozen` installs the exact versions in `uv.lock` and fails instead
of silently rewriting the lock file. The offline demo and tests need no
credentials. Live runs need the credentials for the data sources used.

## 3. Connect your MCPs

Edit `config/mcp.json` (stdio or Streamable HTTP). Bundled servers:

| server | purpose |
|---|---|
| `flood` | USGS instantaneous river stage |
| `usgs_metadata` | station metadata + NWPS gauge thresholds/stageflow |
| `geocode` | place-name → coordinates / city boundary (station identity verification) |
| `nws` / `noaa` | alerts, forecasts, precipitation |
| `gee-flood-extent` | Sentinel-1 SAR flood extent (Google Earth Engine) |
| `population` / `social_vulnerability` | Census ACS population, CDC/ATSDR SVI |
| `osm` / `resource` | infrastructure and emergency resources |
| `gis-raster` / `gis-vector` / `gis-viz` | spatial algebra + interactive maps |

Server commands use two placeholders resolved at load time
(`app/mcp_client.py`): `python` means the interpreter running the app
(the project venv), and `${PROJECT_ROOT}` / `${ENV_VAR}` expand to the
project root and to environment variables (`.env` is loaded first), so
the config is machine-independent. The GEE server reads
`GEE_SERVICE_ACCOUNT_KEY_PATH` / `GEE_SERVICE_ACCOUNT_EMAIL` from the
environment; unset values make that single server fail to connect and be
skipped with a logged warning — everything else keeps working.

## 4. Run

```bash
uv run python -m app.main          # CLI REPL
uv run python -m app.main --web    # Web UI + REST API (default 127.0.0.1:8000)
```

Example tasks:

- `Assess flood impact near Friendswood using USGS station 08077600`
- `Is the area around Houston flooding right now?`
- Historical replay (web UI date field or `event_date` in the API):
  the same task with event date 2017-08-27 replays the event window
  under strict time alignment (every time offset is disclosed in the
  run log's per-source `time_alignment` ledger).
- Offline demo (no network, any date): enable **Offline demo** in the web UI,
  or call `POST /api/assess` with `{"query": "...", "demo": true}`. The
  nominal scenario runs the full fixture-backed decision chain. Fault
  scenarios deliberately block or degrade later products, so indices or
  allocation output are not guaranteed in those cases. Supported
  `demo_scenario` values are `nominal`, `metadata_503`, `no_sar`, `stale_sar`,
  `no_nwps`, `far_station`, and `fusion_conflict`.

Unsupported hazards are refused explicitly (e.g. any wildfire query), never
misrouted to flood.

Note: after pulling changes that alter MCP tool signatures (e.g. the
2026-08-30 `get_flood_observation` time-window parameters), restart the
web server — long-running MCP subprocesses keep the old tool schema and
will reject the new arguments.

Overpass 502/timeout note: all OSM queries go through
`mcp_servers/overpass_client.py` — hedged requests (first/sticky mirror
serially, then the remaining mirrors concurrently; first success wins),
per-mirror cooldown (90 s for HTTP 5xx/429, 10 min for connect-level
failures), mirror-health state shared across the three OSM server
processes via `cache/overpass/mirror_state.json`, a 15-minute TTL
response cache, per-call budget caps, and query slimming (road/bridge
counts use `out tags` instead of recursing every node). If every mirror
fails, a stale cache entry (< 24 h) is served with an explicit
`overpass_stale: true` disclosure instead of failing the tool. Tune via
`OSM_OVERPASS_ENDPOINTS` / `OSM_OVERPASS_TTL_SECONDS` /
`OSM_OVERPASS_CACHE_DIR`.

## 5. What is implemented (flood reference plugin)

End-to-end pipeline, fully auditable in the run log:

1. deterministic routing + geocoding + station identity verification
2. multi-source evidence fusion (hydrology, weather, warnings, census,
   infrastructure, roads) with quality-weighted averaging and conflict
   escalation to HITL
3. Sentinel-1 SAR flood extent (GEE) with honest window extension and
   `data_gaps` accounting — a missing extent is never counted as a
   measured zero: the hazard term degrades to water-ratio-only and the
   substitution is disclosed. A SAR change footprint enters the operational
   flood map only when the gauge is at/above its official action stage or an
   active NWS flood warning corroborates it, or (in realtime mode) observed
   precipitation reaches the configured heavy-rain threshold; otherwise it is
   retained as an unverified candidate and excluded from exposure and resource
   inputs
4. areal-weighted population exposure (flood polygon ∩ census tract,
   EPSG:6933 equal-area) with a method-interval bracket, replacing
   centroid-in-polygon over-counting
5. CDC SVI vulnerability profile (population-weighted); census tracts
   filtered to the same city boundary as the flood polygon so
   numerator and denominator share one spatial scope
6. target-centered facility and route analysis with three deliberately
   separate distance/time meanings:

   - the OSM hospital/shelter search first uses a **15 km planning service
     radius** (`30 km/h × 30 minutes`) and expands to **30 km / 60 minutes**
     only when the first query succeeds but returns no candidates;
   - these circles are cross-event screening proxies, not road-network
     isochrones or live-traffic claims;
   - after a facility is found, the rescue-route layer follows the OSM road
     graph. Its displayed time is route length divided by the disclosed
     30 km/h planning speed. Flood-intersecting links are removed first; if
     that disconnects the graph, a non-avoiding accessibility-reference route
     may be shown with `route_avoids_flood = false`.

   The three policy inputs are
   `GIS_FACILITY_PLANNING_SPEED_KMH`,
   `GIS_FACILITY_PRIMARY_SERVICE_MINUTES`, and
   `GIS_FACILITY_EXTENDED_SERVICE_MINUTES`. The POI request currently keeps
   at most 10 returned facilities, and the map route uses the nearest facility
   among those returned. This is not a service-suitability guarantee. After
   selection, the road download uses a 2 km half-width origin-to-facility
   corridor, expanding once to 4 km only when the base graph is disconnected.
   Flood-avoiding and accessibility-reference routes reuse that road graph.
   This 15/30 km policy applies only to the GIS facility/map-route layer.
   The building layer is a separate default 3 km inventory query. It requests
   all returned OSM buildings instead of silently truncating at a result cap,
   then intersects them with the operational flood polygon.
7. Pareto resource optimization with **capacity-constrained demand
   assignment**: exposed census-tract demand is assigned to eligible shelters
   and hospitals by minimum-cost flow using facility capacity and straight-line
   distance. `coverage`, `unmet_demand`, p95 assignment distance, and
   `vulnerability_coverage` are recomputed from the assignment. The model does
   not claim live traffic, road closures, congestion, or exact clearance time.
   The **equity ledger**
   reports VWUN / threshold coverage gap and a threshold-free,
   population-weighted coverage concentration index for the recommended plan
   (VWUN = Σ P_exposed·(1−C)·(1+λ·SVI), where P_exposed already
   includes the flooded fraction), an efficiency–equity curve over the
   whole frontier, plus served/unmet people, tract assignments, and facility
   utilization for the operating plan
8. CDRI / EPS / DataConfidence decision indices with per-component
   substitution disclosure. CDRI carries a ±0.1 one-at-a-time
   component envelope and the exposed-population estimate carries a
   method interval; EPS and DataConfidence are point scores without
   intervals. Missing SVI/population components use a neutral 0.5
   point value with a [0,1] full-range interval and a
   `(degraded: …)` label suffix. Missing water stage uses a zero hazard;
   missing facility exposure uses neutral 0.5; missing road/route context
   uses zero, with substitutions disclosed. DataConfidence checks four
   evidence dimensions: observed flood extent, affected population,
   affected facilities, and road impact. A modeled extent or missing
   route impact is disclosed as a gap and lowers completeness.
9. adaptive HITL: parameter checkpoints auto-approve at high evidence
   quality; **safety-critical checkpoints (mobilization authorization,
   conflict escalation) time out and run unattended in the DENIAL
   direction** — an unanswered authorization never auto-approves
10. LLM synthesis layer: generic anti-hallucination rules + flood-specific
   rules injected by the plugin; fabrication/flood-stage/city-claim
   validators fall back to an evidence template on violation
11. **time-selectable queries with alignment accounting**: pass a past
    `event_date` for historical replay — the gauge is queried in the
    event window (USGS startDT/endDT), current-only sources (active
    alerts / latest forecasts & observations) are excluded rather than
    time-mismatched, SAR acquisitions must fall within [−1, +6] days of
    the event, and a per-source `time_alignment` ledger discloses every
    offset (real-time mode is relaxed: mismatches are flagged, not
    excluded)

## 6. Offline demo & tests

```bash
uv run python -m pytest -q                    # all offline
uv run python -m pytest tests/test_demo_flood_run.py -v -s   # end-to-end demo
```

`tests/test_demo_flood_run.py` runs the **entire** pipeline against a FakeMCP
fixture set (routing → fusion →
SAR extent → areal exposure → SVI → facility/route GIS → Pareto + equity
ledger → CDRI) and prints a summary. The tool-call count is intentionally not
part of the contract because optional fallback calls change it.
Swap fixtures to replay any historical event or city.

The demo rescue route prefers the local OpenStreetMap cache at
`cache/demo_road_network.json`, built with
`uv run python scripts/build_demo_road_cache.py` (network required). If it is
missing, the offline demo falls back to a synthetic grid. The cache is used
only for the Demo rescue-route display; real historical and real-time runs use
the GIS route tool and its bounded Overpass requests.

Generated runtime artifacts remain untracked. `runs/` and `static/maps/` may
be recreated freely. `cache/` is also rebuildable, but deleting it removes the
Demo's real OSM road graph and all Overpass responses, forcing network access
or a synthetic-grid fallback until the cache is rebuilt.

The operational planning chain assigns exposed census-tract demand to
eligible shelters and hospitals under their available capacity. It reports
served/unmet people and p95 assignment distance. A transfer-time range is
shown only when both planning speed bounds are configured; clearance time and
congestion delay are not estimated without independent traffic/event data.
This allocation-time estimate is separate from the map rescue-route estimate:
allocation currently uses straight-line tract-to-facility distance, whereas
the map route uses OSM road length and a fixed 30 km/h planning speed.
Resource-plan discovery is also separate: with the current
`RESOURCE_DISCOVERY_ARGUMENTS_JSON`, `get_available_resources` uses a 30 km
candidate radius (the server fallback is 50 km if the argument is omitted).
Its candidate-plan `response_time` objective is
still a disclosed 60 km/h straight-line heuristic; capacity-constrained
assignment recomputes coverage and unmet demand, but does not replace that
objective. Therefore the 30 km/h map-route assumption must not be read as a
global speed assumption for resource optimization.

The dashboard presents maintained priority profiles from
`config/plan_priorities.json` (balanced, more people served, vulnerable areas,
shorter transfer, and—when event requirements are supplied—community
compatible). Each profile selects an existing Pareto plan, and a human records
the final choice with a button; the selection is appended to the run
trajectory.

**Community-view status:** SVI weighting and equity metrics remain distinct
from community preferences. An operator can supply four event-level needs:
rescue/evacuation, safe pickup/transport, temporary shelter, and support for
children, older adults, or disabled people. The agent compares those needs to
sourced, tri-state facility capabilities and labels every Pareto plan as
confirmed, unknown, or unmet. Static facility facts live in
`config/community_facilities.json`; explicit OSM facility tags provide a small
baseline. Missing data stays unknown. There is intentionally no live
community-data feed yet; see `docs/COMMUNITY_DATA.md`.

OSM building/facility labels now prefer name, brand, operator, house name, or
street address instead of displaying a raw numeric OSM id. Verified local
names can be maintained in `config/osm_label_overrides.json`; the OSM id stays
in provenance fields. Zillow scraping is not part of the data pipeline.

`tests/test_naive_baseline.py` and `tests/test_plan_execute_baseline.py`
run the two constructed baselines over the same fixtures under the same
six fault scenarios (S0–S5) — the paper's Table 3 reproduces offline:

```bash
uv run python -m pytest tests/test_naive_baseline.py tests/test_plan_execute_baseline.py -s -q
```

The stronger `ThresholdDashboardBaseline` benchmarks the system against an
authoritative gauge-threshold dashboard rather than a silent-zero strawman;
like the other baselines it runs offline through pytest, not in the serving
loop. Held-out historical validation is declared in
`config/validation_events.json` (schema-version checked; malformed events
fail with the missing field named). Harvey is development-only. The held-out
sample has one event in each official hydrologic stratum (below action,
action, minor, moderate, major) across New Jersey, Texas, and Virginia.
Imelda and Ida/Manville are additionally marked for scripted full-Agent
artifact acceptance. The current ten checks cover run completion, station and
historical-mode identity, peak and hydrologic category, allowed extent
provenance, positive population exposure, completed optimization, a Pareto
recommendation, and at least two operator-choice scenarios. SVI and rescue
route may be produced by the run, but are **not** currently acceptance checks.
Refresh all stored peaks directly from USGS with:

```bash
uv run python scripts/validate_historical_events.py --refresh-usgs
```

Run the real-service full Agent cases with:

```bash
uv run python scripts/validate_full_agent_events.py \
  --output-dir /tmp/oasis-full-agent-validation
```

Maintained standalone scripts:

| script | purpose | network |
|---|---|---|
| `scripts/build_demo_road_cache.py` | Build the local Friendswood OSM road cache used by the offline demo | required |
| `scripts/validate_historical_events.py` | Validate the versioned event manifest; add `--refresh-usgs` to refresh official peaks | offline by default; USGS only with refresh |
| `scripts/validate_full_agent_events.py` | Run full-Agent acceptance cases and save artifacts | required |

Clean-checkout verification:

```bash
uv sync --frozen
uv run python -m pytest -q
uv run python scripts/validate_historical_events.py
```

Hydrologic results claim only USGS ingestion, peak extraction, and official
category classification. Full Agent acceptance proves only that its ten
listed checks pass; it does not prove the unchecked parts of the operational
chain and does not turn generated artifacts into ground truth. CDRI band cut
points, extent accuracy, route production, SVI content, and evacuation-time
accuracy remain unvalidated and are never counted as passed checks.

## 7. Adding a hazard

Five steps: one hazard plugin, optional engines/MCPs, plus two small shared
registration/schema edits (`skills/__init__.py` and `HazardType`). No
hazard-specific branch is added to `main.py`, `agent.py`, or the router — see
`docs/EXTENDING.md`.
