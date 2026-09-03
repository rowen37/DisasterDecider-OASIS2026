# Disaster Spatial Agent — extensible hazard decider (flood reference plugin)

Two constructed baselines — a naive catch-all dashboard and a
plan-then-execute scaffold (DORA-inspired) — plus a Skill / Verification /
Adaptive-HITL layer, restructured as a **hazard-plugin architecture**: the flood pipeline is the
reference implementation, and new hazards (wildfire, typhoon, ...) plug in
without touching any shared layer. See `docs/EXTENDING.md`.

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

## 1. Requirements

- Python 3.11+ (3.13/3.14 fine)
- uv
- API key for the model provider you choose

## 2. Install

```bash
uv sync
cp .env.example .env   # then edit .env
```

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
- Offline demo (no network, any date): the green **Demo** button, or
  `POST /api/assess {"query": "...", "demo": true}` — runs the full
  pipeline against fixtures and always produces the decision indices,
  equity ledger, and the lambda slider (with recommendation-flip
  preview).

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
   substitution is disclosed
4. areal-weighted population exposure (flood polygon ∩ census tract,
   EPSG:6933 equal-area) with a method-interval bracket, replacing
   centroid-in-polygon over-counting
5. CDC SVI vulnerability profile (population-weighted); census tracts
   filtered to the same city boundary as the flood polygon so
   numerator and denominator share one spatial scope
6. Pareto resource optimization with a **real equity objective**
   (`vulnerability_coverage`, geodesic coverage radius) and an
   **equity ledger**: VWUN / coverage gap for the recommended plan
   (VWUN = Σ P_exposed·(1−C)·(1+λ·SVI), where P_exposed already
   includes the flooded fraction), an efficiency–equity curve over the
   whole frontier, and a coverage-radius sensitivity sweep
   (R = 2/5/10 km)
7. CDRI / EPS / DataConfidence decision indices with per-component
   substitution disclosure. CDRI carries a ±0.1 one-at-a-time
   component envelope and the exposed-population estimate carries a
   method interval; EPS and DataConfidence are point scores without
   intervals. Missing SVI/population components use a neutral 0.5
   point value with a [0,1] full-range interval and a
   `(degraded: …)` label suffix; missing water-ratio / facility /
   road / route inputs enter as 0 and **every such substitution is
   recorded in the substitution ledger**
8. adaptive HITL: parameter checkpoints auto-approve at high evidence
   quality; **safety-critical checkpoints (mobilization authorization,
   conflict escalation) time out and run unattended in the DENIAL
   direction** — an unanswered authorization never auto-approves
9. LLM synthesis layer: generic anti-hallucination rules + flood-specific
   rules injected by the plugin; fabrication/flood-stage/city-claim
   validators fall back to an evidence template on violation
10. **time-selectable queries with alignment accounting**: pass a past
    `event_date` for historical replay — the gauge is queried in the
    event window (USGS startDT/endDT), current-only sources (active
    alerts / latest forecasts & observations) are excluded rather than
    time-mismatched, SAR acquisitions must fall within [−1, +6] days of
    the event, and a per-source `time_alignment` ledger discloses every
    offset (real-time mode is relaxed: mismatches are flagged, not
    excluded)

## 6. Offline demo & tests

```bash
uv run python -m pytest tests/ -v             # all offline
uv run python -m pytest tests/test_demo_flood_run.py -v -s   # end-to-end demo
```

`tests/test_demo_flood_run.py` runs the **entire** pipeline against a
FakeMCP fixture set (24 tool calls: routing → fusion → SAR extent → areal
exposure → SVI → Pareto + equity ledger → CDRI) and prints a summary.
Swap fixtures to replay any historical event or city.

The demo rescue route prefers a real OSM road network cache
(`cache/demo_road_network.json`, built once with
`uv run python scripts/build_demo_road_cache.py`; needs network). Without
the cache it deterministically falls back to a synthetic street grid, so
the demo always runs offline.

`tests/test_naive_baseline.py` and `tests/test_plan_execute_baseline.py`
run the two constructed baselines over the same fixtures under the same
six fault scenarios (S0–S5) — the paper's Table 3 reproduces offline:

```bash
uv run python -m pytest tests/test_naive_baseline.py tests/test_plan_execute_baseline.py -s -q
```

## 7. Adding a hazard

Five steps, zero shared-layer edits — see `docs/EXTENDING.md`.
