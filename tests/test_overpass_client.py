# tests/test_overpass_client.py
#
# Fault-tolerance tests for the shared Overpass client (fully offline,
# httpx.MockTransport): mirror rotation / cooldown and stickiness /
# TTL cache / no rotation on permanent errors / budget cap.

import asyncio
import sys
import os

import httpx
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcp_servers",
))

import overpass_client as oc          # noqa: E402


EPS = [
    "https://mirror-a.test/api/interpreter",
    "https://mirror-b.test/api/interpreter",
    "https://mirror-c.test/api/interpreter",
]

OK_BODY = {"elements": [{"type": "node", "id": 1, "lat": 29.5, "lon": -95.2}]}


@pytest.fixture(autouse=True)
def _clean_state():
    oc.reset_state()
    yield
    oc.reset_state()


def _handler_script(script: dict[str, httpx.Response | Exception], log: list):
    """Return the scripted response/exception per host; unlisted hosts get 200 OK_BODY."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        log.append(host)
        action = script.get(host)
        if isinstance(action, Exception):
            raise action
        if action is not None:
            return action
        return httpx.Response(200, json=OK_BODY)

    return handler


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _sync_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------
# 1. 502 -> rotate to the next mirror, which succeeds
# ------------------------------------------------------------------
def test_mirror_rotation_on_502(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.Response(502, text="Bad Gateway")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            data, diag = await oc.post_overpass_async(
                "[out:json];node(1);out;",
                endpoints=EPS,
                client=client,
                cache_dir=tmp_path,
                ttl_s=60,
                server_timeout_s=5,
                total_budget_s=10,
            )
            return data, diag
        finally:
            await client.aclose()

    data, diag = asyncio.run(scenario())
    assert data == OK_BODY
    # Hedged semantics: after A 502s, B/C race concurrently, first success wins
    assert diag["endpoint"] in (
        "https://mirror-b.test/api/interpreter",
        "https://mirror-c.test/api/interpreter",
    )
    assert diag["cache_hit"] is False
    # First mirror hit exactly once (502 -> cooldown, no same-mirror retry)
    assert log[0] == "mirror-a.test"
    assert log.count("mirror-a.test") == 1
    assert diag["endpoint"].split("//")[1].split("/")[0] in log
    # Failed mirror enters cooldown
    assert oc._cooldown_until["https://mirror-a.test/api/interpreter"] > 0


# ------------------------------------------------------------------
# 2. Cooldown + stickiness: failed mirrors are skipped, previous winner goes first
# ------------------------------------------------------------------
def test_cooldown_and_sticky_ordering(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.Response(502, text="Bad Gateway")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            _d1, diag1 = await oc.post_overpass_async(
                "query-one", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
            # Second call (different query): A is cooling down -> start
            # from the previous winner (sticky)
            log.clear()
            _data2, diag2 = await oc.post_overpass_async(
                "query-two", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
            return diag1, diag2
        finally:
            await client.aclose()

    diag1, diag2 = asyncio.run(scenario())
    assert diag2["endpoint"] == diag1["endpoint"], (
        "second call must be served by the sticky winner of the first"
    )
    assert "mirror-a.test" not in log, (
        f"cooling mirror must be skipped, got order: {log}"
    )
    assert log[0] == diag1["endpoint"].split("//")[1].split("/")[0]


# ------------------------------------------------------------------
# 3. TTL cache: a repeated query makes zero network requests
# ------------------------------------------------------------------
def test_cache_hit_makes_no_request(tmp_path):
    log: list[str] = []
    handler = _handler_script({}, log)

    async def scenario():
        client = _client(handler)
        try:
            d1, diag1 = await oc.post_overpass_async(
                "cached-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
            log.clear()
            d2, diag2 = await oc.post_overpass_async(
                "cached-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
            return d1, diag1, d2, diag2
        finally:
            await client.aclose()

    d1, diag1, d2, diag2 = asyncio.run(scenario())
    assert diag1["cache_hit"] is False
    assert diag2["cache_hit"] is True
    assert diag2["endpoint"] == "cache"
    assert d2 == d1 == OK_BODY
    assert log == [], "cache hit must not touch the network"


# ------------------------------------------------------------------
# 4. Permanent error (400 bad query syntax): fail immediately, no mirror rotation
# ------------------------------------------------------------------
def test_permanent_error_fails_fast(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.Response(400, text="parse error: bad qstr")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            await oc.post_overpass_async(
                "[out:json]; this is not valid overpass;",
                endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
        except oc.OverpassError as exc:
            return exc
        finally:
            await client.aclose()

    exc = asyncio.run(scenario())
    assert exc is not None
    assert "rejected" in str(exc)
    assert "parse error" in str(exc)
    assert log == ["mirror-a.test"], (
        "a syntactically invalid query must not be retried on other mirrors"
    )


# ------------------------------------------------------------------
# 5. All mirrors 502 -> OverpassError with the full attempt log
# ------------------------------------------------------------------
def test_all_endpoints_fail_raises(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {host: httpx.Response(502, text="Bad Gateway")
         for host in ("mirror-a.test", "mirror-b.test", "mirror-c.test")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            await oc.post_overpass_async(
                "doomed-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
        except oc.OverpassError as exc:
            return exc
        finally:
            await client.aclose()

    exc = asyncio.run(scenario())
    assert exc is not None
    assert {a["endpoint"] for a in exc.attempts} == set(EPS)
    assert len(log) == 3, "each mirror tried exactly once (no same-mirror retry on 502)"


# ------------------------------------------------------------------
# 6. Budget cap: exhausted budget fails immediately (no wasted requests)
# ------------------------------------------------------------------
def test_budget_exhaustion(tmp_path):
    log: list[str] = []
    handler = _handler_script({}, log)

    async def scenario():
        client = _client(handler)
        try:
            await oc.post_overpass_async(
                "budget-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=0.0,
            )
        except oc.OverpassError as exc:
            return exc
        finally:
            await client.aclose()

    exc = asyncio.run(scenario())
    assert exc is not None
    assert "budget" in str(exc)
    assert log == [], "exhausted budget must fail before any request"


# ------------------------------------------------------------------
# 7. Transport error: one backoff retry on the same mirror, then rotate
# ------------------------------------------------------------------
def test_transport_error_retries_then_rotates(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.ConnectError("DNS cold start")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            _data, diag = await oc.post_overpass_async(
                "dns-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=30,
            )
            return diag
        finally:
            await client.aclose()

    diag = asyncio.run(scenario())
    assert diag["endpoint"] in (
        "https://mirror-b.test/api/interpreter",
        "https://mirror-c.test/api/interpreter",
    )
    # A is tried twice (same-endpoint retry for a transient connect
    # error), then B/C are hedged
    assert log[:2] == ["mirror-a.test", "mirror-a.test"]
    assert set(log[2:]) <= {"mirror-b.test", "mirror-c.test"}


# ------------------------------------------------------------------
# 8. Sync variant: same rotation semantics
# ------------------------------------------------------------------
def test_sync_rotation_and_cache(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.Response(502, text="Bad Gateway")},
        log,
    )
    client = _sync_client(handler)
    try:
        data, diag = oc.post_overpass_sync(
            "sync-query", endpoints=EPS, client=client,
            cache_dir=tmp_path, ttl_s=60,
            server_timeout_s=5, total_budget_s=10,
        )
    finally:
        client.close()
    assert data == OK_BODY
    assert diag["endpoint"] in (
        "https://mirror-b.test/api/interpreter",
        "https://mirror-c.test/api/interpreter",
    )

    # Cache hit (the sync path also makes zero network requests)
    log.clear()
    client2 = _sync_client(handler)
    try:
        _d2, diag2 = oc.post_overpass_sync(
            "sync-query", endpoints=EPS, client=client2,
            cache_dir=tmp_path, ttl_s=60,
            server_timeout_s=5, total_budget_s=10,
        )
    finally:
        client2.close()
    assert diag2["cache_hit"] is True
    assert log == []


# ------------------------------------------------------------------
# 9. All mirrors fail -> stale cache within 24h served as fallback
#    (explicitly flagged stale)
# ------------------------------------------------------------------
def test_stale_cache_fallback(tmp_path):
    log: list[str] = []
    # Step 1: all mirrors succeed, populating the cache
    ok_handler = _handler_script({}, log)
    client = _client(ok_handler)
    try:
        asyncio.run(oc.post_overpass_async(
            "stale-demo-query", endpoints=EPS, client=client,
            cache_dir=tmp_path, ttl_s=60,
            server_timeout_s=5, total_budget_s=10,
        ))
    finally:
        asyncio.run(client.aclose())

    # Backdate the cache timestamp by 2h (past the 60s TTL, within the 24h stale window)
    import json as _json
    cache_files = [
        p for p in tmp_path.glob("*.json")
        if p.name != oc._STATE_FILE   # exclude the shared mirror-state file
    ]
    cache_file = cache_files[0]
    payload = _json.loads(cache_file.read_text())
    payload["stored_at"] -= 2 * 3600
    cache_file.write_text(_json.dumps(payload))

    # Step 2: all mirrors 502 -> the stale cache is replayed instead of raising
    log.clear()
    fail_handler = _handler_script(
        {host: httpx.Response(502, text="Bad Gateway")
         for host in ("mirror-a.test", "mirror-b.test", "mirror-c.test")},
        log,
    )
    client2 = _client(fail_handler)

    async def scenario():
        try:
            return await oc.post_overpass_async(
                "stale-demo-query", endpoints=EPS, client=client2,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
        finally:
            await client2.aclose()

    data, diag = asyncio.run(scenario())
    assert data == OK_BODY
    assert diag["stale"] is True
    assert diag["endpoint"] == "stale-cache"
    assert diag["cache_age_s"] > 7000   # ~2h, age disclosed honestly
    assert len(log) == 3                # all mirrors really tried before the fallback


# ------------------------------------------------------------------
# 11. Mirror health state shared across processes: a new process starts
#     directly from the healthy mirrors
#     (the three OSM servers run as independent processes; cooldown/
#     stickiness learned by one process reaches the others via
#     mirror_state.json in the cache directory)
# ------------------------------------------------------------------
def test_shared_mirror_state_across_processes(tmp_path):
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.Response(502, text="Bad Gateway")},
        log,
    )

    async def call(query_name):
        client = _client(handler)
        try:
            return await oc.post_overpass_async(
                query_name, endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=10,
            )
        finally:
            await client.aclose()

    # Process 1: A 502s -> cooldown, one of the hedged B/C wins
    # (state written to the shared file)
    _d1, diag1 = asyncio.run(call("proc-1-query"))
    winner_host = diag1["endpoint"].split("//")[1].split("/")[0]
    assert log[0] == "mirror-a.test"

    # Process 2: reset in-process state (= a fresh MCP server process), same cache dir
    oc.reset_state()
    log.clear()
    _d2, diag2 = asyncio.run(call("proc-2-query"))
    # Shared state applies: cooling A sinks to the back, process 1's
    # winner (shared stickiness) goes first — the new process no longer
    # probes the dead mirror
    assert log[0] == winner_host, (
        f"shared sticky must be tried first, got: {log}"
    )
    assert diag2["endpoint"] == diag1["endpoint"]
    assert oc._state_path(tmp_path).exists()


# ------------------------------------------------------------------
# 12. Two connect-level errors -> long cooldown (~600s) and sunk to the
#     back of the ordering
# ------------------------------------------------------------------
def test_connect_failure_long_cooldown(tmp_path):
    import time as _time
    log: list[str] = []
    handler = _handler_script(
        {"mirror-a.test": httpx.ConnectError("SSL certificate mismatch")},
        log,
    )

    async def scenario():
        client = _client(handler)
        try:
            await oc.post_overpass_async(
                "ssl-dead-query", endpoints=EPS, client=client,
                cache_dir=tmp_path, ttl_s=60,
                server_timeout_s=5, total_budget_s=30,
            )
        finally:
            await client.aclose()

    asyncio.run(scenario())
    # Two connect failures on A -> long cooldown (infrastructure-level
    # failure, ~600s rather than 90s)
    until = oc._cooldown_until["https://mirror-a.test/api/interpreter"]
    assert until - _time.time() > 500, (
        "connect-failure cooldown must be long (~600s), got "
        f"{until - _time.time():.0f}s"
    )
    # A cooling mirror sinks to the back of the candidate order: tried
    # last on the next call
    order = oc._ordered_endpoints(EPS)
    assert order[-1] == "https://mirror-a.test/api/interpreter"
    assert log[:2] == ["mirror-a.test", "mirror-a.test"]
    assert set(log[2:]) <= {"mirror-b.test", "mirror-c.test"}
