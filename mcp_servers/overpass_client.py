"""Shared Overpass client for the three OSM MCP servers.

Addresses the frequent 502/504/429 failures of public Overpass instances:

1. Mirror rotation + cooldown + sticky endpoint
   A mirror failure (429/5xx/timeout/transport error) puts it into a
   cooldown (default 90s) and the next mirror is used immediately; an
   endpoint that succeeded becomes sticky and is tried first next time.
   No more hammering the same instance three times in a row.
2. Error classification
   429/500/502/503/504/timeout/transport error = retryable (switch
   mirror); 400 = bad query syntax (switching mirrors won't help) ->
   fail immediately with a response-body summary. A 429 Retry-After is
   honored and converted into that mirror's cooldown.
3. TTL disk cache
   The same query (place + radius + kinds) within the TTL is served
   from cache with no network call. OSM facility data changes on the
   order of days, so the default 15 minutes is fresh enough; repeated
   runs and overlapping skill/GIS-layer queries no longer burn quota.
4. Total budget cap
   However many mirrors are rotated, a whole call stays within
   total_budget_s (default server_timeout + 20s); retry storms cannot
   eat the caller's budget.
5. Client timeout > server query timeout
   For a query with [timeout:N], the client timeout is N+15s - the
   client giving up before the server only wastes the server-side run
   and its quota.

Environment variables:
  OSM_OVERPASS_API_URL       primary endpoint (legacy config; first in the mirror list)
  OSM_OVERPASS_ENDPOINTS     comma-separated full mirror list (overrides the default)
  OSM_OVERPASS_TTL_SECONDS   cache TTL, default 900; 0 disables the cache
  OSM_OVERPASS_CACHE_DIR     cache dir, default <project>/cache/overpass
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    # overpass.osm.jp serves a TLS hostname mismatch (dead endpoint);
    # add it back explicitly via OSM_OVERPASS_ENDPOINTS only if ever needed.
]

USER_AGENT = "DisasterAgent/1.0 (qiwenb@design.upenn.edu)"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
COOLDOWN_SECONDS = 90.0        # HTTP 5xx/429: overload usually recovers within minutes
CONNECT_FAIL_COOLDOWN = 600.0  # DNS/SSL failures: infrastructure-level outage, slow to recover
MAX_COOLDOWN_SECONDS = 600.0
# Stale-cache fallback limit (seconds) when all mirrors fail: in disaster
# response, day-old data with an explicit stale flag beats failing outright.
STALE_CACHE_LIMIT_S = 24 * 3600.0
# Max mirrors queried concurrently during hedging (stay polite to public instances)
MAX_HEDGE_WIDTH = 3


class OverpassError(RuntimeError):
    """All mirrors failed (or the query itself is invalid). The message
    includes a per-mirror attempt log."""

    def __init__(self, message: str, attempts: list[dict[str, str]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


# ---------------------------------------------------------------------
# Rotation state
# In-process dict + a shared state file (mirror_state.json) in the cache
# dir: the three OSM servers are separate processes, so mirror health
# learned by one process ("which mirrors are down / which is healthy")
# is written to the shared file and other processes start from the
# healthy mirror on their next call instead of bumping into dead ones.
# Writes are atomic (temp file + rename); with concurrent processes the
# last writer wins, which is harmless.
# ---------------------------------------------------------------------
_cooldown_until: dict[str, float] = {}
_sticky_endpoint: str | None = None
_STATE_FILE = "mirror_state.json"


def reset_state() -> None:
    """Clear in-process rotation/cooldown state (for tests)."""
    _cooldown_until.clear()
    global _sticky_endpoint
    _sticky_endpoint = None


def _state_path(cache_dir: Path) -> Path:
    return cache_dir / _STATE_FILE


def _load_shared_state(cache_dir: Path) -> tuple[dict[str, float], str | None]:
    """Load cross-process shared state; expired cooldown entries are dropped on load."""
    try:
        raw = json.loads(_state_path(cache_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    until = {
        str(ep): float(t)
        for ep, t in (raw.get("cooldown_until") or {}).items()
        if float(t) > time.time()
    }
    return until, raw.get("sticky")


def _store_shared_state(cache_dir: Path) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = _state_path(cache_dir)
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {"cooldown_until": _cooldown_until, "sticky": _sticky_endpoint}
            ),
            encoding="utf-8",
        )
        os.replace(tmp, target)
    except OSError:
        pass  # state sharing is an accelerator; a failed write does not affect correctness


def configured_endpoints(explicit: list[str] | None = None) -> list[str]:
    """Endpoint list: explicit arg > OSM_OVERPASS_ENDPOINTS > default (primary first)."""
    if explicit:
        eps = list(explicit)
    else:
        raw = os.environ.get("OSM_OVERPASS_ENDPOINTS", "")
        if raw.strip():
            eps = [e.strip() for e in raw.split(",") if e.strip()]
        else:
            primary = os.environ.get(
                "OSM_OVERPASS_API_URL", DEFAULT_ENDPOINTS[0]
            )
            eps = [primary] + [
                e for e in DEFAULT_ENDPOINTS[1:] if e != primary
            ]
    # Dedupe, preserving order
    seen: set[str] = set()
    return [e for e in eps if not (e in seen or seen.add(e))]


def _ordered_endpoints(
    eps: list[str], cache_dir: Path | None = None
) -> list[str]:
    """Sticky endpoint first; cooling endpoints move to the back (if all
    are cooling, keep the original order).

    Cooldown/sticky state is merged from process memory and the shared
    file in the cache dir, so mirror health learned by other OSM server
    processes takes effect here.
    """
    global _sticky_endpoint
    shared_until, shared_sticky = (
        _load_shared_state(cache_dir) if cache_dir is not None else ({}, None)
    )
    merged = dict(shared_until)
    for ep, t in _cooldown_until.items():
        merged[ep] = max(merged.get(ep, 0.0), t)
    sticky = _sticky_endpoint or shared_sticky

    ordered = list(eps)
    if sticky in ordered:
        ordered.remove(sticky)
        ordered.insert(0, sticky)
    now = time.time()
    live = [e for e in ordered if merged.get(e, 0.0) <= now]
    cooling = [e for e in ordered if merged.get(e, 0.0) > now]
    return live + cooling if live else ordered


def _cooldown(endpoint: str, seconds: float = COOLDOWN_SECONDS) -> None:
    _cooldown_until[endpoint] = time.time() + min(
        max(seconds, 1.0), MAX_COOLDOWN_SECONDS
    )


# ---------------------------------------------------------------------
# TTL disk cache
# ---------------------------------------------------------------------
def _default_cache_dir() -> Path:
    override = os.environ.get("OSM_OVERPASS_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "cache" / "overpass"


def _ttl_seconds(explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, explicit)
    try:
        return max(0.0, float(os.environ.get("OSM_OVERPASS_TTL_SECONDS", "900")))
    except ValueError:
        return 900.0


def _cache_file(query: str, cache_dir: Path) -> Path:
    key = hashlib.sha1(query.strip().encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _cache_load(
    query: str, cache_dir: Path, ttl_s: float
) -> tuple[dict[str, Any], float] | None:
    if ttl_s <= 0:
        return None
    path = _cache_file(query, cache_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stored_at = float(raw.get("stored_at", 0.0))
    age = time.time() - stored_at
    if age > ttl_s:
        return None
    data = raw.get("data")
    if not isinstance(data, dict):
        return None
    return data, age


def _cache_store(query: str, data: dict[str, Any], cache_dir: Path) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file then replace, so concurrent
        # processes never read a partial JSON
        target = _cache_file(query, cache_dir)
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"stored_at": time.time(), "data": data}),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        _prune_cache(cache_dir)
    except OSError:
        # A cache write failure does not affect query results (the cache is only an accelerator)
        pass


def _prune_cache(cache_dir: Path, limit: int = 256) -> None:
    """Prune the oldest cache files when there are too many (avoid unbounded
    growth in long-running processes).

    mirror_state.json is shared mirror health state, not a cache entry,
    and is never pruned.
    """
    try:
        files = sorted(
            (
                p for p in cache_dir.glob("*.json")
                if p.name != _STATE_FILE
            ),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in files[: max(0, len(files) - limit)]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------
# Request core: single attempt + error classification (shared async/sync semantics)
# ---------------------------------------------------------------------
def _classify(resp: httpx.Response) -> tuple[bool, str]:
    """Return (retryable, summary). Retryable = switching mirrors may help."""
    if resp.status_code in RETRYABLE_STATUS:
        return True, f"HTTP {resp.status_code}"
    if resp.status_code >= 400:
        # 400 etc.: the query itself is bad; every mirror would reject it
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    return False, ""


class _Permanent(Exception):
    """The query was rejected by a mirror (400) - abort all retries."""


async def _attempt_async(
    client: httpx.AsyncClient, endpoint: str, query: str,
    attempt_timeout: float,
) -> httpx.Response:
    """One POST to a single mirror. Returns the Response (classified by
    the caller); raises on failure."""
    return await client.post(
        endpoint,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=attempt_timeout,
    )


def _attempt_sync(
    client: httpx.Client, endpoint: str, query: str,
    attempt_timeout: float,
) -> httpx.Response:
    return client.post(
        endpoint,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=attempt_timeout,
    )


def _parse_response(resp: httpx.Response) -> dict:
    """Classify one response. Success returns {"data": ...}; a retryable
    failure raises RuntimeError; an invalid query raises _Permanent."""
    retryable, summary = _classify(resp)
    if not retryable and summary:
        raise _Permanent(summary)
    if retryable:
        raise RuntimeError(summary)
    try:
        return {"data": resp.json()}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON body: {exc}") from exc


def _record_failure(
    endpoint: str, exc: Exception, attempts: list[dict[str, str]]
) -> None:
    """Record one failure in the attempts log and put the mirror into cooldown."""
    attempts.append({"endpoint": endpoint, "error": str(exc)})
    if isinstance(exc, RuntimeError):
        _cooldown(endpoint, COOLDOWN_SECONDS)
    else:
        # connection-level failure: infrastructure-level outage, long cooldown
        _cooldown(endpoint, CONNECT_FAIL_COOLDOWN)


def _stale_or_raise(
    query: str, cdir: Path, attempts: list[dict[str, str]], cause: str
) -> tuple[dict, dict]:
    """Last resort after all mirrors fail: the 24h stale cache (explicitly
    flagged stale); raise OverpassError if absent."""
    stale = _cache_load(query, cdir, STALE_CACHE_LIMIT_S)
    if stale is not None:
        data, age = stale
        return data, {
            "endpoint": "stale-cache",
            "cache_hit": True,
            "stale": True,
            "cache_age_s": round(age, 1),
            "attempts": attempts,
        }
    raise OverpassError(f"{cause}; attempts: {attempts}", attempts)


def _success(
    endpoint: str, data: dict, attempts: list[dict[str, str]]
) -> tuple[dict, dict]:
    global _sticky_endpoint
    _sticky_endpoint = endpoint
    return data, {
        "endpoint": endpoint,
        "cache_hit": False,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------
# Async entry: serial first shot + concurrent hedged requests
# ---------------------------------------------------------------------
async def post_overpass_async(
    query: str,
    *,
    server_timeout_s: float = 60.0,
    total_budget_s: float | None = None,
    endpoints: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
    cache_dir: Path | None = None,
    ttl_s: float | None = None,
) -> tuple[dict, dict]:
    """POST a query to Overpass (async). Returns (data, diagnostics).

    Diagnostics: {"endpoint", "cache_hit", "stale", "cache_age_s",
    "attempts"}. Raises OverpassError when everything fails and no
    stale cache exists.

    Strategy: the first-priority (sticky) mirror is tried once serially
    (zero extra load on the happy path); on a retryable failure, the
    remaining live mirrors are hedged concurrently - public mirrors
    "fail slowly" (running the full server-side [timeout:N] before
    returning 504), so serial rotation would exhaust the budget before
    reaching a healthy mirror. Hedging reduces worst-case latency from
    sum(mirror timeouts) to max(first healthy mirror's latency).
    Read-only queries; concurrency-safe.
    """
    cdir = cache_dir or _default_cache_dir()
    ttl = _ttl_seconds(ttl_s)
    cached = _cache_load(query, cdir, ttl)
    if cached is not None:
        data, age = cached
        return data, {
            "endpoint": "cache", "cache_hit": True,
            "cache_age_s": round(age, 1), "attempts": [],
        }

    eps = _ordered_endpoints(configured_endpoints(endpoints), cdir)
    budget = (
        server_timeout_s + 20.0 if total_budget_s is None else total_budget_s
    )
    deadline = time.monotonic() + budget
    attempts: list[dict[str, str]] = []
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(server_timeout_s + 15.0)
        )

    def remaining() -> float:
        return deadline - time.monotonic()

    def attempt_timeout() -> float:
        # Clamp a single attempt to the remaining budget: a hung request
        # must not eat the total budget
        return min(server_timeout_s + 15.0, max(5.0, remaining()))

    permanent_failure = False
    try:
        if not eps:
            raise OverpassError("No Overpass endpoints configured", attempts)

        # -- Phase 1: try the first mirror serially (one fast retry for
        # transient connection-level errors) --
        first, rest = eps[0], eps[1:]
        data: dict | None = None
        for try_no in range(2):
            if remaining() <= 0:
                break
            try:
                resp = await _attempt_async(
                    client, first, query, attempt_timeout()
                )
                data = _parse_response(resp)["data"]
                break
            except _Permanent:
                raise
            except RuntimeError as exc:
                _record_failure(first, exc, attempts)
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                summary = f"{type(exc).__name__}: {exc}"
                attempts.append({"endpoint": first, "error": summary})
                if try_no == 0:
                    await asyncio.sleep(0.4 + random.uniform(0, 0.3))
                    continue
                _cooldown(first, CONNECT_FAIL_COOLDOWN)
                break
            except httpx.HTTPError as exc:
                _record_failure(first, exc, attempts)
                break

        if data is not None:
            _cache_store(query, data, cdir)
            return _success(first, data, attempts)

        # -- Phase 2: hedge the remaining mirrors concurrently; first success wins --
        hedge = rest[:MAX_HEDGE_WIDTH]
        winner = None
        if hedge and remaining() > 0:
            timeout = attempt_timeout()
            tasks = {
                asyncio.ensure_future(
                    _attempt_async(client, ep, query, timeout)
                ): ep
                for ep in hedge
            }
            try:
                pending = set(tasks)
                while pending and winner is None:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        ep = tasks[task]
                        try:
                            resp = task.result()
                            result = _parse_response(resp)
                        except _Permanent:
                            permanent_failure = True
                            raise
                        except (RuntimeError, httpx.HTTPError) as exc:
                            _record_failure(ep, exc, attempts)
                        else:
                            winner = (ep, result["data"])
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()

        if winner is not None:
            ep, wdata = winner
            _cache_store(query, wdata, cdir)
            return _success(ep, wdata, attempts)

        cause = (
            f"Overpass total budget {budget:.0f}s exhausted"
            if remaining() <= 0 else "All Overpass endpoints failed"
        )
        raise OverpassError(cause, attempts)
    except _Permanent as exc:
        attempts.append({"endpoint": "?", "error": str(exc)})
        raise OverpassError(
            f"Overpass rejected the query: {exc}", attempts
        )
    except OverpassError as exc:
        if not permanent_failure:
            return _stale_or_raise(query, cdir, attempts, str(exc))
        raise
    finally:
        _store_shared_state(cdir)
        if own_client:
            await client.aclose()


# ---------------------------------------------------------------------
# Sync entry: same semantics (concurrent hedging via a thread pool)
# ---------------------------------------------------------------------
def post_overpass_sync(
    query: str,
    *,
    server_timeout_s: float = 60.0,
    total_budget_s: float | None = None,
    endpoints: list[str] | None = None,
    client: httpx.Client | None = None,
    cache_dir: Path | None = None,
    ttl_s: float | None = None,
) -> tuple[dict, dict]:
    cdir = cache_dir or _default_cache_dir()
    ttl = _ttl_seconds(ttl_s)
    cached = _cache_load(query, cdir, ttl)
    if cached is not None:
        data, age = cached
        return data, {
            "endpoint": "cache", "cache_hit": True,
            "cache_age_s": round(age, 1), "attempts": [],
        }

    eps = _ordered_endpoints(configured_endpoints(endpoints), cdir)
    budget = (
        server_timeout_s + 20.0 if total_budget_s is None else total_budget_s
    )
    deadline = time.monotonic() + budget
    attempts: list[dict[str, str]] = []
    own_client = client is None
    if own_client:
        client = httpx.Client(
            timeout=httpx.Timeout(server_timeout_s + 15.0)
        )

    def remaining() -> float:
        return deadline - time.monotonic()

    def attempt_timeout() -> float:
        return min(server_timeout_s + 15.0, max(5.0, remaining()))

    permanent_failure = False
    try:
        if not eps:
            raise OverpassError("No Overpass endpoints configured", attempts)

        first, rest = eps[0], eps[1:]
        data = None
        for try_no in range(2):
            if remaining() <= 0:
                break
            try:
                resp = _attempt_sync(client, first, query, attempt_timeout())
                data = _parse_response(resp)["data"]
                break
            except _Permanent:
                raise
            except RuntimeError as exc:
                _record_failure(first, exc, attempts)
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                summary = f"{type(exc).__name__}: {exc}"
                attempts.append({"endpoint": first, "error": summary})
                if try_no == 0:
                    time.sleep(0.4 + random.uniform(0, 0.3))
                    continue
                _cooldown(first, CONNECT_FAIL_COOLDOWN)
                break
            except httpx.HTTPError as exc:
                _record_failure(first, exc, attempts)
                break

        if data is not None:
            _cache_store(query, data, cdir)
            return _success(first, data, attempts)

        hedge = rest[:MAX_HEDGE_WIDTH]
        winner = None
        if hedge and remaining() > 0:
            timeout = attempt_timeout()
            with ThreadPoolExecutor(
                max_workers=len(hedge), thread_name_prefix="overpass-hedge"
            ) as pool:
                futures = {
                    pool.submit(_attempt_sync, client, ep, query, timeout): ep
                    for ep in hedge
                }
                pending = set(futures)
                while pending and winner is None:
                    done, pending = wait(
                        pending, return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        ep = futures[fut]
                        try:
                            resp = fut.result()
                            result = _parse_response(resp)
                        except _Permanent:
                            permanent_failure = True
                            raise
                        except (RuntimeError, httpx.HTTPError) as exc:
                            _record_failure(ep, exc, attempts)
                        else:
                            winner = (ep, result["data"])
                # Wrap up: wait for the remaining requests (bounded by the
                # remaining budget) and log failures
                if pending:
                    wait(pending, timeout=max(0.0, remaining()))
                    for fut in pending:
                        ep = futures[fut]
                        if fut.done() and not fut.cancelled():
                            exc = fut.exception()
                            if exc is not None:
                                _record_failure(ep, exc, attempts)

        if winner is not None:
            ep, wdata = winner
            _cache_store(query, wdata, cdir)
            return _success(ep, wdata, attempts)

        cause = (
            f"Overpass total budget {budget:.0f}s exhausted"
            if remaining() <= 0 else "All Overpass endpoints failed"
        )
        raise OverpassError(cause, attempts)
    except _Permanent as exc:
        attempts.append({"endpoint": "?", "error": str(exc)})
        raise OverpassError(
            f"Overpass rejected the query: {exc}", attempts
        )
    except OverpassError as exc:
        if not permanent_failure:
            return _stale_or_raise(query, cdir, attempts, str(exc))
        raise
    finally:
        _store_shared_state(cdir)
        if own_client:
            client.close()
