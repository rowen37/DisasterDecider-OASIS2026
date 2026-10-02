"""FusionSourcesSkill — atomic skill for multi-source evidence collection.

Fans out configured fusion sources (NWS alerts, forecast, census, OSM,
roads...) via MCP in parallel and returns raw observation payloads for
the Fusion Engine.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from ..models import Location
from .fusion import _render_template


class FusionSourcesSkill:

    name = "flood_fusion_sources"

    # These configured connectors call current/latest endpoints.  Some
    # upstream products have limited archives, but these tool calls do not
    # accept an event window.  In historical replay they are skipped with
    # an audit record: today's alerts or forecast must never be used to
    # explain a past event.
    CURRENT_ONLY_SOURCE_TYPES = {
        "warning",
        "forecast",
        "weather",
        "precipitation",
    }

    def __init__(
        self,
        state,
        verifier,
        mcp,
        logger,
        flood_analysis_radius_km: float,
        source_configs: list[dict[str, Any]],
    ):
        self.state = state
        self.verifier = verifier
        self.mcp = mcp
        self.logger = logger
        self.flood_analysis_radius_km = flood_analysis_radius_km
        self.source_configs = source_configs
        self.skipped_historical_sources: list[dict[str, Any]] = []

    async def collect(
        self,
        target: str,
        station_id: str,
        location: Location,
        event_date: str | None = None,
        mode: str = "realtime",
    ) -> list[dict[str, Any]]:

        # Event time window: [00:00, +24h) UTC on event_date. Hydrology
        # templates use it for USGS IV startDT/endDT; SAR sources use
        # observation_date.
        event_start = f"{event_date}T00:00:00Z" if event_date else ""
        event_end_dt = (
            datetime.fromisoformat(f"{event_date}T00:00:00+00:00")
            + timedelta(days=1)
            if event_date
            else None
        )
        event_end = (
            event_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if event_end_dt
            else ""
        )

        context = {
            "target": target,
            "station_id": station_id,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "flood_analysis_radius_km": self.flood_analysis_radius_km,
            "warning_radius_km": float(
                os.getenv("NWS_WARNING_RADIUS_KM", "25")
            ),
            # event_date may be injected by the caller (historical replay); defaults to today UTC.
            "event_date": event_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "event_start": event_start,
            "event_end": event_end,
        }

        async def _collect_one(
            source_config: dict[str, Any],
        ) -> dict[str, Any] | None:

            name = source_config.get("name")
            enabled = source_config.get("enabled", True)
            tool_name = source_config.get("tool")
            source_type = str(source_config.get("source_type", "unknown"))

            if not enabled:
                print(f"[DEBUG] Skipping {name} (disabled)", file=sys.stderr)
                return None

            if (
                mode == "historical"
                and source_type in self.CURRENT_ONLY_SOURCE_TYPES
            ):
                self.skipped_historical_sources.append(
                    {
                        "source": name,
                        "source_type": source_type,
                        "reason": (
                            "configured connector uses an active/latest "
                            "endpoint and does not accept the event window; "
                            "excluded for time alignment"
                        ),
                    }
                )
                self.state.log(
                    "skipped_historical_source",
                    source=name,
                    source_type=source_type,
                    reason="connector_not_event_time_capable",
                )
                return None

            if not tool_name:
                raise RuntimeError(
                    "Flood fusion source is missing "
                    "the MCP tool name."
                )

            arguments_template = source_config.get("arguments",{},)
            arguments = _render_template(arguments_template,context,)

            self.logger.log(
                self.state.run_id,
                self.name,
                "tool_call",
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "purpose": "multi_source_fusion",
                },
            )

            # Per-source timeout/retry budget: heavy sources (SAR
            # extraction can take 150s) override the global 60s so the
            # client does not time out before the server.
            _source_timeout = source_config.get("timeout_s")
            _call_kwargs = (
                {"timeout": float(_source_timeout)}
                if _source_timeout is not None
                else {}
            )
            _source_retries = source_config.get("max_retries")
            if _source_retries is not None:
                _call_kwargs["max_retries"] = int(_source_retries)

            raw = await self.mcp.call(tool_name, arguments, **_call_kwargs)

            # Transient-failure retry: intermittent remote API failures
            # (DNS bursts, upstream throttling, satellite catalog
            # flakiness) get one backoff retry here; only sources still
            # failing afterwards reach the rejected list. GEE_TIMEOUT
            # has exhausted its time budget and is never retried.
            try:
                probe = json.loads(raw)
                if (
                    isinstance(probe, dict)
                    and probe.get("status") == "error"
                    and probe.get("error_code") != "GEE_TIMEOUT"
                ):
                    await asyncio.sleep(3.0)
                    raw = await self.mcp.call(
                        tool_name, arguments, **_call_kwargs
                    )
            except (json.JSONDecodeError, TypeError):
                pass

            # Adaptive retry: structural error codes (no imagery in the
            # requested window) do not improve with identical retries;
            # widen the window per the error semantics (double post/pre
            # days, capped at 14), retry once, and keep an audit log.
            try:
                probe = json.loads(raw)
                error_code = (
                    probe.get("error_code")
                    if isinstance(probe, dict)
                    else None
                )
            except (json.JSONDecodeError, TypeError):
                error_code = None

            if error_code in ("NO_OBSERVATION_IMAGERY", "NO_PRE_EVENT_IMAGERY"):
                adapted = dict(arguments)
                day_key = (
                    "post_days"
                    if error_code == "NO_OBSERVATION_IMAGERY"
                    else "pre_days"
                )
                try:
                    _current_days = int(float(adapted.get(day_key, 3)))
                except (TypeError, ValueError):
                    _current_days = 3
                adapted[day_key] = min(_current_days * 2, 14)
                self.logger.log(
                    self.state.run_id,
                    self.name,
                    "adaptive_retry",
                    {
                        "tool": tool_name,
                        "error_code": error_code,
                        "original_arguments": arguments,
                        "adapted_arguments": adapted,
                        "purpose": "widen_satellite_window",
                    },
                )
                raw = await self.mcp.call(tool_name, adapted)

            verification = (self.verifier.validate_tool_text(raw))

            if not verification.passed:
                self.state.log(
                    "fusion_source_rejected",
                    tool=tool_name,
                    issues=[
                        issue.model_dump()
                        for issue in verification.issues
                    ],
                )
                return None

            # The manually-assigned "reliability" config value is not
            # forwarded; evidence quality is computed later in
            # _normalize_fusion_observation from the real MCP response.
            return {
                "source": source_config.get(
                    "name",
                    tool_name,
                ),
                "source_type": source_config.get(
                    "source_type",
                    "unknown",
                ),
                "tool": tool_name,
                "raw": raw,
                "spatial_resolution": (
                    source_config.get(
                        "spatial_resolution"
                    )
                ),
                "temporal_resolution": (
                    source_config.get(
                        "temporal_resolution"
                    )
                ),
            }

        # Fusion sources are independent of each other, so they are
        # fetched concurrently (gather preserves the configured order).
        # Task starts are staggered by 0.25s: nine simultaneous
        # subprocess cold starts burst the local DNS resolver and
        # intermittently cause [Errno 8] resolution failures.
        tasks = []
        for source_config in self.source_configs:
            tasks.append(
                asyncio.create_task(_collect_one(source_config))
            )
            await asyncio.sleep(0.25)

        gathered = await asyncio.gather(*tasks)

        return [obs for obs in gathered if obs is not None]
