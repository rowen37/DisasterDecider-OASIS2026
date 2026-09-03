"""Ops layer — infrastructure base for tool execution.

The OpsExecutor owns the MCPManager lifecycle (construction, connect,
close) and centralizes the global timeout / retry policy so that Skills
and Agents never manage infrastructure concerns themselves.

Policy (environment overridable):

    OPS_CALL_TIMEOUT    per-call hard deadline in seconds (default: MCP_CALL_TIMEOUT or 30)
    OPS_MAX_RETRIES     total attempts per call (default 3)
    OPS_RETRY_BACKOFF   base backoff seconds, exponential (default 1.0)

The executor duck-types ``MCPManager`` (``.call`` / ``.print_catalog``),
so it can be handed to Skills in place of a raw manager and every tool
call flows through the Ops policy.
"""

from __future__ import annotations

import asyncio
import json
import os

from typing import Any

from ..mcp_client import MCPManager


class OpsExecutor:
    """Lightweight Ops base: MCP lifecycle + global timeout/retry policy."""

    def __init__(
        self,
        config_path: str = "config/mcp.json",
        call_timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
    ):
        self.call_timeout = float(
            call_timeout
            if call_timeout is not None
            else os.getenv(
                "OPS_CALL_TIMEOUT",
                os.getenv("MCP_CALL_TIMEOUT", "30.0"),
            )
        )
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else os.getenv("OPS_MAX_RETRIES", os.getenv("MCP_MAX_RETRIES", "3"))
        )
        self.retry_backoff = float(
            retry_backoff
            if retry_backoff is not None
            else os.getenv(
                "OPS_RETRY_BACKOFF",
                os.getenv("MCP_RETRY_BACKOFF", "1.0"),
            )
        )

        # Infrastructure construction lives here, not in Skills/Agents.
        self.manager = MCPManager(config_path)

    # -- lifecycle ---------------------------------------------------

    async def connect(self) -> None:
        await self.manager.connect()

    async def close(self) -> None:
        await self.manager.close()

    async def __aenter__(self) -> "OpsExecutor":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- tool execution with global policy ---------------------------

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Any:
        """
        Call an MCP tool under the Ops timeout/retry policy.

        MCPManager.call already applies per-attempt timeouts and its own
        retry loop; this wrapper adds a hard wall-clock deadline around
        the whole sequence and returns an error payload (instead of
        raising) so Skills can degrade gracefully — same contract the
        raw manager offers.

        ``timeout`` / ``max_retries`` override the global policy for this
        call only — heavy-but-skippable queries (large-radius POI search)
        should pass a short timeout and ``max_retries=1`` so a slow
        upstream costs seconds, not minutes.
        """
        last_error: Exception | None = None

        call_timeout = (
            float(timeout) if timeout is not None else self.call_timeout
        )
        call_retries = (
            int(max_retries) if max_retries is not None else self.max_retries
        )

        # Hard deadline = per-attempt timeout * (retries + 1), so the
        # inner manager's full retry sequence (including exponential
        # backoff) is not cut short by the outer layer; it only bounds
        # a truly hung call.
        hard_deadline = call_timeout * (call_retries + 1)

        for attempt in range(1, call_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.manager.call(
                        tool_name,
                        arguments,
                        timeout=timeout,
                        max_retries=max_retries,
                    ),
                    timeout=hard_deadline,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001 — degrade, don't crash
                last_error = exc

            if attempt < call_retries:
                await asyncio.sleep(
                    self.retry_backoff * (2 ** (attempt - 1))
                )

        return json.dumps(
            {
                "status": "error",
                "error": (
                    f"Ops call to '{tool_name}' failed after "
                    f"{call_retries} attempts. Last error: {last_error}"
                ),
                "metadata_verified": False,
                "action_required": (
                    "Check network connectivity and MCP server status."
                ),
            }
        )

    # -- pass-through helpers (duck-types MCPManager) -----------------

    @property
    def tool_catalog(self) -> dict[str, list[str]]:
        return self.manager.tool_catalog

    def find_server_for_tool(self, tool_name: str) -> str:
        return self.manager.find_server_for_tool(tool_name)

    def print_catalog(self) -> None:
        self.manager.print_catalog()
