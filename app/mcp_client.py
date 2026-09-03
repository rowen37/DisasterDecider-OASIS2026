from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Project root (app/..) — anchors ${PROJECT_ROOT} in config/mcp.json so
# the config stays machine-independent.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class MCPManager:
    """
    Direct MCP client used by the Skill layer.

    Important architectural choice:
    the LLM does NOT directly improvise tool parameters for the core
    hazard workflows. The Skill builds and verifies parameters first,
    then calls the MCP tool deterministically.
    """

    def __init__(self, config_path: str = "config/mcp.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tool_catalog: dict[str, list[str]] = {}
        # Read timeout/retry configuration (with defaults).
        self.timeout = float(os.getenv("MCP_CALL_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("MCP_MAX_RETRIES", "3"))
        self.retry_backoff = float(os.getenv("MCP_RETRY_BACKOFF", "1.0"))

    async def connect(self) -> None:
        """
        Connect to all configured MCP servers.

        A server that fails to connect is logged and skipped; the
        other servers are unaffected.
        """
        for name, cfg in self.config.get("mcpServers", {}).items():
            if "command" not in cfg:
                print(f"[MCP] ❌ Server '{name}' missing 'command', skipping.", file=sys.stderr)
                continue

            command = self._resolve_command(cfg["command"])
            args = [self._resolve_value(a) for a in cfg.get("args", [])]
            env = cfg.get("env")

            merged_env = os.environ.copy()
            if env:
                merged_env.update(
                    {k: self._resolve_value(v) for k, v in env.items()}
                )

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=merged_env,
            )

            try:
                read, write = await self.stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await self.stack.enter_async_context(
                    ClientSession(read, write)
                )
                # Initialize the session (with timeout protection).
                await asyncio.wait_for(session.initialize(), timeout=self.timeout)

                tools = await asyncio.wait_for(session.list_tools(), timeout=self.timeout)
                self.sessions[name] = session
                self.tool_catalog[name] = [tool.name for tool in tools.tools]
                print(f"[MCP] ✅ Connected to {name} (tools: {len(self.tool_catalog[name])})", file=sys.stderr)

            except asyncio.TimeoutError:
                print(f"[MCP] ⚠️ Connection timeout for '{name}', skipping.", file=sys.stderr)
                continue
            except Exception as e:
                print(f"[MCP] ❌ Failed to connect to '{name}': {e}, skipping.", file=sys.stderr)
                continue

        if not self.sessions:
            print("[MCP] ⚠️ No MCP servers connected. Some tools will be unavailable.", file=sys.stderr)

    async def close(self) -> None:
        await self.stack.aclose()

    @staticmethod
    def _resolve_command(command: str) -> str:
        # "python" in the config means "the interpreter running this
        # app", so MCP servers reuse the project venv without hardcoding
        # machine-specific paths.
        if command in ("python", "python3"):
            return sys.executable
        return command

    @staticmethod
    def _resolve_value(value: str) -> str:
        # Expand ${PROJECT_ROOT} and any ${ENV_VAR} placeholder; unset
        # variables resolve to "" (the server then fails to connect and
        # is skipped with a logged warning).
        def _sub(m: re.Match) -> str:
            if m.group(1) == "PROJECT_ROOT":
                return str(_PROJECT_ROOT)
            return os.environ.get(m.group(1), "")

        return _PLACEHOLDER_RE.sub(_sub, value)

    def find_server_for_tool(self, tool_name: str) -> str:
        for server, tools in self.tool_catalog.items():
            if tool_name in tools:
                return server
        raise KeyError(
            f"MCP tool '{tool_name}' was not found. Available tools: {self.tool_catalog}"
        )

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Any:
        """
        Call an MCP tool with per-attempt timeout and jittered
        exponential-backoff retries.

        Args:
            tool_name: Tool name.
            arguments: Argument dict.
            timeout: Per-attempt timeout in seconds for this call;
                defaults to the global config. Skippable heavy queries
                (e.g. large-radius POI search) should pass a shorter
                value so a slow upstream cannot stall the whole run.
            max_retries: Max attempts for this call; defaults to the
                global config. Pass 1 for fail-fast layers whose
                results are optional.

        Returns:
            The tool's text content (string) or the raw content object.

        Raises:
            RuntimeError: If the tool is not found or all retries fail.
        """
        call_timeout = float(timeout) if timeout is not None else self.timeout
        attempts = int(max_retries) if max_retries is not None else self.max_retries
        # Locate the server hosting the tool.
        try:
            server = self.find_server_for_tool(tool_name)
        except KeyError as e:
            # Graceful degradation: return an error payload instead of raising.
            return json.dumps({
                "status": "error",
                "error": f"MCP tool '{tool_name}' not available (server not connected).",
                "metadata_verified": False,
                "action_required": "Check MCP server connectivity."
            })

        session = self.sessions.get(server)
        if session is None:
            return json.dumps({
                "status": "error",
                "error": f"MCP server for '{tool_name}' is not connected.",
                "metadata_verified": False,
                "action_required": "Check MCP server connectivity."
            })

        last_exception = None

        for attempt in range(1, attempts + 1):
            try:
                # Async call with timeout.
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=call_timeout
                )
                # Success: extract text content.
                chunks: list[str] = []
                for item in result.content:
                    text = getattr(item, "text", None)
                    if text is not None:
                        chunks.append(text)

                if chunks:
                    return "\n".join(chunks)
                return result.content

            except asyncio.TimeoutError as e:
                last_exception = e
                print(f"[MCP] ⚠️ Call to {tool_name} timed out (attempt {attempt}/{attempts})", file=sys.stderr)
            except Exception as e:
                # Network errors, protocol errors, etc.
                last_exception = e
                print(f"[MCP] ⚠️ Call to {tool_name} failed (attempt {attempt}/{attempts}): {e}", file=sys.stderr)

            # Retry with jittered exponential backoff (fixed-interval
            # retries against a 503 just hammer the server)
            if attempt < attempts:
                import random
                base = self.retry_backoff * (2 ** (attempt - 1))
                await asyncio.sleep(base * (0.5 + random.random()))

        # All attempts failed.
        error_msg = f"MCP call to '{tool_name}' failed after {attempts} attempts. Last error: {last_exception}"
        return json.dumps({
            "status": "error",
            "error": error_msg,
            "metadata_verified": False,
            "action_required": "Check network connectivity and MCP server status."
        })

    def print_catalog(self) -> None:
        print("\nMCP TOOL CATALOG")
        print("-" * 72)
        for server, tools in self.tool_catalog.items():
            print(f"{server}:")
            for tool in tools:
                print(f"  - {tool}")