"""Ops layer — infrastructure base (MCP lifecycle, timeout/retry policy)."""

from .executor import OpsExecutor  # noqa: F401

__all__ = ["OpsExecutor"]
