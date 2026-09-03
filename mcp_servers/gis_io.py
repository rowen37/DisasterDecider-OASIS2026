"""Path helpers shared by the GIS MCP servers.

The three GIS servers (raster/vector/viz) are separate stdio processes
launched by script path (the script directory lands on sys.path), so
they can import this module directly.

output_path is an MCP caller (LLM) controlled string: without a
restriction, something like "../../.zshrc" or any absolute path would
let a tool write files outside the allowed roots (creating arbitrary
directories along the way). Writes are therefore restricted to two
allowed roots: the system temp directory and the project's
static/maps output directory.
"""

import os
import tempfile

from pathlib import Path


def _allowed_roots() -> list[Path]:
    roots = [Path(tempfile.gettempdir()).resolve()]
    # Project root = parent of mcp_servers/. Register both candidates to
    # cover launches from the project root and from any other cwd.
    project_root = Path(__file__).resolve().parent.parent
    for base in (project_root, Path.cwd()):
        roots.append((base / "static" / "maps").resolve())
    # Dedupe, preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def is_allowed_path(path: str | os.PathLike) -> bool:
    """Whether the path (after resolve) falls inside an allowed root."""
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return any(
        resolved == root or root in resolved.parents
        for root in _allowed_roots()
    )


def tmp_path(suffix: str) -> str:
    """Create a temp file with the given suffix and return its path (handle closed immediately)."""

    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p


def out_path(path: str, suffix: str) -> str:
    """Resolve an output path: empty falls back to a temp file; otherwise
    the path must lie inside an allowed root (system temp dir or the
    project's static/maps), or ValueError is raised - fail rather than
    write to an arbitrary file."""

    if not path:
        return tmp_path(suffix)
    resolved = Path(path).expanduser().resolve()
    if not is_allowed_path(resolved):
        raise ValueError(
            f"output_path escapes allowed roots "
            f"(temp dir, static/maps): {path}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)
