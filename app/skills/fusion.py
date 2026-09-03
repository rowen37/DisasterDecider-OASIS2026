"""Multi-source fusion and Pareto optimization utilities.

Verbatim extraction from the original monolithic app/skills.py.
"""

from __future__ import annotations

import json
import os

from typing import Any

def _load_json_env(
    name: str,
    required: bool = True,
) -> Any:
    """
    Load JSON configuration from environment variables.

    No operational parameter is hard-coded in Python.
    """

    value = os.getenv(name)

    if not value:
        if required:
            raise RuntimeError(
                f"Required environment variable '{name}' "
                f"is not configured."
            )
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Environment variable '{name}' "
            f"does not contain valid JSON."
        ) from exc


from typing import Any


def _render_template(
    value: Any,
    context: dict[str, Any],
) -> Any:

    if isinstance(value, str):
        # Detect a "pure" single placeholder: the whole string is
        # exactly "{key}" with nothing else around it.
        if (
            value.startswith("{")
            and value.endswith("}")
            and value.count("{") == 1
            and value.count("}") == 1
        ):
            key = value[1:-1]
            if key not in context:
                raise RuntimeError(f"Missing template variable '{key}'.")
            return context[key]  # preserve original type (float/int/bool/etc.)

        try:
            return value.format(**context)
        except KeyError as exc:
            raise RuntimeError(
                f"Missing template variable '{exc.args[0]}'."
            ) from exc

    if isinstance(value, dict):
        return {
            key: _render_template(
                item,
                context,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _render_template(
                item,
                context,
            )
            for item in value
        ]

    return value


