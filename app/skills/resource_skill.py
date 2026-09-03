"""ResourceSkill — atomic skill for emergency resource discovery.

Calls the configured resource MCP tool with SVI-aware arguments and
normalizes the response into a resource list for the Allocation
Engine.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import Location
from .fusion import _render_template


class ResourceSkill:

    name = "resource_discovery"

    def __init__(
        self,
        state,
        verifier,
        mcp,
        logger,
        resource_tool: str | None,
        resource_tool_arguments: dict[str, Any],
    ):
        self.state = state
        self.verifier = verifier
        self.mcp = mcp
        self.logger = logger
        self.resource_tool = resource_tool
        self.resource_tool_arguments = resource_tool_arguments

    async def discover(
        self,
        target: str,
        location: Location,
        fused_evidence: dict[str, Any],
        social_vulnerability: dict[str, Any] | None = None,
        affected_population: float | None = None,
        water_level_ratio: float | None = None,
    ) -> list[dict[str, Any]]:

        if not self.resource_tool:
            raise RuntimeError(
                "RESOURCE_DISCOVERY_TOOL is not configured."
            )

        # --- Extract the SVI profile with safe degradation: missing SVI must not block optimization.
        profile = (
            social_vulnerability.get("profile", {})
            if (
                social_vulnerability is not None
                and isinstance(social_vulnerability, dict)
                and social_vulnerability.get("status") == "ok"
            )
            else {}
        )

        population_weighted_svi = profile.get("population_weighted_svi")
        high_vuln_tracts        = profile.get("high_vulnerability_tract_count", 0)
        svi_available           = population_weighted_svi is not None

        context = {
            "target":              target,
            "latitude":            location.latitude,
            "longitude":           location.longitude,
            "vulnerability_score":            population_weighted_svi,
            "high_vulnerability_tract_count": high_vuln_tracts,
            # Inputs for resource-demand estimation (supplied by run();
            # when empty the resource MCP applies its documented
            # conservative defaults). Values must be strings: FastMCP
            # strictly type-checks str parameters and rejects numbers.
            "affected_population": (
                str(int(affected_population))
                if affected_population is not None
                else ""
            ),
            "water_level_ratio": (
                f"{water_level_ratio:.4f}"
                if water_level_ratio is not None
                else ""
            ),
        }

        arguments = _render_template(
            self.resource_tool_arguments,
            context,
        )

        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool":      self.resource_tool,
                "arguments": arguments,
                "purpose":   "resource_discovery",
                # Log the SVI integration status for auditing
                "svi_integrated":          svi_available,
                "vulnerability_score":     population_weighted_svi,
                "high_vuln_tract_count":   high_vuln_tracts,
            },
        )

        # Resource discovery runs serial Overpass queries for 5 facility
        # categories and degrades gracefully (the optimizer skips it):
        # one 90s budget with a single attempt bounds the cost of
        # upstream congestion.
        raw = await self.mcp.call(
            self.resource_tool,
            arguments,
            timeout=90.0,
            max_retries=1,
        )

        verification = self.verifier.validate_tool_text(raw)

        if not verification.passed:
            raise RuntimeError(
                verification.issues[0].message
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Resource discovery MCP did not return valid JSON."
            ) from exc

        if data.get("status") != "ok":
            raise RuntimeError(
                "Resource discovery MCP failed: "
                + str(data.get("error", "unknown error"))
            )

        resources = data.get("resources")

        if not isinstance(resources, list):
            raise RuntimeError(
                "Resource discovery MCP must return a 'resources' list."
            )

        return resources
