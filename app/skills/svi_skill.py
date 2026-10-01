"""SviSkill — atomic skill for CDC/ATSDR social vulnerability data.

Fetches tract-level SVI features via MCP and assembles the SVI
Evidence object.
"""

from __future__ import annotations

import json
from ..utils import utc_now
from typing import Any

from ..models import Evidence, Location
from ..social_good import vulnerability_profile


class SviSkill:

    name = "social_vulnerability"

    def __init__(
        self,
        state,
        verifier,
        mcp,
        logger,
        svi_tool: str,
        svi_radius_km: float,
        svi_max_features: int,
    ):
        self.state = state
        self.verifier = verifier
        self.mcp = mcp
        self.logger = logger
        self.svi_tool = svi_tool
        self.svi_radius_km = svi_radius_km
        self.svi_max_features = svi_max_features

    async def collect(
        self,
        location: Location,
    ) -> dict[str, Any]:
        self.state.log(
            "skill_step",
            skill=self.name,
            step="social_vulnerability",
            tool=self.svi_tool,
        )

        arguments = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "radius_km": self.svi_radius_km,
            "max_features": self.svi_max_features,
        }

        self.logger.log(
            self.state.run_id,
            self.name,
            "tool_call",
            {
                "tool": self.svi_tool,
                "arguments": arguments,
                "purpose": "social_vulnerability_evidence",
            },
        )

        raw = await self.mcp.call(
            self.svi_tool,
            arguments,
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
                "Social vulnerability MCP did not return valid JSON."
            ) from exc

        if data.get("status") != "ok":
            raise RuntimeError(
                "Social vulnerability MCP failed: "
                + str(
                    data.get(
                        "error",
                        "unknown error",
                    )
                )
            )

        tracts = data.get("tracts")
        if not isinstance(tracts, list):
            raise RuntimeError(
                "Social vulnerability MCP must return a 'tracts' list."
            )

        profile = vulnerability_profile(tracts)

        return {
            "status": "ok",
            "source": data.get(
                "source",
                "CDC/ATSDR SVI 2022",
            ),
            "dataset_year": data.get(
                "dataset_year"
            ),
            "spatial_unit": data.get(
                "spatial_unit",
                "census_tract",
            ),
            "radius_km": data.get(
                "radius_km",
                self.svi_radius_km,
            ),
            "tract_scope": "search_radius",
            "profile": profile,
            # Tract details (with geometry) for affected-population estimation
            "tracts": tracts,
        }



    def build_evidence(
        self,
        vulnerability: dict[str, Any],
    ) -> Evidence:
        profile = vulnerability["profile"]

        return Evidence(
            evidence_id="social_vulnerability",
            source=str(
                vulnerability.get(
                    "source",
                    "CDC/ATSDR SVI 2022",
                )
            ),
            observation=(
                "CDC/ATSDR SVI evidence covers "
                f"{profile['tract_count']} census tract(s) "
                f"with a population-weighted SVI of "
                f"{profile['population_weighted_svi']:.3f}."
            ),
            timestamp=utc_now().isoformat(),
            attributes={
                "dataset_year": vulnerability.get("dataset_year"),
                "spatial_unit": vulnerability.get("spatial_unit"),
                "radius_km": vulnerability.get("radius_km"),
                "tract_scope": vulnerability.get("tract_scope"),
                "tract_count": profile["tract_count"],
                "excluded_tract_count": profile["excluded_tract_count"],
                "total_population": profile["total_population"],
                "population_weighted_svi": profile["population_weighted_svi"],
                "high_vulnerability_threshold": profile["high_vulnerability_threshold"],
                "high_vulnerability_population": profile["high_vulnerability_population"],
                "high_vulnerability_tract_count": profile.get(
                    "high_vulnerability_tract_count"
                ),
                "coverage_completeness": profile["coverage_completeness"],
            },
        )
