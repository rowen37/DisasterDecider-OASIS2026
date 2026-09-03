"""GeocodeSkill — atomic skill for place-name geocoding.

Resolves a target place name to a verified Location via the geocoder
MCP and registers the target SpatialObject on the run state.  Performs
no station or hazard work.
"""

from __future__ import annotations

import re

from ..utils import utc_now

from ..models import Location, SpatialObject


class GeocodeSkill:

    name = "geocode"

    def __init__(
        self,
        state,
        verifier,
        mcp,
        logger,
    ):
        self.state = state
        self.verifier = verifier
        self.mcp = mcp
        self.logger = logger

    async def run(self, target, bias: Location | None = None):
        """Resolve the target place name.

        ``bias`` is a disambiguation anchor (usually the verified
        station location): for duplicate names (Manhattan NY vs KS),
        candidates near the anchor are preferred.
        """
        self.state.log("skill_step", skill=self.name, step="geocode", target=target)
        args: dict = {"place_name": target}
        if bias is not None:
            args["bias_lat"] = bias.latitude
            args["bias_lon"] = bias.longitude
        self.logger.log(
            self.state.run_id, self.name, "tool_call",
            {"tool": "geocode_location", "arguments": args},
        )
        raw = await self.mcp.call("geocode_location", args)
        verification = self.verifier.validate_tool_text(raw)
        if not verification.passed:
            raise RuntimeError(verification.issues[0].message)

        match = re.search(
            r"纬度\s*([-+]?\d+(?:\.\d+)?)\s*[,，]\s*经度\s*([-+]?\d+(?:\.\d+)?)",
            raw,
        )
        if not match:
            match = re.search(
                r"latitude\s*[:=]\s*([-+]?\d+(?:\.\d+)?).*?"
                r"longitude\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
                raw, flags=re.IGNORECASE | re.DOTALL,
            )
        if not match:
            raise RuntimeError(f"Geocoder returned an unparseable result: {raw}")

        location = Location(
            name=target,
            latitude=float(match.group(1)),
            longitude=float(match.group(2)),
            display_name=raw,
            source="Nominatim via Geocoder MCP",
        )
        check = self.verifier.validate_location(location)
        if not check.passed:
            raise RuntimeError(check.issues[0].message)

        self.state.add_object(
            SpatialObject(
                object_id="target_location",
                object_type="place",
                geometry={
                    "type": "Point",
                    "coordinates": [location.longitude, location.latitude],
                },
                crs="EPSG:4326",
                source=location.source,
                timestamp=utc_now().isoformat(),
                attributes={"name": target, "location_verified": True},
            )
        )
        return location
