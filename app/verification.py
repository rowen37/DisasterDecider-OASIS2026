from __future__ import annotations

import math
from typing import Any

from .models import Location, ValidationIssue, VerificationResult
from .utils import geodesic_km


class Verifier:
    """Deterministic verification layer. LLM output must not bypass these checks."""

    def validate_location(self, location: Location) -> VerificationResult:
        issues: list[ValidationIssue] = []

        if not math.isfinite(location.latitude) or not -90 <= location.latitude <= 90:
            issues.append(ValidationIssue(
                severity="critical",
                code="LAT_OUT_OF_RANGE",
                message=f"Latitude {location.latitude} is invalid.",
                field="latitude",
            ))

        if not math.isfinite(location.longitude) or not -180 <= location.longitude <= 180:
            issues.append(ValidationIssue(
                severity="critical",
                code="LON_OUT_OF_RANGE",
                message=f"Longitude {location.longitude} is invalid.",
                field="longitude",
            ))

        return VerificationResult(
            passed=not any(i.severity == "critical" for i in issues),
            issues=issues,
        )

    def validate_tool_text(self, text: Any) -> VerificationResult:
        issues: list[ValidationIssue] = []
        if text is None:
            issues.append(ValidationIssue(
                severity="critical",
                code="NULL_TOOL_RESULT",
                message="MCP returned no result.",
            ))
        elif not isinstance(text, str):
            issues.append(ValidationIssue(
                severity="warning",
                code="NON_STRING_TOOL_RESULT",
                message="This project currently expects text MCP results.",
            ))
        elif not text.strip():
            issues.append(ValidationIssue(
                severity="warning",
                code="EMPTY_TOOL_RESULT",
                message="MCP returned an empty string.",
            ))

        return VerificationResult(
            passed=not any(i.severity == "critical" for i in issues),
            issues=issues,
        )

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Delegate to the project-wide single implementation.
        return geodesic_km(lat1, lon1, lat2, lon2)

    def verify_station_target_distance(
        self,
        target: Location,
        station: Location,
        max_distance_km: float,
    ) -> VerificationResult:
        """
        Verify that a USGS station is spatially relevant to the
        requested flood-assessment target.

        The distance threshold is supplied by runtime configuration.
        This method never hard-codes an operational radius.
        """
        issues: list[ValidationIssue] = []

        # Validate both coordinate pairs before calculating distance.
        target_check = self.validate_location(target)
        station_check = self.validate_location(station)

        issues.extend(target_check.issues)
        issues.extend(station_check.issues)

        if any(issue.severity == "critical" for issue in issues):
            return VerificationResult(
                passed=False,
                issues=issues,
            )

        if not math.isfinite(max_distance_km) or max_distance_km <= 0:
            issues.append(
                ValidationIssue(
                    severity="critical",
                    code="INVALID_STATION_DISTANCE_POLICY",
                    message=(
                        "FLOOD_STATION_MAX_DISTANCE_KM must be "
                        "a finite positive number."
                    ),
                    field="max_distance_km",
                )
            )
            return VerificationResult(
                passed=False,
                issues=issues,
            )

        distance_km = self.haversine_km(
            target.latitude,
            target.longitude,
            station.latitude,
            station.longitude,
        )

        if distance_km > max_distance_km:
            issues.append(
                ValidationIssue(
                    severity="critical",
                    code="STATION_OUTSIDE_TARGET_RADIUS",
                    message=(
                        f"USGS station '{station.name}' is "
                        f"{distance_km:.2f} km from target "
                        f"'{target.name}', exceeding the configured "
                        f"maximum of {max_distance_km:.2f} km."
                    ),
                    field="station_target_distance_km",
                )
            )

        return VerificationResult(
            passed=not any(
                issue.severity == "critical"
                for issue in issues
            ),
            issues=issues,
        )    

    def verify_event_coordinates(
        self, latitude: float, longitude: float
    ) -> VerificationResult:
        return self.validate_location(
            Location(name="hazard_event", latitude=latitude, longitude=longitude)
        )