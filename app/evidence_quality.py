from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class EvidenceQuality:
    """
    Runtime quality assessment for a single MCP observation.

    IMPORTANT:
    quality_score is NOT a probability of correctness and must not
    be described as statistical reliability.

    It measures the quality of the returned evidence based on:
      1. source authority
      2. response validity
      3. completeness
      4. temporal freshness
      5. metadata verification
    """

    source: str
    quality_score: float

    source_authority: str
    source_authority_score: float

    response_valid: bool
    completeness_score: float
    freshness_score: float

    metadata_verified: bool

    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "quality_score": self.quality_score,
            "quality_type": "evidence_quality",
            "source_authority": self.source_authority,
            "source_authority_score": self.source_authority_score,
            "response_valid": self.response_valid,
            "completeness_score": self.completeness_score,
            "freshness_score": self.freshness_score,
            "metadata_verified": self.metadata_verified,
            "reasons": list(self.reasons),
        }


# ------------------------------------------------------------------
# Source authority
# ------------------------------------------------------------------
#
# These values are NOT "reliability probabilities"; they encode the
# provenance tier of the source:
#
#   1.0 = authoritative governmental operational source
#   0.8 = authoritative but indirect/derived operational source
#   0.6 = secondary operational source
#   0.4 = non-authoritative source
#
# Unlisted sources deliberately get 0.5 rather than a guessed tier.
# ------------------------------------------------------------------

SOURCE_AUTHORITY = {
    "USGS": {
        "tier": "authoritative_government",
        "score": 1.0,
    },
    "NWS": {
        "tier": "authoritative_government",
        "score": 1.0,
    },
    "NOAA": {
        "tier": "authoritative_government",
        "score": 1.0,
    },
    "CDC": {
        "tier": "authoritative_government",
        "score": 1.0,
    },
    "CENSUS": {
        "tier": "authoritative_government",
        "score": 1.0,
    },
    "OSM": {
        "tier": "open_geospatial",
        "score": 0.8,
    },
    # Authoritative space program, but the flood-extent product is THIS
    # SYSTEM's own change-detection computation on top of it: revisit
    # latency (6-12 days) makes the acquisition timing uncertain
    # relative to a flood peak, and single-pair change detection has
    # known false positives (wet soil / vegetation / geometry). Tiered
    # below operational gauge/hydrology sources; it refines the
    # spatial picture, it does not establish the hazard.
    "SENTINEL": {
        "tier": "authoritative_platform_derived_product",
        "score": 0.6,
    },
    "COPERNICUS": {
        "tier": "authoritative_platform_derived_product",
        "score": 0.6,
    },
}


def _normalize_source(source: str | None) -> str:
    if not source:
        return "UNKNOWN"

    return source.strip().upper()


def _authority(
    source: str | None,
) -> tuple[str, float]:
    """
    Resolve the provenance tier of a source string.

    Sources arrive as free text ("USGS Water Services via Flood Alert
    MCP", "Google Earth Engine / Sentinel-1 SAR", ...), so an exact
    dict lookup never matches in practice and every source silently
    falls back to 0.5.  Substring containment on the uppercased
    string keeps the tiering honest without a registry of exact names.
    """
    normalized = _normalize_source(source)

    for key, info in SOURCE_AUTHORITY.items():
        if key in normalized:
            return (
                str(info["tier"]),
                float(info["score"]),
            )

    return "unknown", 0.5


def _freshness_score(
    timestamp: Any,
    *,
    max_age_minutes: float,
    reference_now: datetime | None = None,
) -> tuple[float, str]:
    """
    Convert observation age into a freshness score.

    This does NOT claim that older observations are incorrect.
    It only measures temporal relevance for a real-time assessment.

    ``reference_now`` allows historical replay to score freshness
    against the event window instead of the wall clock — a 2017 gauge
    reading replayed in 2026 must not be punished as "stale".

    score:
        1.0 -> current
        0.5 -> at the configured maximum age
        0.0 -> substantially older
    """

    if not timestamp:
        return 0.0, "No observation timestamp available."

    try:
        value = str(timestamp).replace("Z", "+00:00")
        observed = datetime.fromisoformat(value)

        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)

        now = reference_now or datetime.now(timezone.utc)

        age_minutes = (
            now - observed.astimezone(timezone.utc)
        ).total_seconds() / 60.0

    except (TypeError, ValueError, OverflowError):
        return 0.0, "Observation timestamp could not be parsed."

    # Future timestamps are suspicious rather than "very fresh".
    if age_minutes < -5:
        return 0.0, "Observation timestamp is unexpectedly in the future."

    if age_minutes <= 0:
        return 1.0, "Observation is current."

    if age_minutes >= max_age_minutes * 2:
        return 0.0, "Observation is substantially older than the freshness window."

    # Linear decay:
    # age = 0       -> 1.0
    # age = max_age -> 0.5
    # age = 2*max  -> 0.0
    score = 1.0 - age_minutes / (2.0 * max_age_minutes)

    return max(0.0, min(1.0, score)), (
        f"Observation age is approximately {age_minutes:.1f} minutes."
    )


def _completeness_score(
    observation: dict[str, Any],
    required_fields: tuple[str, ...],
) -> tuple[float, list[str]]:
    if not observation:
        return 0.0, ["Observation object is empty."]

    missing = [
        field
        for field in required_fields
        if observation.get(field) in (None, "")
    ]

    if not required_fields:
        return 1.0, []

    score = (
        len(required_fields) - len(missing)
    ) / len(required_fields)

    reasons: list[str] = []

    if missing:
        reasons.append(
            "Missing fields: " + ", ".join(missing)
        )

    return score, reasons


def assess_observation_quality(
    *,
    source: str | None,
    observation: dict[str, Any] | None,
    response_status: str,
    timestamp_field: str = "observation_time",
    required_fields: tuple[str, ...] = (),
    max_age_minutes: float = 180.0,
    metadata_verified: bool = False,
    reference_now: datetime | None = None,
) -> EvidenceQuality:
    """
    Evaluate the quality of an actual MCP response.

    No manually assigned reliability value is accepted.

    The final quality score is calculated from the actual response.
    ``reference_now`` shifts the freshness clock for historical replay.

    IMPORTANT:
    quality_score is an evidence-quality index, not a probability
    that the observation is correct.
    """

    authority_name, authority_score = _authority(source)

    # --------------------------------------------------------------
    # Explicitly narrow observation before using it.
    #
    # Pylance cannot safely infer that observation is non-None from
    # the separate response_valid boolean.
    # --------------------------------------------------------------
    if (
        response_status != "ok"
        or observation is None
        or not isinstance(observation, dict)
        or not observation
    ):
        return EvidenceQuality(
            source=_normalize_source(source),
            quality_score=0.0,
            source_authority=authority_name,
            source_authority_score=authority_score,
            response_valid=False,
            completeness_score=0.0,
            freshness_score=0.0,
            metadata_verified=metadata_verified,
            reasons=(
                "MCP response is not a valid successful observation.",
            ),
        )

    # At this point observation is guaranteed to be:
    # dict[str, Any]
    completeness_score, completeness_reasons = (
        _completeness_score(
            observation,
            required_fields,
        )
    )

    freshness_score, freshness_reason = _freshness_score(
        observation.get(timestamp_field),
        max_age_minutes=max_age_minutes,
        reference_now=reference_now,
    )

    metadata_score = (
        1.0
        if metadata_verified
        else 0.0
    )

    reasons: list[str] = [
        f"Source authority tier: {authority_name}.",
        freshness_reason,
    ]

    reasons.extend(completeness_reasons)

    if metadata_verified:
        reasons.append(
            "Source metadata was explicitly verified."
        )
    else:
        reasons.append(
            "Source metadata was not explicitly verified."
        )

    # --------------------------------------------------------------
    # Evidence quality index
    #
    # This is NOT a statistical probability.
    #
    # Components:
    #   source provenance     -> 40%
    #   temporal freshness    -> 30%
    #   response completeness -> 25%
    #   metadata verification -> 5%
    # --------------------------------------------------------------

    quality_score = (
        0.40 * authority_score
        + 0.30 * freshness_score
        + 0.25 * completeness_score
        + 0.05 * metadata_score
    )

    return EvidenceQuality(
        source=_normalize_source(source),
        quality_score=round(
            max(
                0.0,
                min(1.0, quality_score),
            ),
            4,
        ),
        source_authority=authority_name,
        source_authority_score=round(
            authority_score,
            4,
        ),
        response_valid=True,
        completeness_score=round(
            completeness_score,
            4,
        ),
        freshness_score=round(
            freshness_score,
            4,
        ),
        metadata_verified=metadata_verified,
        reasons=tuple(reasons),
    )