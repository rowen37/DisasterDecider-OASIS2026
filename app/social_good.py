from __future__ import annotations

import math
import os

from .utils import geodesic_km
from typing import Any


class SocialGoodError(RuntimeError):
    """Raised when social-good metrics cannot be computed from evidence."""


# ============================================================================
# Weight and threshold rationale
# ============================================================================

"""
Weight and threshold rationale:

1. CDC/ATSDR SVI structure and the 0.90 threshold
   - SVI aggregates 16 equally weighted variables into 4 themes, and
     the 4 themes into a composite index; the official weights are
     used as-is.
   - 0.90 marks the nationally most-vulnerable 10% under the CDC SVI
     percentile definition and serves as the high-vulnerability cut.

2. Lambda (VULNERABILITY_WEIGHT) and the multi-objective weights are
   project conventions, not literature-derived values.
   - Default lambda = 1.0; it is exposed as a tunable parameter
     (HITL slider / environment variable) with sensitivity analysis.
   - Default objective weights are likewise conventions; there is no
     objectively optimal cross-objective weighting, so weights are
     logged to the audit trail and support manual override.

Notes:
   - CDC sentinel value -999 is treated as null at the SVI MCP layer.
"""


# ============================================================================
# Utility functions
# ============================================================================

def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ============================================================================
# Demand-impact pure computation: shared by lambda / coverage-radius
# sensitivity analysis
# ============================================================================

def build_demand_records(
    tract_exposure: list[dict[str, Any]],
    svi_tracts: list[dict[str, Any]] | None,
    centroid_of,
) -> list[dict[str, Any]]:
    """
    Merge per-tract exposure records from geometry_engine.tract_exposure
    with SVI census tracts (centroids) into demand records, the standard
    input for VWUN / equity gap.

    Each record: tract_id / exposed_population / flooded_fraction / svi /
    centroid (lat, lon). Tracts missing svi or exposed_population are
    skipped rather than imputed. ``centroid_of(tract)`` returns
    (lat, lon) or None.
    """
    by_id = {
        t.get("tract_id"): t
        for t in (svi_tracts or [])
        if isinstance(t, dict)
    }
    records: list[dict[str, Any]] = []
    for record in tract_exposure or []:
        if not isinstance(record, dict):
            continue
        svi = _finite(record.get("svi"))
        exposed = _finite(record.get("exposed_population"))
        if svi is None or exposed is None:
            continue
        tract = by_id.get(record.get("tract_id"))
        centroid = centroid_of(tract) if tract is not None else None
        records.append(
            {
                "tract_id": record.get("tract_id"),
                "exposed_population": exposed,
                "flooded_fraction": _finite(
                    record.get("flooded_fraction")
                )
                or 0.0,
                "svi": svi,
                "centroid": centroid,
            }
        )
    return records


def demand_impacts_for_plan(
    demand_records: list[dict[str, Any]],
    allocation_points: list[tuple[float, float]],
    coverage_radius_km: float,
) -> list[dict[str, Any]]:
    """
    Compute per-tract demand impacts for a plan's facility locations and
    coverage radius (coverage is a binary test of whether the tract
    centroid falls within R km of any facility; sensitivity analysis
    exposes the impact of that radius).

    Shared with flood_skill / /api/equity/sensitivity so pipeline
    numbers and frontend recomputation agree.
    """
    impacts = []
    for record in demand_records:
        centroid = record.get("centroid")
        covered = False
        if centroid is not None and allocation_points:
            for f_lat, f_lon in allocation_points:
                if (
                    geodesic_km(centroid[0], centroid[1], f_lat, f_lon)
                    <= coverage_radius_km
                ):
                    covered = True
                    break
        impacts.append(
            {
                "tract_id": record.get("tract_id"),
                "exposed_population": record["exposed_population"],
                "hazard_exposure": record["flooded_fraction"],
                "coverage": 1.0 if covered else 0.0,
                "svi": record["svi"],
            }
        )
    return impacts

def vulnerability_profile(tracts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build an evidence-based vulnerability profile.

    Never invents population, SVI, or hazard values; tracts missing
    population or SVI are excluded and reported as missing evidence.
    """
    valid = []
    excluded = []

    for tract in tracts:
        population = _finite(tract.get("population"))
        svi = _finite(tract.get("svi"))

        if population is None or population < 0 or svi is None:
            excluded.append(
                {"tract_id": tract.get("tract_id"), "reason": "missing_population_or_svi"}
            )
            continue

        valid.append(
            {
                **tract,
                "population": population,
                "svi": _clamp01(svi),
            }
        )

    total_population = sum(item["population"] for item in valid)

    if total_population <= 0:
        raise SocialGoodError("No population-weighted SVI evidence is available.")

    weighted_svi = sum(item["population"] * item["svi"] for item in valid) / total_population

    high_vulnerability = [item for item in valid if item["svi"] >= 0.90]
    high_vulnerability_population = sum(item["population"] for item in high_vulnerability)

    return {
        "status": "ok",
        "tract_count": len(valid),
        "excluded_tract_count": len(excluded),
        "total_population": total_population,
        "population_weighted_svi": weighted_svi,
        "high_vulnerability_threshold": 0.90,
        "high_vulnerability_population": high_vulnerability_population,
        "high_vulnerability_tract_count": len(high_vulnerability),
        "coverage_completeness": len(valid) / len(tracts) if tracts else 0.0,
        "tracts": valid,
        "excluded": excluded,
    }


def compute_vulnerability_weighted_unmet_need(
    demand_impacts: list[dict[str, Any]],
    vulnerability_weight: float,
) -> float:
    """
    Compute VWUN (vulnerability-weighted unmet need).

    Formula:
        VWUN = Σ(P_exposed * (1 - C) * (1 + λ * SVI))

    P_exposed is already weighted by flooded fraction
    (exposed_population = tract_population * flooded_fraction, from
    geometry_engine's area-weighted computation). Do NOT multiply by
    flooded_fraction again here — that would square the fraction in
    highly exposed tracts and systematically understate VWUN.

    Args:
        demand_impacts: per-tract impact records, each with:
            - tract_id (str)
            - exposed_population (float, already includes flooded
              fraction weighting)
            - coverage (float, [0,1])
            - svi (float, [0,1])
            - hazard_exposure (float, [0,1]; audit pass-through only,
              not part of the formula)
        vulnerability_weight: lambda, default 1.0.

    No fallback values; missing fields raise SocialGoodError.
    """
    if not math.isfinite(vulnerability_weight) or vulnerability_weight < 0:
        raise SocialGoodError("vulnerability_weight must be finite and non-negative.")

    total = 0.0
    for impact in demand_impacts:
        tract_id = str(impact.get("tract_id") or impact.get("geoid") or "unknown")
        pop = _finite(impact.get("exposed_population"))
        if pop is None or pop < 0:
            raise SocialGoodError(f"VWUN impact for {tract_id} has invalid exposed_population.")
        cov = _finite(impact.get("coverage"))
        if cov is None or not 0 <= cov <= 1:
            raise SocialGoodError(f"VWUN impact for {tract_id} has invalid coverage.")
        svi = _finite(impact.get("svi"))
        if svi is None or not 0 <= svi <= 1:
            raise SocialGoodError(f"VWUN impact for {tract_id} has invalid svi.")

        total += pop * (1 - cov) * (1 + vulnerability_weight * svi)

    return total


def compute_equity_gap(demand_impacts: list[dict[str, Any]], threshold: float) -> float:
    """
    Coverage gap between high- and low-vulnerability tracts.

    A positive value means high-vulnerability tracts receive higher
    coverage.
    """
    if not 0 <= threshold <= 1:
        raise SocialGoodError("Equity threshold must be within [0,1].")

    high, low = [], []
    for impact in demand_impacts:
        svi = _finite(impact.get("svi"))
        cov = _finite(impact.get("coverage"))
        if svi is None or cov is None:
            raise SocialGoodError("Equity audit requires svi and coverage for every impact.")
        if not (0 <= svi <= 1 and 0 <= cov <= 1):
            raise SocialGoodError("SVI and coverage must be within [0,1].")
        if svi >= threshold:
            high.append(cov)
        else:
            low.append(cov)

    if not high or not low:
        raise SocialGoodError("Equity gap requires both high- and lower-SVI groups.")

    return sum(high) / len(high) - sum(low) / len(low)


def compute_equity_gap_or_none(
    demand_impacts: list[dict[str, Any]], threshold: float
) -> tuple[float | None, str]:
    """
    Return-None variant of the equity gap.

    For demographically homogeneous areas the gap is mathematically
    undefined (no tract reaches the national top-10% SVI threshold) —
    normal for real events, not an error. In that case returns
    (None, reason); other computable metrics such as VWUN are
    unaffected. Invalid input (out-of-range values, missing fields)
    still raises.

    Returns (gap | None, note); note is disclosed in the audit ledger.
    """
    try:
        return compute_equity_gap(demand_impacts, threshold), "ok"
    except SocialGoodError as exc:
        if "both high- and lower-SVI groups" in str(exc):
            return None, (
                "undefined: no demand tract at/above the national "
                f"top-10% SVI threshold ({threshold:.2f})"
            )
        raise


# ============================================================================
# Weight-rationale helpers (for reports)
# ============================================================================

