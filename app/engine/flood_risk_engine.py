"""Risk Engine — transparent decision-support indices.

Computes CDRI (Composite Disaster Risk Index), EPS (Emergency Priority
Score) and DataConfidence. The Skill layer only collects inputs and
attaches the engine's output as evidence; all index math lives here.

Conventions:

- Water severity is MAJOR-referenced: continuous saturation
  x/(1+x) on (stage − action)/(major − action), with the
  action-stage ratio kept as a disclosed fallback when no major
  stage is published.
- Share-based dimensions (area, population, facilities) use physical
  shares ∈[0,1]; exposure combines them with max() — a hazard is as
  severe as its worst dimension.
- NO upper truncation on the index path: severities are threshold-
  segmented at 0 (physically meaningful) and linearly unbounded
  above, so CDRI/EPS grow strictly with hazard magnitude.
- Missing components take the neutral point 0.5 with an interval
  spanning [0,1]; every substitution is disclosed via data_gaps.
- Road density and route access are reported as response-capacity
  context, not as index factors.

All returned indices are normalized scores (not probabilities or
forecasts) and retain their component values for audit. Label bands
are calibrated on the two live extremes (dry-day Friendswood → Low,
Hurricane Harvey → top bands), not on fixtures.
"""

from __future__ import annotations

import math

from typing import Any

from ..utils import safe_float as _safe_float
from .uncertainty_engine import component_band


def _nz(value: Any) -> float:
    """Threshold-segmented lower bound: negatives clamp to 0;
    linear and uncapped above (no truncation)."""
    value = _safe_float(value)
    return max(0.0, value if value is not None else 0.0)


def _clamp01(value: Any) -> float:
    """Only for quantities that do NOT enter the index path: input
    sanitizing (SVI percentiles are already ∈[0,1]), response-capacity
    context, and confidence. Index-path severities always use _nz
    (no truncation)."""
    value = _safe_float(value)
    return max(0.0, min(1.0, value if value is not None else 0.0))


def cdri_risk_label(cdri: float) -> str:
    """CDRI relative-risk bands (single implementation; tests assert
    directly on it).

    CDRI = hazard × (0.5 + 0.5·exposure) × vulnerability with all
    three factors ∈[0,1]; 1.0 is the physical worst case of every
    dimension. Bands are calibrated on the two live extremes (dry
    day → Low, Harvey → Very High): <0.5% Low, 0.5–2% Moderate,
    2–10% High, ≥10% Very High.
    """
    if cdri < 0.005:
        return "Low"
    if cdri < 0.02:
        return "Moderate"
    if cdri < 0.10:
        return "High"
    return "Very High"


# Following Kron (2005): share dimensions use physical shares
# (flooded/city area, affected/regional population, affected/nearby
# facilities; numerator ⊆ denominator), naturally ∈[0,1] with no
# amplification reference. Water level is the only dimension without
# a physical cap, mapped through the continuous saturation x/(1+x)
# (asymptotic to 1, never truncated).
# Combination axioms:
#   A1 no hazard → no risk (any factor 0 → risk 0) → hazard multiplier;
#   A2 exposure enters as an equal-weight modulator [0.5, 1]
#      (OECD/JRC neutral default);
#   A3 vulnerability multiplier (CDC SVI ∈[0,1]).
EXPOSURE_MODULATOR_BASE = 0.5


class RiskEngine:
    """Pure computation: explicit inputs in, decision indices out."""

    def compute_decision_indices(
        self,
        *,
        water_level: Any,
        action_stage: float | None,
        major_stage: float | None,
        flooded_area_km2: float | None,
        city_area_km2: float | None,
        fallback_analysis_radius_km: float,
        social_vulnerability: dict[str, Any] | None,
        gis_stats: dict[str, Any],
        fused_measurements: dict[str, Any],
        fusion_sources: list[Any] | None,
        observations: list[Any],
    ) -> dict[str, Any]:
        # ── Missing-input ledger ─────────────────────────────
        # Honesty rule: every substitution of missing data must be
        # listed explicitly (data_gaps) so consumers can see which
        # gaps depressed the index.
        data_gaps: list[str] = []
        substitutions: dict[str, str] = {}

        # Analysis area: city boundary area; fall back to a circular
        # buffer when unavailable.
        if city_area_km2 is None or not (city_area_km2 and city_area_km2 > 0):
            substitutions["analysis_area"] = "city_area_missing→circular_buffer"
        analysis_area_km2 = (
            city_area_km2
            if city_area_km2 and city_area_km2 > 0
            else math.pi * fallback_analysis_radius_km ** 2
        )

        # ── Water severity (major-referenced) ────────────────
        _stage = _safe_float(water_level)
        _action = _safe_float(action_stage)
        _major = _safe_float(major_stage)
        water_basis = None
        if _stage is None or _action is None or _action <= 0:
            data_gaps.append("action_stage_or_water_level")
            substitutions["water_severity"] = "missing→0"
            water_severity = 0.0
            water_basis = "missing"
        elif _major is not None and _major > _action:
            # Major-referenced continuous saturation: x = multiples of
            # action exceeded; water_severity = x/(1+x) — monotone,
            # continuous, asymptotic to 1, never truncated (higher
            # above major always scores higher, with diminishing gain).
            _x = _nz((_stage - _action) / (_major - _action))
            water_severity = _x / (1.0 + _x)
            water_basis = "major_referenced"
        else:
            # Disclosed fallback without a major stage: action-referenced
            # ratio through the same saturation map.
            _x = _nz(_stage / _action)
            water_severity = _x / (1.0 + _x)
            water_basis = "action_referenced_fallback"

        # ── Extent severity (physical share) ─────────────────
        _extent_value = _safe_float(flooded_area_km2)
        if _extent_value is None:
            data_gaps.append("flooded_area_km2")
            # A missing flooded area must not collapse to a measured 0 —
            # that would silently halve hazard and masquerade as "no
            # flooding". Hazard degrades to water severity only, with
            # the substitution semantics recorded.
            substitutions["extent_severity"] = (
                "missing→hazard_degrades_to_water_severity_only"
            )
            extent_ratio = None
            extent_severity = None
        else:
            extent_ratio = _nz(_extent_value / analysis_area_km2)
            extent_severity = extent_ratio  # physical share ∈[0,1] (clipped)

        # ── Hazard = water dimension ─────────────────────────
        # Water level is available daily and directly measures the
        # driver variable; the satellite footprint has latency
        # (6–12 day revisit) and false positives. The extent share
        # stays spatial context (maps, exposure, facilities, routes,
        # cross-dimension contradiction alerts) and does not enter
        # the index.
        if water_basis != "missing":
            hazard = water_severity
            hazard_basis = "water_severity"
        else:
            hazard = 0.0
            hazard_basis = "water_severity_missing"

        _svi_profile = (social_vulnerability.get("profile", {}) if isinstance(social_vulnerability, dict) else {})
        _svi_raw = _safe_float(_svi_profile.get("population_weighted_svi")) if isinstance(_svi_profile, dict) else None
        if _svi_raw is None:
            data_gaps.append("population_weighted_svi")
            # A missing component must not collapse to 0 — that reads
            # "unknown vulnerability" as "zero vulnerability" and shows
            # a misleading Low. The point estimate uses the documented
            # neutral 0.5 (SVI percentile median); the interval spans
            # [0,1] (unconstrained) and the label gets a (degraded)
            # suffix.
            substitutions["vulnerability"] = "missing→neutral_0.5_(interval_spans_[0,1])"
            svi = 0.5
            vulnerability_unconstrained = True
        else:
            svi = _clamp01(_svi_raw)
            vulnerability_unconstrained = False
        total_population = _safe_float(_svi_profile.get("total_population")) or 0.0
        # Affected population prefers the areal-weighted flood ∩ tract
        # estimate; fall back to the containing tract's population with
        # the source disclosed. total_population is the tract population
        # filtered to the analysis area (city boundary) — same basis as
        # the numerator (tract granularity; boundary-crossing tracts
        # count in full).
        affected_population = _safe_float(
            gis_stats.get("affected_population")
        )
        affected_population_source = (
            (
                "census_tract_areal_weighted"
                if gis_stats.get("affected_population_method", "").startswith(
                    "areal_weighted"
                )
                else "census_tract_centroid_in_flood_polygon"
            )
            if affected_population is not None
            else "containing_tract_population"
        )
        if affected_population is None:
            affected_population = _safe_float((fused_measurements.get("population_exposure", {}) or {}).get("value")) or 0.0
        # When the population dimension is missing (no affected estimate
        # and no tract base), population severity follows the same
        # neutral-0.5 + [0,1]-interval convention as vulnerability.
        population_data_missing = (
            total_population is None or total_population <= 0
        ) or (
            _safe_float(gis_stats.get("affected_population")) is None
            and _safe_float(
                (fused_measurements.get("population_exposure", {}) or {}).get("value")
            )
            is None
        )
        if population_data_missing:
            substitutions["population_severity"] = "missing→neutral_0.5"
            population_factor = 0.5
            population_severity = 0.5
        else:
            population_factor = _nz(affected_population / total_population) if total_population else 0.0
            population_severity = population_factor  # physical share ∈[0,1]
        # Facility factor: affected facilities / total facilities in the
        # analysis area (fall back to the absolute count normalized by a
        # reference value when no affected statistic exists).
        affected_facilities = _safe_float(
            gis_stats.get("affected_facilities")
        )
        nearby_facilities = _safe_float(
            (fused_measurements.get("facility_count", {}) or {}).get("value")
        )
        facility_data_missing = False
        if (
            affected_facilities is not None
            and nearby_facilities
            and nearby_facilities > 0
        ):
            facility_factor = _nz(affected_facilities / nearby_facilities)
        elif affected_facilities is None:
            # Affected count unknown (POI ∩ flood intersection failed or
            # missing this round): the nearby absolute count is retrieval
            # density, not an exposure share. A missing dimension follows
            # the system-wide convention: neutral 0.5 + data_gaps
            # disclosure + interval spanning [0,1].
            data_gaps.append("affected_facilities")
            substitutions["facility_factor"] = (
                "affected_facilities missing→neutral_0.5"
            )
            facility_data_missing = True
            facility_factor = 0.5
        else:
            # Known affected count but missing denominator: normalize the
            # absolute count by the reference value 100 (existing
            # convention).
            facility_count = nearby_facilities or 0.0
            if nearby_facilities is None:
                substitutions["facility_factor"] = "missing→0"
            facility_factor = _nz(facility_count / 100.0)
        facility_severity = _clamp01(facility_factor)  # physical share, sanitized
        # Exposure = worst share across affected elements.
        exposure = max(population_severity, facility_severity)

        # ── Response capacity (context only; not in CDRI/EPS) ──
        # Road density and route access are still computed and reported
        # for response-resource planning, but are not index factors.
        road_count = _safe_float((fused_measurements.get("road_count", {}) or {}).get("value"))
        if road_count is None:
            road_count = 0.0
            substitutions["road_density_factor"] = "missing→0"
        # The road count comes from the fusion source's circular search
        # area (default 10 km radius); divide by that area, not the
        # city area.
        road_source_radius_km = 10.0
        for _src in fusion_sources or []:
            if isinstance(_src, dict) and _src.get("tool") == "get_road_status":
                road_source_radius_km = (
                    _safe_float((_src.get("arguments") or {}).get("radius_km"))
                    or road_source_radius_km
                )
                break
        road_area_km2 = math.pi * road_source_radius_km ** 2
        road_density_per_km2 = (
            road_count / road_area_km2 if road_area_km2 > 0 else 0.0
        )
        road_density_factor = _clamp01(road_density_per_km2 / 100.0)
        travel_time = _safe_float(gis_stats.get("travel_time_min"))
        if travel_time is None:
            substitutions["route_access_factor"] = "missing→0"
            route_access_factor = 0.0
        else:
            route_access_factor = _clamp01(1.0 - travel_time / 120.0)
        response_capacity = (road_density_factor + route_access_factor) / 2.0

        # ── CDRI = hazard × (0.5 + 0.5·exposure) × vulnerability ──
        # A1 hazard multiplier (no hazard → no risk; dry-day contract);
        # A2 exposure as an equal-weight modulator [0.5, 1] (OECD/JRC
        # neutral default); A3 vulnerability multiplier. All three
        # factors are naturally ∈[0,1]: no truncation, no amplification.
        cdri = hazard * (EXPOSURE_MODULATOR_BASE
                         + (1.0 - EXPOSURE_MODULATOR_BASE) * exposure) * svi
        # Percent display + relative-risk band (thresholds in
        # cdri_risk_label).
        cdri_percent = round(cdri * 100.0, 2)
        cdri_risk_label_value = cdri_risk_label(cdri)
        # Any input gap downgrades the label explicitly, so a reader
        # can see the score is not fully computed and does not read
        # missing data as low risk.
        if data_gaps:
            cdri_risk_label_value = (
                f"{cdri_risk_label_value} (degraded: {', '.join(data_gaps)})"
            )
        # EPS shares CDRI's measure: water severity × vulnerability
        # amplification × exposure amplification.
        eps = (
            water_severity
            * ((1.0 + svi) / 2.0)
            * ((1.0 + population_severity) / 2.0)
        )  # water_severity is a saturated map ∈[0,1); EPS ∈[0,1)
        quality_values = [
            _safe_float(item.get("quality_score"))
            for item in observations
            if isinstance(item, dict) and _safe_float(item.get("quality_score")) is not None
        ]
        quality_mean = sum(quality_values) / len(quality_values) if quality_values else 0.0
        completeness = sum((flooded_area_km2 is not None, total_population > 0, (nearby_facilities or 0) > 0, road_count > 0)) / 4.0
        data_confidence = _clamp01(0.7 * quality_mean + 0.3 * completeness)

        # First-order component-perturbation envelope: how much ±10% of
        # each component can move CDRI. Missing components (neutral
        # placeholders) go into the unconstrained list — their interval
        # spans [0,1], honestly expressing "this input is unknown".
        _unconstrained: list[str] = []
        if vulnerability_unconstrained:
            _unconstrained.append("vulnerability")
        if population_data_missing or facility_data_missing:
            _unconstrained.append("exposure")
        uncertainty = component_band(
            compose=lambda c: c["hazard"] * (
                EXPOSURE_MODULATOR_BASE
                + (1.0 - EXPOSURE_MODULATOR_BASE) * c["exposure"]
            ) * c["vulnerability"],
            components={
                "hazard": hazard,
                "exposure": exposure,
                "vulnerability": svi,
            },
            delta=0.1,
            unconstrained=_unconstrained,
        )

        return {
            "cdri": round(cdri, 4), "cdri_percent": cdri_percent,
            "data_gaps": data_gaps,
            "substitutions": substitutions,
            "cdri_risk_label": cdri_risk_label_value,
            "eps": round(eps, 4), "data_confidence": round(data_confidence, 4),
            "uncertainty": uncertainty,
            "components": {
                "hazard": round(hazard, 4),
                "exposure": round(exposure, 4),
                "vulnerability": round(svi, 4),
                # Context disclosure only (not in the CDRI product)
                "response_capacity": round(response_capacity, 4),
            },
            "inputs": {
                "affected_population": int(affected_population) if affected_population is not None else None,
                "affected_population_source": affected_population_source,
                # Method interval: bracket of the [areal-weighted,
                # centroid-whole-tract] estimates
                "affected_population_interval": gis_stats.get(
                    "affected_population_interval"
                ),
                "total_population": int(total_population) if total_population else None,
                "affected_facilities": int(affected_facilities) if affected_facilities is not None else None,
                "analysis_area_km2": round(analysis_area_km2, 3),
                "analysis_area_basis": "city_boundary" if city_area_km2 else "circular_buffer",
                "extent_ratio": round(extent_ratio, 4) if extent_ratio is not None else None,
                "water_severity": round(water_severity, 4),
                "extent_severity": (
                    round(extent_severity, 4)
                    if extent_severity is not None
                    else None
                ),
                "hazard_basis": hazard_basis,
                "extent_role": (
                    "spatial context only (maps, areal exposure, "
                    "facilities, routes, contradiction warning) — "
                    "not a CDRI hazard factor since 2026-08-31"
                ),
            },
            "definitions": {
                "water_severity_reference": (
                    "major flood stage: clamp01((stage−action)/(major−action)); "
                    "falls back to the action-stage ratio when no major "
                    "category is published"
                ),
                "share_normalization": (
                    "extent, exposed-population and affected-facility "
                    "dimensions use PHYSICAL SHARES (numerator ⊆ "
                    "denominator), ∈[0,1] by construction — no reference "
                    "constant, no amplification, no truncation"
                ),
                "water_severity_map": (
                    "x/(1+x), x = (stage−action)/(major−action): "
                    "continuous, monotone, asymptotic to 1"
                ),
                "combination_axioms": (
                    "A1 hazard = water dimension only (gauge is the daily "
                    "primary evidence; no hazard → no risk); "
                    "A2 exposure enters as an equal-weight modulator "
                    "[0.5, 1] (OECD/JRC neutral default); "
                    "A3 vulnerability multiplicative (CDC SVI — social "
                    "vulnerability substituting engineering loss ratios, "
                    "disclosed limitation)"
                ),
                "hazard_and_exposure_combination": (
                    "max of dimension severities — a hazard is as severe "
                    "as its worst dimension"
                ),
                "response_capacity_note": (
                    "road density and route access are reported as context "
                    "only; the former (1 − resilience) multiplier measured "
                    "urbanization and structurally capped metropolitan CDRI "
                    "(removed 2026-08-31 after the Harvey case)"
                ),
                "road_density_reference_segments_per_km2": 100,
                "road_measurement_radius_km": road_source_radius_km,
                "route_time_reference_minutes": 120,
                "facility_reference_count": 100,
                "missing_component_point_value": (
                    "neutral 0.5 (SVI percentile median); the reported "
                    "interval spans [0,1] for that component"
                ),
            },
        }
