"""Uncertainty Engine — first-order component-perturbation bands for index engines.

Philosophy (shared by every composite index this project produces): the
decision indices are point estimates assembled from normalized components.
This module quantifies how much the composite value can move when each
component is perturbed by ±delta, producing:

    interval      conservative envelope over {base, each ±delta
                  one-at-a-time, joint all-minus, joint all-plus}
    sensitivity   per-component one-at-a-time deviation from base
                  (larger = the composite depends more on it)

This is a first-order approximation, NOT a probabilistic confidence
interval (no distributional assumptions are made).  The ±10% default
delta is a project convention.
"""

from __future__ import annotations

from typing import Any, Callable

from ..utils import safe_float as _safe_float


def component_band(
    compose: Callable[[dict[str, float]], float],
    components: dict[str, float],
    delta: float = 0.1,
    unconstrained: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate ``compose`` on perturbed component dicts.

    ``compose`` must be a pure function of the component values; it is
    called 2n+3 times (base, each component ±delta, joint ±delta).
    Components missing/None are treated as 0.0 for perturbation purposes
    (the base compose call itself decides how to handle them — callers
    normally pre-normalize).

    ``unconstrained`` names components whose value is a documented
    neutral placeholder because the underlying datum is MISSING.  For
    those, the interval additionally evaluates the component at 0 and 1
    (its full physical range), so the reported band honestly spans
    "we do not know this input" instead of pretending the placeholder
    is a measurement.
    """
    base_components = {
        k: (_safe_float(v) or 0.0) for k, v in components.items()
    }
    base_value = compose(base_components)
    unconstrained = unconstrained or []

    def _nz(x: float) -> float:
        # Piecewise lower bound at zero (negatives clamp to 0), no upper
        # truncation -- matches the index path's no-clamp convention.
        return max(0.0, x)

    def _evaluate(overrides: dict[str, float]) -> float:
        merged = dict(base_components)
        merged.update(overrides)
        return compose(merged)

    lo_candidates = [base_value]
    hi_candidates = [base_value]
    sensitivity: dict[str, float] = {}

    for name, value in base_components.items():
        if name in unconstrained:
            # Missing component: bound over its full [0, 1] range
            # (a ±10% perturbation is meaningless).
            v_zero = _evaluate({name: 0.0})
            v_one = _evaluate({name: 1.0})
            sensitivity[name] = round(
                max(abs(v_zero - base_value), abs(v_one - base_value)), 6
            )
            lo_candidates.extend([v_zero, v_one])
            hi_candidates.extend([v_zero, v_one])
            continue
        plus = _nz(value + delta)
        minus = _nz(value - delta)
        v_plus = _evaluate({name: plus})
        v_minus = _evaluate({name: minus})
        sensitivity[name] = round(
            max(abs(v_plus - base_value), abs(v_minus - base_value)), 6
        )
        lo_candidates.extend([v_plus, v_minus])
        hi_candidates.extend([v_plus, v_minus])

    # Joint perturbation (conservative envelope): all components shifted ±delta together.
    joint_plus = _evaluate(
        {k: _nz(v + delta) for k, v in base_components.items()}
    )
    joint_minus = _evaluate(
        {k: _nz(v - delta) for k, v in base_components.items()}
    )
    lo_candidates.append(joint_minus)
    lo_candidates.append(joint_plus)
    hi_candidates.append(joint_minus)
    hi_candidates.append(joint_plus)

    # Joint unconstrained: all missing components set to 0 / 1 together (joint extremes).
    if unconstrained:
        all_zero = _evaluate({k: 0.0 for k in unconstrained})
        all_one = _evaluate({k: 1.0 for k in unconstrained})
        lo_candidates.extend([all_zero, all_one])
        hi_candidates.extend([all_zero, all_one])

    lo = min(lo_candidates)
    hi = max(hi_candidates)
    interval_percent = [round(lo * 100, 2), round(hi * 100, 2)]

    ranked = sorted(
        sensitivity.items(), key=lambda kv: kv[1], reverse=True
    )

    return {
        "delta": delta,
        "base_value": round(base_value, 6),
        "interval": [round(lo, 6), round(hi, 6)],
        "interval_percent": interval_percent,
        "sensitivity": sensitivity,
        "top_components": [name for name, _ in ranked[:2]],
        "unconstrained_components": list(unconstrained),
        "method": (
            "first_order_perturbation_with_full_range_for_missing_inputs"
            if unconstrained
            else "first_order_perturbation"
        ),
    }
