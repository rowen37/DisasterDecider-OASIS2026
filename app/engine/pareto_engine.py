"""Pareto Engine — multi-objective optimization for allocation plans.

Moved verbatim from the former app/skills.py (later app/skills/fusion.py)
so that the Skills layer contains orchestration only, no math.
"""

from __future__ import annotations

import math

from typing import Any


def _normalize_objective(
    value: float,
    values: list[float],
    direction: str,
) -> float:

    minimum = min(values)
    maximum = max(values)

    if math.isclose(
        minimum,
        maximum,
    ):
        return 1.0

    normalized = (
        (value - minimum)
        / (maximum - minimum)
    )

    if direction == "minimize":
        return 1.0 - normalized

    if direction == "maximize":
        return normalized

    raise RuntimeError(
        f"Unsupported objective direction: {direction}"
    )


def _dominates(
    candidate: dict[str, Any],
    other: dict[str, Any],
    objectives: list[dict[str, Any]],
) -> bool:
    """
    candidate dominates other when:

    - candidate is no worse on every objective
    - candidate is strictly better on at least one objective
    """

    better_or_equal = True
    strictly_better = False

    for objective in objectives:

        name = objective["name"]
        direction = objective["direction"]

        a = float(candidate["objectives"][name])
        b = float(other["objectives"][name])

        if direction == "maximize":

            if a < b:
                better_or_equal = False
                break

            if a > b:
                strictly_better = True

        elif direction == "minimize":

            if a > b:
                better_or_equal = False
                break

            if a < b:
                strictly_better = True

        else:
            raise RuntimeError(
                f"Unsupported objective direction: "
                f"{direction}"
            )

    return (
        better_or_equal
        and strictly_better
    )


def pareto_frontier(
    plans: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return the non-dominated feasible allocation plans.
    """

    frontier = []

    for candidate in plans:

        dominated = False

        for other in plans:

            if candidate is other:
                continue

            if _dominates(
                other,
                candidate,
                objectives,
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


def select_best_pareto_plan(
    frontier: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:

    if not frontier:
        raise RuntimeError(
            "Pareto frontier is empty."
        )

    total_weight = sum(
        float(weights.get(
            objective["name"],
            0.0,
        ))
        for objective in objectives
    )

    if total_weight <= 0:
        raise RuntimeError(
            "At least one positive objective weight "
            "is required to select a recommended plan."
        )

    scores = []

    for plan in frontier:

        score = 0.0

        for objective in objectives:

            name = objective["name"]

            weight = float(
                weights.get(
                    name,
                    0.0,
                )
            )

            if weight <= 0:
                continue

            all_values = [
                float(
                    item["objectives"][name]
                )
                for item in frontier
            ]

            normalized = _normalize_objective(
                float(
                    plan["objectives"][name]
                ),
                all_values,
                objective["direction"],
            )

            score += (
                weight
                * normalized
            )

        plan["recommendation_score"] = round(score, 6)

        scores.append(
            (
                score,
                plan,
            )
        )

    scores.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scores[0][1]
