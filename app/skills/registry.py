"""Hazard skill plugin registry - zero-intrusion entry point for new hazards.

Usage (on the orchestration Skill class definition)::

    from .registry import register_skill

    @register_skill(
        hazard="wildfire",
        keywords=("wildfire", "fire", "blaze"),          # strong keywords
        weak_keywords=("burn scar",),                     # weak keywords
        unsupported=False,
    )
    class WildfireSkill:
        OUTPUT_RULES = "..."        # hazard-specific rules injected into synthesis
        ...

After registration:
    * ``MasterRouter.classify``     - keyword routing takes effect automatically
    * ``create_skill(hazard, ...)`` - builds instances; main.py has no if/elif
    * ``hazard_prompt_rules``       - the synthesis layer (agent.py) reads the
      hazard-specific output rules, so shared prompts carry no hazard knowledge

Note: the registry imports no Skill modules (avoiding circular imports);
registration happens when each Skill module is imported (importing the
app.skills package completes registration).

Keyword semantics:
    keywords        strong - presence alone identifies the hazard
    weak_keywords   weak - common in other hazards' contexts (e.g. the flood
                    word "river" in wildfire coverage); considered only when
                    no unsupported-hazard signal is present
    unsupported_hazards  vocabulary for "known but unsupported" hazards:
                    matching text is rejected explicitly, never mis-routed
                    into a registered hazard
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Type


@dataclass(frozen=True)
class SkillSpec:
    """All routing and prompt information for one registered hazard skill."""

    hazard: str
    skill_class: Type[Any]
    keywords: tuple[str, ...] = ()
    weak_keywords: tuple[str, ...] = ()
    unsupported_hazard_words: tuple[str, ...] = ()
    description: str = ""
    prompt_rules: str = ""
    output_validator: Any = None


_SKILL_SPECS: dict[str, SkillSpec] = {}


def register_skill(
    hazard: str,
    *,
    keywords: tuple[str, ...] | list[str] = (),
    weak_keywords: tuple[str, ...] | list[str] = (),
    unsupported_hazard_words: tuple[str, ...] | list[str] = (),
    description: str = "",
):
    """Class decorator: register an orchestration Skill in the plugin table.

    ``prompt_rules`` comes from the class attribute ``OUTPUT_RULES`` and the
    hazard-specific output validator from ``OUTPUT_VALIDATOR`` (both empty if
    absent) - injected/invoked by the synthesis layer when producing the
    final report. Hazard knowledge stays in the hazard plugin; the shared
    agent stays generic.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        _SKILL_SPECS[hazard] = SkillSpec(
            hazard=hazard,
            skill_class=cls,
            keywords=tuple(keywords),
            weak_keywords=tuple(weak_keywords),
            unsupported_hazard_words=tuple(unsupported_hazard_words),
            description=description or (cls.__doc__ or "").strip().split("\n")[0],
            prompt_rules=getattr(cls, "OUTPUT_RULES", "") or "",
            output_validator=getattr(cls, "OUTPUT_VALIDATOR", None),
        )
        return cls

    return decorator
def registered_hazards() -> tuple[str, ...]:
    return tuple(sorted(_SKILL_SPECS))


def get_spec(hazard: str) -> SkillSpec | None:
    return _SKILL_SPECS.get(hazard)


def classify(text: str) -> str:
    """Deterministic routing via word boundaries, strong/weak keywords and
    the unsupported-hazard guard.

    Returns the hazard name; raises ValueError (with the supported list)
    when the task cannot be classified safely.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The user task is empty.")
    lowered = text.lower()

    def _hit(words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words)

    def _matched_words(words: tuple[str, ...]) -> list[str]:
        """Return the actual vocabulary items found in the request.

        The old implementation recorded the registered skill name
        (``flood``) when an unsupported word such as ``wildfire`` was
        found.  That produced the misleading message "flood is not
        supported" even though flood is the only supported hazard.
        """
        return [
            word
            for word in words
            if re.search(rf"\b{re.escape(word)}\b", lowered)
        ]

    matches: list[str] = []
    weak_only_matches: list[str] = []
    unsupported_hits: list[str] = []

    for spec in _SKILL_SPECS.values():
        if _hit(spec.keywords):
            matches.append(spec.hazard)
        elif _hit(spec.weak_keywords):
            weak_only_matches.append(spec.hazard)
        unsupported_hits.extend(
            _matched_words(spec.unsupported_hazard_words)
        )

    if len(matches) > 1:
        raise ValueError(
            "The request contains multiple hazard types. "
            "Please specify one hazard type explicitly."
        )
    if matches and unsupported_hits:
        # Registered and unsupported hazard both appear: reject as multi-hazard ambiguity.
        raise ValueError(
            "The request contains multiple hazard types. "
            "Please specify one hazard type explicitly."
        )
    if matches:
        return matches[0]

    # Unsupported-hazard signals take precedence over weak keywords: "river" in a wildfire report is not a flood task.
    if unsupported_hits:
        raise ValueError(
            f"{unsupported_hits[0]} is not a supported hazard in this system. "
            f"Supported hazards: {', '.join(registered_hazards())}."
        )

    if len(weak_only_matches) == 1:
        return weak_only_matches[0]
    if len(weak_only_matches) > 1:
        raise ValueError(
            "The request contains multiple hazard types. "
            "Please specify one hazard type explicitly."
        )

    raise ValueError(
        "Cannot classify the task. Please explicitly mention one of: "
        f"{', '.join(registered_hazards())}."
    )


def create_skill(
    hazard: str,
    state: Any,
    verifier: Any,
    hitl: Any,
    mcp: Any,
    logger: Any,
) -> Any:
    """Build a skill instance from the registry (the single dispatch entry used by main.py)."""
    spec = _SKILL_SPECS.get(hazard)
    if spec is None:
        raise ValueError(
            f"Unsupported hazard: {hazard!r}. Registered: "
            f"{', '.join(registered_hazards())}."
        )
    return spec.skill_class(state, verifier, hitl, mcp, logger)


def hazard_prompt_rules(hazard: str) -> str:
    """Return the synthesis-layer output rules declared by the hazard plugin (empty string if none)."""
    spec = _SKILL_SPECS.get(hazard)
    return spec.prompt_rules if spec else ""
