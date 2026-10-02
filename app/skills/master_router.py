"""Deterministic task router — thin shell over the hazard skill registry.

All hazard knowledge (keyword tables, strong/weak semantics, the
unsupported-hazard guard) is declared by each Skill via
``@register_skill``; this class only keeps target-place extraction.
Adding a hazard requires no change to this file. See docs/EXTENDING.md.
"""

from __future__ import annotations

import re

from . import registry


class MasterRouter:
    """Deterministic task router backed by the plugin registry."""

    # USGS station IDs: 8-15 digits, the unique key. Matches
    # "station 06887000" / "USGS 06887000" / "site 06887000" / a bare ID.
    STATION_ID_RE = re.compile(
        r"(?:station|usgs|gauge|site)\s*#?\s*(\d{8,15})\b"
        r"|\b(\d{8,15})\b"
    )

    def classify(self, user_task: str) -> str:
        return registry.classify(user_task)

    def extract_station_id(self, user_task: str) -> str | None:
        """Extract the USGS station ID (deterministic regex, no LLM).

        The station ID is the unique key; place names are ambiguous
        labels. Parse the ID first and use it as the geocoding anchor
        when present (see GeocodeSkill's bias logic). Format is
        validated here; authenticity is verified against the USGS site
        metadata service at the Skill layer.
        """
        if not isinstance(user_task, str):
            return None
        match = self.STATION_ID_RE.search(user_task)
        if not match:
            return None
        station = match.group(1) or match.group(2)
        return station if station else None

    def extract_target(self, user_task: str, hazard: str) -> str:
        if not isinstance(user_task, str) or not user_task.strip():
            raise ValueError("The user task is empty.")
        if hazard not in registry.registered_hazards():
            raise ValueError(f"Unsupported hazard type: {hazard}")

        patterns = [
            r"(?:near|around|in)\s+([A-Za-z,\s]+?)(?=\s+using|\s+station|\s+USGS|\s+with|\s+for|\s+and|\s+via|\s+based|\s+flooding\b|\s+flooded\b|\s+right\s+now\b|\s+currently\b|\s+today\b|[.!?]|$)",
            r"(?:near|around|in)\s+(.+?)(?:[.!?]|$)",
            # Three-part form: "<hazard> <place> [<station>]", e.g.
            # "flood Manhattan Kansas 06887000" / "flood Manhattan"
            r"^\s*(?:assess\s+|check\s+|evaluate\s+)?"
            r"(?:flood|flooding|flooded|inundation)\s+"
            r"([A-Za-z,\s]+?)"
            r"(?=\s+(?:using|with|via)\b|\s+(?:station|usgs|gauge|site)\b"
            r"|\s+\d{8,15}\b|[.!?]|$)",
        ]
        _CUTOFF_WORDS = {
            "using", "with", "for", "and", "via",
            "based", "station", "usgs", "id", "flooding",
            "flooded", "currently", "today", "now",
        }

        for pattern in patterns:
            match = re.search(pattern, user_task, flags=re.IGNORECASE)
            if match:
                # Normalize commas to spaces: "Manhattan, Kansas" must
                # be geocoded as a whole - dropping the state qualifier
                # would locate Manhattan, KS in New York.
                raw = " ".join(
                    match.group(1).replace(",", " ").split()
                )
                words = raw.split()
                clean_words = []
                for word in words:
                    if word.lower() in _CUTOFF_WORDS:
                        break
                    clean_words.append(word)
                target = " ".join(clean_words).strip()
                if target:
                    return target

        raise ValueError(
            "Could not determine the target location from the user request. "
            "Please provide an explicit location, for example "
            "'near Los Angeles' or 'around Houston'."
        )
