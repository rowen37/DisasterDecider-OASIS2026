from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .models import SkillResult


# LLM evidence compaction. The LLM needs facts and magnitudes, not
# coordinate strings: geometry is summarized, long strings truncated,
# long lists capped, and a hard total-length limit enforced. The
# fallback template is unaffected (it reads SkillResult directly).
_LLM_MAX_STRING = 400
_LLM_MAX_LIST = 25
_LLM_MAX_PROMPT_CHARS = 120_000
_GEOMETRY_KEYS = frozenset({"geometry", "geojson", "coordinates", "rings"})


def _build_compact_packet(packet: dict) -> dict:
    """Tighten progressively until the packet fits the model context budget (400/25, then 200/12, then 120/6)."""
    for string_limit, list_limit in ((400, 25), (200, 12), (120, 6)):
        compact = _compact_for_llm(
            packet, string_limit=string_limit, list_limit=list_limit
        )
        if len(json.dumps(compact, ensure_ascii=False)) <= _LLM_MAX_PROMPT_CHARS:
            return compact
    return compact


def _geometry_summary(value: Any) -> dict:
    """Summarize geometry as type plus size estimate; no coordinates."""
    size = 0
    try:
        size = len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        size = -1
    vtype = value.get("type") if isinstance(value, dict) else type(value).__name__
    return {
        "geometry_omitted": True,
        "type": vtype,
        "approx_serialized_chars": size,
        "note": "coordinates removed for LLM; see map layers / geojson artifacts",
    }


def _compact_for_llm(value: Any, string_limit: int = _LLM_MAX_STRING,
                     list_limit: int = _LLM_MAX_LIST):
    """Recursively compact: summarize geometry, truncate long strings, cap long lists."""
    if isinstance(value, str):
        if len(value) > string_limit:
            return value[:string_limit] + (
                f"…[truncated {len(value) - string_limit} chars]"
            )
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _GEOMETRY_KEYS and isinstance(v, (dict, list)):
                out[k] = _geometry_summary(v)
            else:
                out[k] = _compact_for_llm(v)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        head = [_compact_for_llm(i, string_limit, list_limit)
                for i in items[:list_limit]]
        if len(items) > list_limit:
            head.append(f"…[{len(items) - list_limit} more items omitted]")
        return head
    return str(value)


@dataclass
class StrategyDecision:
    """Outcome of the pre-LLM strategy selection step."""

    strategy: str  # "llm_synthesis" | "policy_no_report"
    skip_llm: bool
    reason: str
    message: str | None = None


class StrategySelector:
    """
    Agent-layer policy gate, evaluated BEFORE FinalDecisionAgent.

    If the current water level is below NO_REPORT_WATER_LEVEL_RATIO of
    the NWS action stage, the situation is operationally quiet: the
    agent outputs a deterministic "no report needed" message and skips
    the LLM synthesis call entirely (cost + latency optimization).
    Unknown water level or action stage never skips; the LLM path is
    the safe default.

    A single-gauge water level is not a sufficient statistic for
    city-wide flooding: when SAR has detected a flood extent or active
    alerts exist (urban pluvial flooding is weakly correlated with an
    upstream gauge), never short-circuit even at a low ratio; the
    report must still be generated.
    """

    NO_REPORT_WATER_LEVEL_RATIO = 0.5

    def select(
        self,
        *,
        water_level: float | None,
        action_stage: float | None,
        skill_name: str = "",
        flooded_area_km2: float | None = None,
        alert_count: int | None = None,
    ) -> StrategyDecision:
        # Flood extent or alerts present: never classify as quiet; use
        # the LLM report path.
        if (flooded_area_km2 or 0) > 0 or (alert_count or 0) > 0:
            return StrategyDecision(
                strategy="llm_synthesis",
                skip_llm=False,
                reason=(
                    "flood extent or active alerts present; "
                    "quiet-gauge policy short-circuit suppressed"
                ),
            )

        if water_level is None or action_stage is None or action_stage <= 0:
            return StrategyDecision(
                strategy="llm_synthesis",
                skip_llm=False,
                reason="water level or action stage unknown; cannot apply policy",
            )

        ratio = float(water_level) / float(action_stage)

        if ratio < self.NO_REPORT_WATER_LEVEL_RATIO:
            return StrategyDecision(
                strategy="policy_no_report",
                skip_llm=True,
                reason=(
                    f"water level {water_level} is below "
                    f"{self.NO_REPORT_WATER_LEVEL_RATIO:.0%} of action stage "
                    f"{action_stage} (ratio {ratio:.2f})"
                ),
                message=(
                    "NO REPORT REQUIRED (policy short-circuit)\n\n"
                    f"Current water level ({water_level} ft) is below "
                    f"{self.NO_REPORT_WATER_LEVEL_RATIO:.0%} of the NWS action "
                    f"stage ({action_stage} ft); ratio = {ratio:.2f}.\n\n"
                    "Per the agent strategy policy, no flood report is "
                    "generated and LLM synthesis was skipped to save cost "
                    "and latency. Data collection, verification and "
                    "decision indices were still executed and remain "
                    "available in the structured result."
                ),
            )

        return StrategyDecision(
            strategy="llm_synthesis",
            skip_llm=False,
            reason=(
                f"water level ratio {ratio:.2f} >= "
                f"{self.NO_REPORT_WATER_LEVEL_RATIO:.0%} of action stage"
            ),
        )


class FinalDecisionAgent:
    """
    Evidence-grounded final decision-support agent.

    Architecture:

        SkillResult
            |
        Evidence Gate
            |
        LLM synthesis
            |
        Output validation
            |
        Final report

    The LLM is NOT allowed to:
        - call tools
        - select operational parameters
        - invent observations
        - invent confidence values
        - invent coordinates
        - invent timestamps
        - invent event IDs
        - invent affected infrastructure
        - infer hazard severity without evidence
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.model = os.getenv(
            "MODEL",
            "deepseek-v4-flash",
        )

        self.base_url = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.deepseek.com/v1",
        )

        self.client = (
            AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            if self.api_key
            else None
        )

    async def synthesize(
        self,
        user_task: str,
        skill_name: str,
        result: SkillResult,
        hazard_rules: str = "",
        output_validator=None,
    ) -> str:

        evidence = [
            item.model_dump()
            for item in result.evidence
        ]

        objects = [
            item.model_dump()
            for item in result.spatial_objects
        ]

        validation_issues = [
            item.model_dump()
            for item in result.validation_issues
        ]

        evidence_packet = _build_compact_packet({
            "task": user_task,
            "skill": skill_name,
            "status": result.status,
            "summary": result.summary,
            "evidence": evidence,
            "spatial_objects": objects,
            "validation_issues": validation_issues,
            "next_actions": result.next_actions,
        })

        # ---------------------------------------------------------
        # Evidence gate
        # ---------------------------------------------------------

        self._validate_evidence_packet(
            evidence_packet
        )

        # ---------------------------------------------------------
        # No LLM available
        # ---------------------------------------------------------

        if not self.client:
            return self._fallback(
                user_task,
                skill_name,
                result,
            )

        # ---------------------------------------------------------
        # LLM synthesis
        # ---------------------------------------------------------

        # Hazard-specific rules are declared by skill plugins
        # (registry.prompt_rules); the shared prompt carries no hazard
        # knowledge, so changing hazards never touches this file.
        system = self._system_prompt()
        if hazard_rules:
            system = (
                f"{system}\n"
                "============================================================\n"
                f"{hazard_rules.strip()}\n"
                "============================================================\n"
            )

        user = (
            "USER TASK:\n"
            f"{user_task}\n\n"
            "VERIFIED SKILL RESULT:\n"
            f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}"
        )
        # Hard-cap safety net: if the prompt is still too long after
        # compaction, truncate rather than fail synthesis.
        if len(user) > _LLM_MAX_PROMPT_CHARS:
            user = (
                user[:_LLM_MAX_PROMPT_CHARS]
                + "\n…[evidence truncated to fit model context]"
            )

        # LLM failures (rate limits, quota, network) must not fail the
        # whole assessment; fall back to the evidence template.
        try:
            response = await self.client.responses.create(
                model=self.model,
                temperature=0,
                instructions=system,
                input=user,
            )
        except Exception as llm_err:
            print(
                f"\nWARNING: LLM synthesis failed "
                f"({type(llm_err).__name__}: {llm_err}). "
                "Falling back to evidence-based template."
            )
            return self._fallback(user_task, skill_name, result)

        output = response.output_text

        if not output:
            raise RuntimeError(
                "Final Decision Agent returned empty output."
            )

        # ---------------------------------------------------------
        # Plain-text enforcement: strip any Markdown the LLM emitted
        # (# headings, - / * bullets, *italic* / **bold**). The final
        # report is consumed as plain text; operators must never see
        # literal markup characters.
        # ---------------------------------------------------------
        output = self._to_plain_text(output)

        # ---------------------------------------------------------
        # Output validation
        # ---------------------------------------------------------
        try:
            self._validate_output(output, evidence_packet)
            # Hazard-specific validation (from skill plugins, e.g.
            # flood forbidden-phrase checks)
            if output_validator is not None:
                output_validator(output, evidence_packet)
        except RuntimeError as e:
            # On validation failure, fall back to the evidence template
            print(f"\nWARNING: Output validation failed: {e}")
            print("Falling back to evidence-based template.")
            return self._fallback(user_task, skill_name, result)

        return output

    @staticmethod
    def _system_prompt() -> str:
        return """
You are the final reporting layer of a real-time disaster
spatial decision-support system.

Your job is to synthesize VERIFIED STRUCTURED EVIDENCE into
a decision-support report.

You are NOT the data acquisition layer.

You MUST follow these rules.

============================================================
EVIDENCE RULES
============================================================

1. Use only facts contained in the supplied SkillResult.

2. Never invent:
   - measurements
   - confidence scores
   - coordinates
   - timestamps
   - station IDs
   - event IDs
   - distances
   - population counts
   - infrastructure impacts
   - hazard extents
   - risk probabilities

3. If confidence is null or absent, DO NOT create a
   numerical confidence score.

4. A point observation is NOT evidence of a city-wide condition.

5. If a spatial relationship has not been computed by the
   Skill, do not infer it yourself.

6. Do not silently fill missing operational parameters.

7. Do not interpret user assumptions as observations.

============================================================
OBSERVATION VS INFERENCE
============================================================

Explicitly distinguish:

OBSERVED:
Facts directly returned by the data source.

VERIFIED INFERENCE:
A conclusion already established by the Skill.

NOT ESTABLISHED:
Claims that cannot be supported by the supplied evidence.

============================================================
CONFIDENCE
============================================================

A confidence value is allowed ONLY if it is explicitly
present in the supplied structured evidence.

If the evidence contains computed decision indices with a
"data_confidence" value, you may cite that exact number as
"Data confidence: X" — it is a pipeline-computed value, not
a fabrication.

If confidence is null:

DO NOT write:
"confidence = 0.95"

DO NOT write:
"confidence = high"

unless the evidence explicitly supports that statement.

You may describe limitations qualitatively.

============================================================
OUTPUT FORMAT
============================================================

PLAIN TEXT ONLY. The report is consumed by operators in plain
terminals and web panels; Markdown must not appear:

- Do NOT use heading markers (#).
- Do NOT use bullet markers (- or *).
- Do NOT use emphasis markers (*italic* or **bold**).

Use numbered sections (1., 2., ...) and complete sentences.
Any Markdown syntax found in the output is automatically
stripped by the reporting layer.

Use this structure:

1. Current situation

2. Verified evidence

3. What is established

4. What is not established

5. Uncertainty and coverage limitations

6. Recommended next actions

7. Human approval required, if applicable

Keep the report factual and operationally useful.
"""

    @staticmethod
    def _validate_evidence_packet(
        packet: dict[str, Any],
    ) -> None:

        if "evidence" not in packet:
            raise ValueError(
                "Evidence packet is missing evidence."
            )

        if "spatial_objects" not in packet:
            raise ValueError(
                "Evidence packet is missing spatial objects."
            )

        for evidence in packet["evidence"]:

            confidence = evidence.get(
                "confidence"
            )

            if confidence is not None:

                if not isinstance(
                    confidence,
                    (int, float),
                ):
                    raise ValueError(
                        "Evidence confidence must be numeric."
                    )

                if not 0 <= confidence <= 1:
                    raise ValueError(
                        "Evidence confidence must be "
                        "between 0 and 1."
                    )

    @staticmethod
    def _to_plain_text(output: str) -> str:
        """
        Strip Markdown syntax from an LLM report.

        Rules (final report must contain no '#', '-', '*' markup):
          - ATX headings: leading '#' runs are removed, text kept.
          - Bullets: leading '- ' / '* ' become '• '.
          - Emphasis: '**bold**' / '*italic*' / '_italic_' markers
            are removed, inner text kept.
        Numeric list markers ("1. ") are plain text and survive.
        """
        lines: list[str] = []
        for raw_line in output.splitlines():
            line = raw_line.rstrip()
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            # Headings: '#...' -> drop the markers, keep the title
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").lstrip()
            # Bullets: '- ' / '* ' -> '• '
            elif stripped.startswith("- ") or stripped.startswith("* "):
                stripped = "• " + stripped[2:]
            lines.append(indent + stripped)
        text = "\n".join(lines)

        # Emphasis markers (multiple passes for nested cases)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"\1", text)
        text = re.sub(r"(?<![\w])_([^_\n]+?)_(?![\w])", r"\1", text)
        # Heading underline style ('===' / '---') is left untouched:
        # '=' is allowed and '---' lines are removed entirely
        text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _validate_output(
        output: str,
        packet: dict[str, Any],
    ) -> None:

        # ---------------------------------------------------------
        # 1. Prevent fabricated confidence
        # ---------------------------------------------------------

        # Two legitimate confidence sources: an evidence item's
        # confidence field, or the pipeline-computed data_confidence in
        # its decision_indices attributes.
        def _has_decision_confidence(item: dict[str, Any]) -> bool:
            attributes = item.get("attributes")
            if not isinstance(attributes, dict):
                return False
            indices = attributes.get("decision_indices")
            return (
                isinstance(indices, dict)
                and indices.get("data_confidence") is not None
            )

        has_confidence = any(
            item.get("confidence") is not None
            or _has_decision_confidence(item)
            for item in packet["evidence"]
        )

        if not has_confidence:

            confidence_patterns = [
                r"\bconfidence\s*[:=]\s*0\.\d+",
                r"\b置信度\s*[:：=]\s*0\.\d+",
                r"\bconfidence\s+(?:is|of)\s+\d+%",
                r"\b置信度\s*(?:为|是)\s*\d+%",
            ]

            for pattern in confidence_patterns:

                if re.search(
                    pattern,
                    output,
                    flags=re.IGNORECASE,
                ):
                    raise RuntimeError(
                        "LLM output contains a numerical "
                        "confidence value that is not "
                        "supported by the evidence."
                    )

        # Hazard-specific forbidden claims (flood stage assertions,
        # city-wide flood assertions, etc.) come from skill plugins via
        # OUTPUT_VALIDATOR; this layer keeps only generic checks.

    @staticmethod
    def _fallback(
        user_task: str,
        skill_name: str,
        result: SkillResult,
    ) -> str:

        lines = [
            f"Task: {user_task}",
            f"Skill: {skill_name}",
            f"Status: {result.status}",
            "",
            "1. Current situation",
            result.summary,
            "",
            "2. Verified evidence",
        ]

        for evidence in result.evidence:

            lines.append(
                f"- [{evidence.source}] "
                f"{evidence.observation}"
            )

            if evidence.timestamp:
                lines.append(
                    f"  Timestamp: {evidence.timestamp}"
                )

        lines.extend(
            [
                "",
                "3. What is established",
                (
                    "Only the observations explicitly "
                    "reported by the verified evidence "
                    "are established."
                ),
                "",
                "4. What is not established",
                (
                    "Claims not supported by the SkillResult "
                    "are not established."
                ),
                "",
                "5. Uncertainty and coverage limitations",
                (
                    "The assessment is limited to the "
                    "available verified observations."
                ),
            ]
        )

        if result.next_actions:

            lines.extend(
                [
                    "",
                    "6. Recommended next actions",
                ]
            )

            lines.extend(
                f"- {action}"
                for action in result.next_actions
            )

        return "\n".join(lines)
    
