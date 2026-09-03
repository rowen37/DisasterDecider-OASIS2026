from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


# Registered hazard types live in app/skills/registry.py; this Literal
# describes the API-level values (to add one: add a line here plus a
# registered plugin).
HazardType = Literal[
    "flood",
]


class Location(BaseModel):
    name: str
    latitude: float
    longitude: float
    display_name: str | None = None
    source: str = "unknown"


class SpatialObject(BaseModel):
    object_id: str
    object_type: str

    geometry: dict[str, Any] | None = None

    crs: str | None = None

    source: str | None = None

    timestamp: str | None = None

    # This is intentionally optional.
    #
    # Do not assign a probability unless a statistical
    # model actually produces one.
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    attributes: dict[str, Any] = Field(
        default_factory=dict
    )


class ValidationIssue(BaseModel):
    severity: Literal[
        "warning",
        "error",
        "critical",
    ]

    code: str

    message: str

    field: str | None = None


class VerificationResult(BaseModel):
    passed: bool

    issues: list[ValidationIssue] = Field(
        default_factory=list
    )


class HITLRequest(BaseModel):
    reason: str

    question: str

    proposed_value: Any | None = None

    context: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    evidence_id: str
    source: str
    observation: str
    timestamp: str | None = None
    quality_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    attributes: dict[str, Any] = Field(default_factory=dict)

class SkillResult(BaseModel):
    status: Literal[
        "completed",
        "no_event",
        "needs_human",
        "error",
    ]

    summary: str

    evidence: list[Evidence] = Field(default_factory=list)
    spatial_objects: list[SpatialObject] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    gis_map_path: str | None = None

@dataclass
class RunState:

    run_id: str

    hazard_type: HazardType | None = None

    spatial_objects: dict[str,SpatialObject,] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    hitl_requests: list[HITLRequest] = field(default_factory=list)
    gis_results: dict[str, Any] = field(default_factory=dict)

    def add_object(self,obj: SpatialObject,) -> None:
        self.spatial_objects[obj.object_id] = obj

    def log(self,event: str,**payload: Any,) -> None:
        self.events.append({"event": event,**payload,})