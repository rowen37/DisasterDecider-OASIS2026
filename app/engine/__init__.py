"""Engine layer — pure computation, no I/O, no MCP, no state mutation.

Engines take explicit inputs and return results.  Skills orchestrate data
flow and own all side effects (state logs, evidence objects, HITL).

Naming convention: hazard-specific modules are ``<hazard>_<role>_engine``,
shared modules are ``<role>_engine``; add engines for new hazards under
the same convention (see docs/EXTENDING.md).

Hazard-specific:
    flood_fusion_engine   flood multi-source observation fusion
    flood_risk_engine     flood CDRI / EPS / DataConfidence indices

Shared:
    geometry_engine       spatial clipping, areal-weighted exposure
    allocation_engine     resource plans, Pareto optimization, equity objective
    pareto_engine         multi-objective Pareto frontier and plan selection
"""

from .allocation_engine import AllocationEngine  # noqa: F401
from .flood_fusion_engine import (  # noqa: F401
    fuse_flood_evidence,
    normalize_fusion_observation,
)
from .geometry_engine import (  # noqa: F401
    ClipResult,
    GeometryEngine,
)
from .pareto_engine import (  # noqa: F401
    pareto_frontier,
    select_best_pareto_plan,
)
from .flood_risk_engine import RiskEngine  # noqa: F401

__all__ = [
    "AllocationEngine",
    "ClipResult",
    "GeometryEngine",
    "RiskEngine",
    "fuse_flood_evidence",
    "normalize_fusion_observation",
    "pareto_frontier",
    "select_best_pareto_plan",
]
