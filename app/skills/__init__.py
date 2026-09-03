"""Skills layer — orchestration and atomic data-acquisition units.

Skills own data flow, HITL interaction and evidence objects.  All pure
computation lives in the Engine layer (``app.engine``); infrastructure
lives in the Ops layer (``app.ops``).

Hazard plugins: new hazards register via ``@register_skill`` (see
registry.py and docs/EXTENDING.md); ``main.py`` dispatches through the
``create_skill`` factory with no if/elif.

- ``flood_skill``          FloodSkill (flood impact assessment orchestration, hazard plugin)
- ``geocode_skill``        GeocodeSkill (geocoding, atomic)
- ``usgs_skill``           UsgsSkill (USGS/NWPS station data, atomic)
- ``fusion_sources_skill`` FusionSourcesSkill (concurrent multi-source collection, atomic)
- ``svi_skill``            SviSkill (social vulnerability, atomic)
- ``resource_skill``       ResourceSkill (emergency resource discovery, atomic)
- ``master_router``        MasterRouter (registry-driven task routing)
- ``registry``             hazard plugin registry
"""

from .flood_skill import FloodSkill
from .fusion_sources_skill import FusionSourcesSkill
from .geocode_skill import GeocodeSkill
from .master_router import MasterRouter
from .registry import register_skill
from .resource_skill import ResourceSkill
from .svi_skill import SviSkill
from .usgs_skill import UsgsSkill

__all__ = [
    "FloodSkill",
    "FusionSourcesSkill",
    "GeocodeSkill",
    "MasterRouter",
    "ResourceSkill",
    "SviSkill",
    "UsgsSkill",
    "register_skill",
]
