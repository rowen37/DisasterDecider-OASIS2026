import json

import pytest

from app.demo_fixtures import FakeMCP
from app.threshold_dashboard_baseline import ThresholdDashboardBaseline


@pytest.mark.asyncio
async def test_dashboard_baseline_uses_authoritative_gauge_thresholds():
    result = await ThresholdDashboardBaseline().run(FakeMCP(), "08077600")

    assert result["status"] == "ok"
    assert result["hydrologic_category"] == "action"
    assert "capacity-aware transfer planning" in result["out_of_scope"]


@pytest.mark.asyncio
async def test_dashboard_baseline_discloses_missing_thresholds():
    mcp = FakeMCP()

    async def missing(_args):
        return json.dumps({"status": "error", "error": "not available"})

    mcp._fx_get_nwps_gauge = missing
    result = await ThresholdDashboardBaseline().run(mcp, "08077600")

    assert result["status"] == "unavailable"
    assert "authoritative_flood_categories" in result["data_gaps"]
