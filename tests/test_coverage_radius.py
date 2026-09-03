# tests/test_coverage_radius.py
#
# End-to-end tests for the coverage-radius configuration (real runs use 2 km).
#
# With VULNERABILITY_COVERAGE_RADIUS_KM at 10 km the radius covers the
# whole small town (the 20 demand-tract centroids lie 0.61-6.86 km from
# the nearest facility): every frontier plan reaches full binary
# coverage, VWUN saturates at 0, the Pareto scatter degenerates into a
# horizontal line, and the lambda slider becomes inert. At 2 km (the
# demo/paper convention) the frontier has a real tradeoff ladder.
#
# These tests pin the end-to-end semantics of both configurations on
# the demo pipeline. _DemoEnv uses setdefault semantics: values preset
# via monkeypatch take precedence over the demo defaults.
# Run: python -m pytest tests/test_coverage_radius.py -v

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.demo_fixtures import run_demo_assessment  # noqa: E402


def _ledger(result):
    for ev in result.evidence:
        led = (ev.attributes or {}).get("equity_ledger")
        if isinstance(led, dict):
            return led
    raise AssertionError("demo 管线未产出公平账本")


@pytest.mark.asyncio
async def test_larger_radius_monotonically_shrinks_vwun(monkeypatch):
    """A larger radius only improves coverage: VWUN is monotonically
    non-increasing across plans, with at least one strict decrease.

    (In the demo the fire-station-only plan sits ~10.2 km from tract 1,
    just outside R=10, so the demo is not fully saturated at R=10; the
    real Harvey scene, with max centroid distance 6.86 km, saturates to
    0. The invariant is therefore stated as monotonicity.)
    """
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")
    result2, _state = await run_demo_assessment()
    led2 = _ledger(result2)
    assert led2["coverage_radius_km"] == 2

    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "10")
    result10, _state = await run_demo_assessment()
    led10 = _ledger(result10)
    assert led10["coverage_radius_km"] == 10

    vw2 = {p["plan_id"]: p.get("vulnerability_weighted_unmet_need") or 0.0
           for p in led2.get("frontier_equity_curve") or []}
    vw10 = {p["plan_id"]: p.get("vulnerability_weighted_unmet_need") or 0.0
            for p in led10.get("frontier_equity_curve") or []}
    common = set(vw2) & set(vw10)
    assert common, "两种半径下应有共同的前沿方案"
    for pid in common:
        assert vw10[pid] <= vw2[pid] + 1e-6, (
            f"{pid}: 半径放大只会提高覆盖，VWUN 不应上升 "
            f"({vw2[pid]} -> {vw10[pid]})"
        )
    assert any(vw10[pid] < vw2[pid] for pid in common), (
        "至少一个方案应因覆盖改善而 VWUN 严格下降"
    )
    assert any(vw10[pid] > 0 for pid in common), (
        "R=10 下 demo 仍应有未饱和方案（fire-station-only 距 tract1 ≈10.2 km）"
    )


@pytest.mark.asyncio
async def test_operating_radius_two_gives_real_tradeoff(monkeypatch):
    """R=2 km (operating convention): the frontier has a real tradeoff — some plans at 0, some above 0."""
    monkeypatch.setenv("VULNERABILITY_COVERAGE_RADIUS_KM", "2")
    result, _state = await run_demo_assessment()
    led = _ledger(result)
    assert led["coverage_radius_km"] == 2
    curve = led.get("frontier_equity_curve") or []
    vwuns = [p.get("vulnerability_weighted_unmet_need") for p in curve]
    assert any(v == 0 for v in vwuns), "应存在覆盖全部需求区的方案（VWUN=0）"
    assert any((v or 0) > 0 for v in vwuns), (
        "应存在未全覆盖方案（VWUN>0）—— 效率-公平前沿的权衡阶梯"
    )
