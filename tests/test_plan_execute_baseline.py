"""Plan-then-Execute baseline — third condition of the §4.3 comparison.

The PE scaffold plans the full eight-call chain UPFRONT and executes it
without re-planning; scoring conventions are the naive pipeline's.  The
tests pin both properties: identical outputs to the naive column in all
six scenarios (S0–S5, same FakeMCP fixtures, same fault injections), and
the zero-replan / fault-propagation instrumentation quoted in the paper.
Run verbosely to see the three-condition table:

    uv run python -m pytest tests/test_plan_execute_baseline.py -s -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.demo_fixtures import FakeMCP  # noqa: E402
from app.plan_execute_baseline import PlanExecuteBaseline  # noqa: E402
from test_naive_baseline import (  # noqa: E402
    SCENARIOS,
    _apply_fault,
    _print_row,
    _run_naive,
    _run_ours,
    _scenario_args,
)

# Fields both constructed baselines emit (identical by construction —
# that identity IS the finding; see the module docstring).
SHARED_FIELDS = (
    "cdri_percent",
    "cdri_label",
    "band",
    "eps",
    "data_confidence",
    "exposed_population",
    "total_population",
    "recommended_plan",
    "allocation_emitted",
    "vwun_recommended",
    "station_distance_km",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_plan_execute_matches_naive_and_never_replans(
    tmp_path, monkeypatch, scenario
):
    mcp = FakeMCP()
    _apply_fault(mcp, scenario)
    target, station, _ = _scenario_args(scenario)
    pe = await PlanExecuteBaseline(coverage_radius_km=2.0).run(
        mcp, target=target, station_id=station
    )
    naive = await _run_naive(scenario)
    ours = await _run_ours(monkeypatch, tmp_path, scenario)
    _print_row(scenario, naive, ours)
    print(f"  PE     : CDRI {pe['cdri_percent']}% ({pe['cdri_label']}) | "
          f"plan {pe['plan_steps']} steps, replans {pe['replans']}, "
          f"first_fail {pe['first_failed_step']}, "
          f"steps_after {pe['steps_after_first_failure']}, "
          f"arg_defaults {pe['arg_default_substitutions']} | "
          f"alloc={pe['allocation_emitted']} | disclosed "
          f"{len(pe['disclosed_issues'])} / silent {len(pe['silent_issues'])}")

    # -- scaffold identity: same outputs as the naive column ----------
    for field in SHARED_FIELDS:
        n, p = naive[field], pe[field]
        if isinstance(n, float):
            assert p == pytest.approx(n, abs=1e-9), (
                f"[{scenario}] {field}: PE {p} != naive {n}"
            )
        else:
            assert p == n, f"[{scenario}] {field}: PE {p!r} != naive {n!r}"
    assert pe["silent_issues"] == naive["silent_issues"]

    # -- scaffold contract: full plan, zero re-planning, no disclosure -
    assert pe["plan_steps"] == 8
    assert pe["replans"] == 0
    assert pe["disclosed_issues"] == []

    if scenario == "S1-metadata-503":
        assert pe["first_failed_step"] == "metadata"
        # nwps, extent, svi, infra, resources execute on broken context
        assert pe["steps_after_first_failure"] == 5
        assert pe["allocation_emitted"] is True
    elif scenario == "S2-no-nwps":
        assert pe["first_failed_step"] == "nwps"
        assert pe["steps_after_first_failure"] == 4
        assert "degraded" not in pe["cdri_label"]  # confident Low, silently
    elif scenario == "S3-far-station":
        # mis-resolved anchor propagates through every planned spatial call
        assert pe["first_failed_step"] is None
        assert pe["station_distance_km"] > 1000
        assert pe["allocation_emitted"] is True
    elif scenario == "S4-no-extent":
        # tool-level ok (a dry scene is a valid observation); the collapse
        # happens in the scoring convention: exposed 0, VWUN 0
        assert pe["first_failed_step"] is None
        assert pe["exposed_population"] == 0
        assert pe["vwun_recommended"] == 0
    elif scenario == "S5-stale-sar":
        assert pe["first_failed_step"] is None
        assert pe["exposed_population"] > 0  # misaligned scene consumed
