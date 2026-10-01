import pytest


@pytest.mark.asyncio
async def test_slider_preview_recomputes_each_plan_at_requested_lambda(monkeypatch):
    import uvicorn

    from app import main

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.setdefault("app", app))
    main.start_web_server()
    endpoint = next(
        route.endpoint
        for route in captured["app"].routes
        if getattr(route, "path", None) == "/api/equity/sensitivity"
    )

    a = [{"tract_id": "t", "exposed_population": 100, "coverage": 0.5, "svi": 0.5}]
    b = [{"tract_id": "t", "exposed_population": 100, "coverage": 0.1, "svi": 0.5}]
    run_id = "slider-lambda-regression"
    main._remember_equity_context(run_id, {
        "run_id": run_id,
        "demand_impacts": a,
        "equity_threshold": 0.9,
        "sensitivity_context": {
            "frontier_plan_impacts": {"a": a, "b": b},
            "frontier_plans": [
                {"plan_id": "a", "objectives": {"vulnerability_coverage": 0.5, "cost": 10}},
                {"plan_id": "b", "objectives": {"vulnerability_coverage": 0.1, "cost": 5}},
            ],
            "optimization_objectives": [
                {"name": "vulnerability_coverage", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ],
            "base_objective_weights": {"vulnerability_coverage": 0.2, "cost": 0.8},
            "population_weighted_svi": 0.5,
            "recommended_plan_id": "a",
        },
    })
    try:
        response = await endpoint({"run_id": run_id, "vulnerability_weight": 1.9})
    finally:
        main._equity_contexts.pop(run_id, None)

    assert response["vulnerability_weighted_unmet_need"] == 97.5
    assert response[
        "normalized_vulnerability_weighted_unmet_need"
    ] == 0.5
    assert response["frontier_preview"]["vwun_by_plan"] == {
        "a": 97.5,
        "b": 175.5,
    }
    assert response["frontier_preview"]["normalized_vwun_by_plan"] == {
        "a": 0.5,
        "b": 0.9,
    }
