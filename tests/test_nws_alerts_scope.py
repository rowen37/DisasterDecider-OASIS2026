import json

import httpx
import pytest

from mcp_servers import nws_alerts


@pytest.mark.asyncio
async def test_active_alerts_query_applies_to_target_point(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "features": [
                {"properties": {"event": "Flood Warning"}},
                {"properties": {"event": "Wind Advisory"}},
            ]
        })

    transport = httpx.MockTransport(respond)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        nws_alerts.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    data = json.loads(await nws_alerts.get_flood_warnings(
        40.8823215, -74.0831971, radius_km=50
    ))

    assert len(requests) == 1
    assert requests[0].url.params["point"] == "40.8823,-74.0832"
    assert "area" not in requests[0].url.params
    assert data["scope"] == "target_point"
    assert data["filtered_count"] == 1
