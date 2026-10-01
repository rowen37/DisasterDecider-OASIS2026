from mcp_servers.osm_labels import osm_display_label


def test_osm_address_replaces_numeric_display_id():
    result = osm_display_label(
        {"addr:housenumber": "4125", "addr:street": "Chestnut Street"},
        feature_type="building",
        osm_type="way",
        osm_id=123,
        lat=39.95,
        lon=-75.20,
    )
    assert result["label"] == "4125 Chestnut Street"
    assert "123" not in result["label"]


def test_verified_override_takes_precedence():
    result = osm_display_label(
        {"name": "Old name"},
        feature_type="building",
        osm_type="way",
        osm_id=123,
        lat=39.95,
        lon=-75.20,
        overrides={"way/123": {"name": "Uno on Chestnut", "address": "4125 Chestnut Street"}},
    )
    assert result["label"] == "Uno on Chestnut"
    assert result["address"] == "4125 Chestnut Street"
    assert result["label_source"] == "verified_local_override"
