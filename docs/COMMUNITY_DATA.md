# Community requirement data

The agent separates two kinds of data:

1. Event requirements are selected in the dashboard for the current run.
2. Facility capabilities are relatively static facts stored in
   `config/community_facilities.json` or derived from explicit OpenStreetMap
   facility tags.

There is no live community-data interface. An unchecked event requirement
means “not supplied for this run,” not “the community does not need it.” A
missing facility capability is reported as `unknown`, not `false`.

The resulting community-compatible plan is advisory. It is shown as a
separate operator choice and does not silently replace the model's main
recommendation. The default registry path is
`config/community_facilities.json`; set `COMMUNITY_FACILITIES_PATH` only when
deploying a separately maintained registry.

## Static facility record

Each entry must identify a facility by its stable OSM id or exact name and
must include a human-readable source:

```json
{
  "schema_version": 1,
  "facilities": [
    {
      "facility_id": "osm:node:123456",
      "name": "Example Community Center",
      "source": "County emergency shelter registry, 2026 edition",
      "source_url": "https://example.gov/shelters/123456",
      "last_verified_at": "2026-09-01",
      "verification_status": "verified",
      "capabilities": {
        "temporary_shelter": true,
        "overnight_shelter": true,
        "wheelchair_accessible": true,
        "family_support": null,
        "multilingual_support": false
      }
    }
  ]
}
```

`true` means confirmed available, `false` means confirmed unavailable, and
`null` means unknown. Supported capability keys are defined in
`app/community.py`. Do not copy the example facility into the active config;
only add locally verified records.
