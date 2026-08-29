from datetime import datetime, timezone

import numpy as np
import xarray as xr

from app.outlook.geojson import probability_field_to_geojson


def test_probability_grid_generates_valid_feature_collection():
    field = xr.DataArray(
        np.array(
            [
                [0.00, 0.02, 0.02],
                [0.00, 0.05, 0.10],
                [0.00, 0.05, 0.10],
            ]
        ),
        dims=("latitude", "longitude"),
        coords={"latitude": [-30.0, -29.9, -29.8], "longitude": [-52.0, -51.9, -51.8]},
    )

    start = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    end = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
    geojson = probability_field_to_geojson(
        field,
        [0.02, 0.05, 0.10],
        product="tornado",
        valid_start=start,
        valid_end=end,
        min_area_deg2=0.0,
        simplify_tolerance_deg=0.0,
    )

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 3
    assert {f["properties"]["probability"] for f in geojson["features"]} == {
        0.02,
        0.05,
        0.10,
    }
    for feature in geojson["features"]:
        assert feature["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        assert feature["properties"]["distance_km"] == 40.0
