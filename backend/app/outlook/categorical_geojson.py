from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .categories import CATEGORY_CODE


def _edges(values: np.ndarray) -> np.ndarray:
    mids = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - (mids[0] - values[0])], mids, [values[-1] + (values[-1] - mids[-1])])
    )


def categorical_field_to_geojson(
    field: xr.DataArray,
    *,
    valid_start: datetime,
    valid_end: datetime,
    simplify_tolerance_deg: float = 0.05,
    min_area_deg2: float = 0.04,
) -> dict:
    da = field.transpose("latitude", "longitude").sortby("latitude").sortby("longitude")
    values = np.asarray(da.values, dtype=int)
    lats = np.asarray(da.latitude.values, dtype=float)
    lons = np.asarray(da.longitude.values, dtype=float)
    lat_edges, lon_edges = _edges(lats), _edges(lons)

    features = []
    for category in ["TSTM", "MRGL", "SLGT", "ENH", "MDT", "HIGH"]:
        code = CATEGORY_CODE[category]
        mask = values >= code
        cells = [
            box(lon_edges[ix], lat_edges[iy], lon_edges[ix + 1], lat_edges[iy + 1])
            for iy, ix in np.argwhere(mask)
        ]
        if not cells:
            continue
        merged = unary_union(cells).simplify(simplify_tolerance_deg, preserve_topology=True)
        parts = [merged] if merged.geom_type == "Polygon" else list(getattr(merged, "geoms", []))
        parts = [part for part in parts if part.area >= min_area_deg2]
        if not parts:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(unary_union(parts)),
                "properties": {
                    "product": "categorical",
                    "category": category,
                    "category_code": code,
                    "valid_start": _iso(valid_start),
                    "valid_end": _iso(valid_end),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "product": "categorical",
            "valid_start": _iso(valid_start),
            "valid_end": _iso(valid_end),
        },
    }


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
