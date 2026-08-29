from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .categories import CATEGORY_CODE
from .land_mask import clip_geometry_to_land


def _edges(values: np.ndarray) -> np.ndarray:
    mids = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - (mids[0] - values[0])], mids, [values[-1] + (values[-1] - mids[-1])])
    )


def _coherent_geometry(cells, *, bridge_gap_deg, simplify_tolerance_deg, min_area_deg2, land_only):
    geom = unary_union(cells)
    if bridge_gap_deg > 0:
        radius = bridge_gap_deg / 2.0
        geom = geom.buffer(radius, join_style=1).buffer(-radius, join_style=1)
    if land_only:
        geom = clip_geometry_to_land(geom)
    if simplify_tolerance_deg > 0 and not geom.is_empty:
        geom = geom.simplify(simplify_tolerance_deg, preserve_topology=True)
    if geom.is_empty:
        return None
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    parts = [p for p in parts if p.geom_type == "Polygon" and p.area >= min_area_deg2]
    return unary_union(parts) if parts else None


def categorical_field_to_geojson(
    field: xr.DataArray,
    *,
    valid_start: datetime,
    valid_end: datetime,
    simplify_tolerance_deg: float = 0.05,
    min_area_deg2: float = 0.04,
    bridge_gap_deg: float = 0.0,
    land_only: bool = False,
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
        merged = _coherent_geometry(
            cells,
            bridge_gap_deg=bridge_gap_deg,
            simplify_tolerance_deg=simplify_tolerance_deg,
            min_area_deg2=min_area_deg2,
            land_only=land_only,
        )
        if merged is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "product": "categorical",
                    "category": category,
                    "category_code": code,
                    "valid_start": _iso(valid_start),
                    "valid_end": _iso(valid_end),
                    "land_only": land_only,
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
            "land_only": land_only,
        },
    }


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
