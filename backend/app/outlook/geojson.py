from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import xarray as xr
from shapely.geometry import box, mapping
from shapely.ops import unary_union


def _coord_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Latitude/longitude coordinates must be 1-D with >=2 points")
    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def probability_field_to_geojson(
    field: xr.DataArray,
    thresholds: Iterable[float],
    *,
    product: str,
    valid_start: datetime,
    valid_end: datetime,
    distance_km: float = 40.0,
    simplify_tolerance_deg: float = 0.05,
    min_area_deg2: float = 0.04,
) -> dict:
    """Convert a lat/lon probability grid into nested threshold polygons.

    The MVP builds grid-cell polygons and unions them by threshold. Production
    versions can replace this with contour-based polygonization while keeping
    the same GeoJSON contract.
    """

    required = {"latitude", "longitude"}
    if not required.issubset(field.coords):
        raise ValueError("field must contain latitude and longitude coordinates")

    da = field.transpose("latitude", "longitude")
    values = np.asarray(da.values, dtype=float)
    lats = np.asarray(da.latitude.values, dtype=float)
    lons = np.asarray(da.longitude.values, dtype=float)

    if np.any(np.diff(lats) < 0):
        da = da.sortby("latitude")
        values = np.asarray(da.values, dtype=float)
        lats = np.asarray(da.latitude.values, dtype=float)
    if np.any(np.diff(lons) < 0):
        da = da.sortby("longitude")
        values = np.asarray(da.values, dtype=float)
        lons = np.asarray(da.longitude.values, dtype=float)

    lat_edges = _coord_edges(lats)
    lon_edges = _coord_edges(lons)

    features = []
    for threshold in sorted({float(x) for x in thresholds}):
        mask = np.isfinite(values) & (values >= threshold)
        cells = []
        for iy, ix in np.argwhere(mask):
            cells.append(
                box(
                    lon_edges[ix],
                    lat_edges[iy],
                    lon_edges[ix + 1],
                    lat_edges[iy + 1],
                )
            )
        if not cells:
            continue

        geom = unary_union(cells)
        if simplify_tolerance_deg > 0:
            geom = geom.simplify(simplify_tolerance_deg, preserve_topology=True)

        if geom.geom_type == "Polygon":
            geoms = [geom]
        else:
            geoms = list(getattr(geom, "geoms", []))

        kept = [g for g in geoms if g.area >= min_area_deg2]
        if not kept:
            continue
        merged = unary_union(kept)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "product": product,
                    "probability": threshold,
                    "distance_km": distance_km,
                    "valid_start": _iso(valid_start),
                    "valid_end": _iso(valid_end),
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "product": product,
            "valid_start": _iso(valid_start),
            "valid_end": _iso(valid_end),
            "distance_km": distance_km,
        },
    }


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
