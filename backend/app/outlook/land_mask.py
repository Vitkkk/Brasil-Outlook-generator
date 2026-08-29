from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr
from shapely import contains_xy
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@lru_cache(maxsize=1)
def land_geometry() -> BaseGeometry:
    """Return a 50-m Natural Earth land union.

    Cartopy is kept as an optional geospatial dependency and imported lazily so
    the core API can still start without it. Operational map/GeoJSON workflows
    install the geospatial extras and therefore receive coastline-aware clipping.
    """
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError as exc:
        raise RuntimeError(
            "Land-only outlook products require the optional geospatial dependencies "
            "(install with `pip install -e '.[geospatial]'`)."
        ) from exc

    path = shpreader.natural_earth(
        resolution="50m",
        category="physical",
        name="land",
    )
    reader = shpreader.Reader(path)
    return unary_union(list(reader.geometries()))


def mask_dataarray_to_land(field: xr.DataArray) -> xr.DataArray:
    """Mask a latitude/longitude field to physical land using cell centres."""
    if "latitude" not in field.coords or "longitude" not in field.coords:
        raise ValueError("field must contain latitude and longitude coordinates")

    da = field.transpose("latitude", "longitude")
    lon2d, lat2d = np.meshgrid(
        np.asarray(da.longitude, dtype=float),
        np.asarray(da.latitude, dtype=float),
    )
    geometry = land_geometry()
    mask = contains_xy(geometry, lon2d, lat2d)
    mask_da = xr.DataArray(
        mask,
        dims=("latitude", "longitude"),
        coords={"latitude": da.latitude, "longitude": da.longitude},
    )
    return da.where(mask_da)


def clip_geometry_to_land(geometry: BaseGeometry) -> BaseGeometry:
    """Clip an already-polygonized outlook geometry to physical land."""
    if geometry.is_empty:
        return geometry
    return geometry.intersection(land_geometry())
