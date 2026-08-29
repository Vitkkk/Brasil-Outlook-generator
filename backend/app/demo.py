from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr

from .config import get_config


def _gaussian(lon2d, lat2d, lon0, lat0, sx, sy, amplitude):
    return amplitude * np.exp(
        -0.5 * (((lon2d - lon0) / sx) ** 2 + ((lat2d - lat0) / sy) ** 2)
    )


def build_synthetic_probability_fields() -> xr.Dataset:
    """Synthetic fields for exercising the pipeline before live model ingestion.

    These are deliberately labelled demo data and must never be interpreted as
    a real forecast. The synthetic grid is intentionally coarser than the
    configured operational grid so GeoJSON polygonization stays lightweight.
    """

    cfg = get_config().domain
    step = max(cfg.grid_resolution_deg, 0.25)
    lats = np.arange(cfg.south, cfg.north + step, step)
    lons = np.arange(cfg.west, cfg.east + step, step)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Smooth synthetic maxima over southern/central South America make it
    # possible to test nested polygons and UI rendering without external data.
    core = _gaussian(lon2d, lat2d, -52.5, -27.5, 4.5, 3.0, 0.55)
    secondary = _gaussian(lon2d, lat2d, -59.0, -34.0, 3.8, 2.5, 0.24)

    thunder = np.clip(0.05 + 1.35 * (core + secondary), 0, 0.92)
    severe = np.clip(core + secondary, 0, 0.72)
    tornado = np.clip(core * 0.22 + secondary * 0.05, 0, 0.30)
    hail = np.clip(core * 0.70 + secondary * 0.42, 0, 0.60)
    wind = np.clip(core * 0.58 + secondary * 0.55, 0, 0.60)

    ds = xr.Dataset(
        {
            "thunderstorm": (("latitude", "longitude"), thunder),
            "severe": (("latitude", "longitude"), severe),
            "tornado": (("latitude", "longitude"), tornado),
            "hail": (("latitude", "longitude"), hail),
            "wind": (("latitude", "longitude"), wind),
        },
        coords={"latitude": lats, "longitude": lons},
    )
    ds.attrs["source"] = "synthetic_mvp"
    ds.attrs["grid_resolution_deg"] = step
    return ds


def demo_valid_period() -> tuple[datetime, datetime]:
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return start, start + timedelta(hours=24)
