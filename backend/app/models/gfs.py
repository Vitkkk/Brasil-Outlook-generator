from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from ..config import get_config
from ..ingestion.nomads import (
    DomainBox,
    build_gfs_filter_url,
    discover_gfs_forecast_hours,
    discover_latest_gfs_cycle,
    download_url,
)
from .base_adapter import ModelAdapter


# Live severe-convective manifest. V3 adds pressure-level specific humidity and a
# denser vertical pressure grid so parcel thermodynamics can be reconstructed
# from the model sounding instead of relying on native CAPE/CIN products.
GFS_VARIABLES = (
    "CAPE",
    "CIN",
    "DPT",
    "GUST",
    "HGT",
    "HLCY",
    "PRES",
    "PWAT",
    "REFC",
    "RH",
    "SPFH",
    "TMP",
    "UGRD",
    "USTM",
    "VGRD",
    "VSTM",
    "VVEL",
)

# 50-hPa spacing through most of the troposphere, with extra 25-hPa levels near
# the surface, gives substantially better parcel/LCL/CIN reconstruction while
# keeping the South-America NOMADS subset small enough for operational testing.
GFS_LEVELS = (
    "surface",
    "2_m_above_ground",
    "10_m_above_ground",
    "3000-0_m_above_ground",
    "6000-0_m_above_ground",
    "1000_mb",
    "975_mb",
    "950_mb",
    "925_mb",
    "900_mb",
    "850_mb",
    "800_mb",
    "750_mb",
    "700_mb",
    "650_mb",
    "600_mb",
    "550_mb",
    "500_mb",
    "450_mb",
    "400_mb",
    "350_mb",
    "300_mb",
    "250_mb",
    "200_mb",
    "entire_atmosphere_(considered_as_a_single_layer)",
)


def _scalar_coord_value(ds: xr.Dataset, name: str) -> float | None:
    if name not in ds.coords:
        return None
    value = ds.coords[name]
    if value.size != 1:
        return None
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _rename_existing(ds: xr.Dataset, mapping: dict[str, str]) -> xr.Dataset:
    usable = {old: new for old, new in mapping.items() if old in ds and new not in ds}
    return ds.rename(usable) if usable else ds


def _drop_scalar_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    # Once a scalar 2-m/10-m/surface/layer field receives an unambiguous name,
    # keeping its scalar vertical coordinate only creates merge conflicts.
    for name in (
        "heightAboveGround",
        "heightAboveGroundLayer",
        "pressureFromGroundLayer",
        "surface",
        "entireAtmosphere",
        "atmosphereSingleLayer",
    ):
        if name in ds.coords and ds.coords[name].size == 1 and name not in ds.dims:
            ds = ds.drop_vars(name)
    return ds


def _prepare_cfgrib_group(ds: xr.Dataset) -> xr.Dataset:
    """Give every useful cfgrib group collision-free standardized names."""
    if "valid_time" in ds.coords and ds["valid_time"].ndim == 0:
        valid = ds["valid_time"].values
        ds = ds.expand_dims(valid_time=[valid])

    # Pressure-level profile group: retain the pressure dimension. Specific
    # humidity is the preferred moisture variable for V3 thermodynamics; RH is
    # retained as a fallback and for later diagnostics.
    if "isobaricInhPa" in ds.dims or "isobaricInPa" in ds.dims:
        ds = _rename_existing(
            ds,
            {
                "t": "air_temperature",
                "r": "relative_humidity",
                "q": "specific_humidity",
                "u": "eastward_wind",
                "v": "northward_wind",
                "gh": "geopotential_height",
                "w": "lagrangian_tendency_of_air_pressure",
            },
        )
        return ds

    height = _scalar_coord_value(ds, "heightAboveGround")
    if height is not None and abs(height - 2.0) < 0.25:
        ds = _rename_existing(
            ds,
            {
                "t": "temperature_2m",
                "t2m": "temperature_2m",
                "dpt": "dewpoint_2m",
                "d2m": "dewpoint_2m",
                "2t": "temperature_2m",
                "2d": "dewpoint_2m",
                "q": "specific_humidity_2m",
            },
        )
        return _drop_scalar_vertical_coords(ds)

    if height is not None and abs(height - 10.0) < 0.25:
        ds = _rename_existing(
            ds,
            {
                "u": "u_wind_10m",
                "v": "v_wind_10m",
                "u10": "u_wind_10m",
                "v10": "v_wind_10m",
                "10u": "u_wind_10m",
                "10v": "v_wind_10m",
            },
        )
        return _drop_scalar_vertical_coords(ds)

    # These fields have unique short names but are spread across several GRIB
    # level types. Surface HGT is exposed as terrain_height so diagnostics can
    # interpolate pressure-level profiles to AGL heights.
    ds = _rename_existing(
        ds,
        {
            "cape": "native_cape",
            "cin": "native_cin",
            "hlcy": "native_helicity",
            "pwat": "precipitable_water",
            "refc": "composite_reflectivity",
            "gust": "surface_gust",
            "ustm": "storm_motion_u",
            "vstm": "storm_motion_v",
            "orog": "terrain_height",
            "gh": "terrain_height",
            "sp": "surface_air_pressure",
            "pres": "surface_air_pressure",
            "prmsl": "air_pressure_at_mean_sea_level",
        },
    )
    return _drop_scalar_vertical_coords(ds)


class GFSAdapter(ModelAdapter):
    name = "GFS"

    def latest_cycle(self) -> datetime:
        return discover_latest_gfs_cycle()

    def discover_forecast_hours(self, cycle: datetime) -> list[int]:
        return discover_gfs_forecast_hours(cycle)

    def _domain(self) -> DomainBox:
        cfg = get_config().domain
        return DomainBox(
            north=cfg.north,
            south=cfg.south,
            west=cfg.west,
            east=cfg.east,
        )

    def subset_url(self, cycle: datetime, forecast_hour: int) -> str:
        return build_gfs_filter_url(
            cycle,
            forecast_hour,
            variables=GFS_VARIABLES,
            levels=GFS_LEVELS,
            domain=self._domain(),
        )

    def download(
        self,
        cycle: datetime,
        forecast_hours: Iterable[int],
        destination: Path,
    ) -> list[Path]:
        result: list[Path] = []
        for forecast_hour in forecast_hours:
            target = destination / (
                f"gfs_{cycle:%Y%m%d%H}_f{int(forecast_hour):03d}_samerica.grib2"
            )
            if not target.exists():
                download_url(self.subset_url(cycle, int(forecast_hour)), target)
            result.append(target)
        return result

    def open_native(self, paths: Iterable[Path]) -> xr.Dataset:
        try:
            import cfgrib
        except ImportError as exc:
            raise RuntimeError(
                "GFS GRIB ingestion requires the optional 'grib' dependencies "
                "(cfgrib + eccodes)."
            ) from exc

        groups: list[xr.Dataset] = []
        for path in paths:
            native_groups = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
            groups.extend(_prepare_cfgrib_group(ds) for ds in native_groups)

        if not groups:
            raise RuntimeError("No GRIB datasets were decoded")

        merged = xr.merge(groups, compat="override", join="outer")
        merged.attrs.update(model="GFS", native_grid="0.25_degree")
        return merged

    def standardize(self, dataset: xr.Dataset) -> xr.Dataset:
        # Most useful fields are already standardized group-by-group so that 10-m
        # wind and pressure-level wind cannot overwrite each other. The base
        # adapter still normalizes latitude/longitude conventions here.
        ds = super().standardize(dataset)
        ds.attrs.update(model="GFS", native_grid="0.25_degree")
        return ds
