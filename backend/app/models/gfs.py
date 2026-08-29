from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

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


# Keep this manifest deliberately focused on fields needed for the first live
# convective pipeline. Additional levels/features can be added without changing
# the rest of the architecture.
GFS_VARIABLES = (
    "CAPE",
    "CIN",
    "DPT",
    "GUST",
    "HGT",
    "HLCY",
    "PWAT",
    "REFC",
    "RH",
    "TMP",
    "UGRD",
    "USTM",
    "VGRD",
    "VSTM",
    "VVEL",
)

GFS_LEVELS = (
    "surface",
    "2_m_above_ground",
    "10_m_above_ground",
    "3000-0_m_above_ground",
    "6000-0_m_above_ground",
    "180-0_mb_above_ground",
    "90-0_mb_above_ground",
    "1000_mb",
    "925_mb",
    "850_mb",
    "700_mb",
    "500_mb",
    "300_mb",
    "250_mb",
    "0C_isotherm",
    "entire_atmosphere_(considered_as_a_single_layer)",
)


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
            for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
                # forecast time is retained by cfgrib as time + step. Expand a
                # valid_time dimension so multiple forecast hours can be merged.
                if "valid_time" in ds.coords and ds["valid_time"].ndim == 0:
                    valid = ds["valid_time"].values
                    ds = ds.expand_dims(valid_time=[valid])
                groups.append(ds)

        if not groups:
            raise RuntimeError("No GRIB datasets were decoded")

        # Different GRIB level types naturally produce separate xarray groups.
        # Merge what is compatible; downstream selectors understand the level
        # dimensions (isobaricInhPa, heightAboveGround, etc.).
        merged = xr.merge(groups, compat="override", join="outer")
        merged.attrs.update(model="GFS", native_grid="0.25_degree")
        return merged

    def standardize(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = super().standardize(dataset)
        # Common cfgrib short names from GFS. Preserve native variables too when
        # they are useful for diagnostics/proxy hazards.
        rename = {
            "t": "air_temperature",
            "r": "relative_humidity",
            "q": "specific_humidity",
            "u": "eastward_wind",
            "v": "northward_wind",
            "gh": "geopotential_height",
            "w": "lagrangian_tendency_of_air_pressure",
            "sp": "surface_air_pressure",
            "prmsl": "air_pressure_at_mean_sea_level",
            "t2m": "temperature_2m",
            "d2m": "dewpoint_2m",
            "u10": "u_wind_10m",
            "v10": "v_wind_10m",
            "cape": "native_cape",
            "cin": "native_cin",
            "hlcy": "native_helicity",
            "pwat": "precipitable_water",
            "refc": "composite_reflectivity",
            "gust": "surface_gust",
            "ustm": "storm_motion_u",
            "vstm": "storm_motion_v",
        }
        available = {
            old: new for old, new in rename.items() if old in ds and new not in ds
        }
        if available:
            ds = ds.rename(available)
        return ds
