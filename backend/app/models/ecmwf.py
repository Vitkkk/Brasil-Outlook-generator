from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import xarray as xr

from .base_adapter import ModelAdapter
from .gfs import _prepare_cfgrib_group


ECMWF_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200]
ECMWF_PROFILE_PARAMS = ["t", "q", "u", "v", "gh", "w"]
ECMWF_SURFACE_PARAMS = ["2t", "2d", "10u", "10v", "sp", "msl"]


class ECMWFAdapter(ModelAdapter):
    """ECMWF IFS open-data adapter.

    The public IFS 0.25-degree feed is used so the project can run without a
    private ECMWF licence. The adapter intentionally downloads only the fields
    required by the shared severe-weather diagnostics.
    """

    name = "ECMWF"

    def _client(self):
        try:
            from ecmwf.opendata import Client
        except ImportError as exc:
            raise RuntimeError("ECMWF ingestion requires the 'ecmwf-opendata' package") from exc
        return Client(source="ecmwf", model="ifs")

    def latest_cycle(self) -> datetime:
        value = self._client().latest(type="fc", stream="oper", step=0, param="2t")
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def discover_forecast_hours(self, cycle: datetime) -> list[int]:
        # Open IFS has at least 3-hourly deterministic output through Day 1.
        return list(range(0, 25, 3))

    def download(self, cycle: datetime, forecast_hours: Iterable[int], destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        client = self._client()
        date = cycle.strftime("%Y%m%d")
        time = cycle.hour
        paths: list[Path] = []
        for hour in forecast_hours:
            hour = int(hour)
            target = destination / f"ecmwf_{cycle:%Y%m%d%H}_f{hour:03d}.grib2"
            if not target.exists():
                profile = destination / f"ecmwf_{cycle:%Y%m%d%H}_f{hour:03d}_pl.grib2"
                surface = destination / f"ecmwf_{cycle:%Y%m%d%H}_f{hour:03d}_sfc.grib2"
                client.retrieve(
                    date=date, time=time, stream="oper", type="fc", step=hour,
                    param=ECMWF_PROFILE_PARAMS, levelist=ECMWF_PRESSURE_LEVELS,
                    target=str(profile),
                )
                client.retrieve(
                    date=date, time=time, stream="oper", type="fc", step=hour,
                    param=ECMWF_SURFACE_PARAMS, target=str(surface),
                )
                target.write_bytes(profile.read_bytes() + surface.read_bytes())
            paths.append(target)
        return paths

    def open_native(self, paths: Iterable[Path]) -> xr.Dataset:
        try:
            import cfgrib
        except ImportError as exc:
            raise RuntimeError("ECMWF GRIB ingestion requires cfgrib + eccodes") from exc

        groups: list[xr.Dataset] = []
        for path in paths:
            for native in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
                prepared = _prepare_cfgrib_group(native)
                # ECMWF uses short names 2t/2d/10u/10v in some cfgrib builds.
                renames = {}
                for old, new in {
                    "t2m": "temperature_2m", "d2m": "dewpoint_2m",
                    "u10": "u_wind_10m", "v10": "v_wind_10m",
                    "sp": "surface_air_pressure", "msl": "air_pressure_at_mean_sea_level",
                }.items():
                    if old in prepared and new not in prepared:
                        renames[old] = new
                if renames:
                    prepared = prepared.rename(renames)
                groups.append(prepared)
        if not groups:
            raise RuntimeError("No ECMWF GRIB groups decoded")
        merged = xr.merge(groups, compat="override", join="outer")
        merged.attrs.update(model="ECMWF IFS", native_grid="0.25_degree")
        return merged

    def standardize(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = super().standardize(dataset)
        ds.attrs.update(model="ECMWF IFS", native_grid="0.25_degree")
        return ds
