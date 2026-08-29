from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import xarray as xr


STANDARD_VARIABLES = {
    "temperature": "air_temperature",
    "dewpoint": "dew_point_temperature",
    "relative_humidity": "relative_humidity",
    "specific_humidity": "specific_humidity",
    "u_wind": "eastward_wind",
    "v_wind": "northward_wind",
    "geopotential_height": "geopotential_height",
    "omega": "lagrangian_tendency_of_air_pressure",
    "surface_pressure": "surface_air_pressure",
    "mean_sea_level_pressure": "air_pressure_at_mean_sea_level",
}


@dataclass(slots=True)
class ModelRunInfo:
    model: str
    cycle: datetime
    available_forecast_hours: list[int]
    missing_variables: list[str]
    complete: bool


class ModelAdapter(ABC):
    """Interface every numerical model adapter must implement.

    Adapters are responsible only for obtaining/decoding native model data and
    exposing a standardized xarray Dataset. Meteorological diagnostics belong in
    downstream modules so all models are treated consistently.
    """

    name: str
    critical_variables: tuple[str, ...] = (
        "air_temperature",
        "relative_humidity",
        "eastward_wind",
        "northward_wind",
        "geopotential_height",
        "surface_air_pressure",
    )

    @abstractmethod
    def latest_cycle(self) -> datetime:
        raise NotImplementedError

    @abstractmethod
    def discover_forecast_hours(self, cycle: datetime) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def download(
        self,
        cycle: datetime,
        forecast_hours: Iterable[int],
        destination: Path,
    ) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def open_native(self, paths: Iterable[Path]) -> xr.Dataset:
        raise NotImplementedError

    def standardize(self, dataset: xr.Dataset) -> xr.Dataset:
        rename_map = {
            native: standard
            for native, standard in STANDARD_VARIABLES.items()
            if native in dataset and standard not in dataset
        }
        result = dataset.rename(rename_map)

        coord_renames = {}
        if "longitude" not in result.coords and "lon" in result.coords:
            coord_renames["lon"] = "longitude"
        if "latitude" not in result.coords and "lat" in result.coords:
            coord_renames["lat"] = "latitude"
        if coord_renames:
            result = result.rename(coord_renames)

        if "longitude" in result.coords:
            lon = result["longitude"]
            if float(lon.max()) > 180.0:
                result = result.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)
                result = result.sortby("longitude")

        return result

    def validate(self, dataset: xr.Dataset) -> list[str]:
        return [name for name in self.critical_variables if name not in dataset]

    def run_info(self, cycle: datetime, dataset: xr.Dataset) -> ModelRunInfo:
        missing = self.validate(dataset)
        return ModelRunInfo(
            model=self.name,
            cycle=cycle,
            available_forecast_hours=self.discover_forecast_hours(cycle),
            missing_variables=missing,
            complete=not missing,
        )
