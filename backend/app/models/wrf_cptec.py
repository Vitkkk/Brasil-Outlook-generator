from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
import re

import requests
import xarray as xr

from .base_adapter import ModelAdapter
from .gfs import _prepare_cfgrib_group, _merge_cfgrib_groups


CPTEC_WRF_ROOT = "https://dataserver.cptec.inpe.br/dataserver_modelos/wrf/ams_07km/brutos"
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
_FILE_RE = re.compile(r"WRF_cpt_07KM_(\d{10})_(\d{10})\.grib2$")


def _get_text(url: str, timeout: int = 25) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _directory_hrefs(url: str) -> list[str]:
    return _HREF_RE.findall(_get_text(url))


def _cycle_url(cycle: datetime) -> str:
    return f"{CPTEC_WRF_ROOT}/{cycle:%Y/%m/%d/%H}/"


def _candidate_cycles(now: datetime | None = None) -> list[datetime]:
    now = now or datetime.now(timezone.utc)
    candidates: list[datetime] = []
    # CPTEC WRF is normally available at 00Z and, on some days, 12Z.
    for day_back in range(0, 4):
        d = (now - timedelta(days=day_back)).date()
        for hour in (12, 0):
            c = datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)
            if c <= now:
                candidates.append(c)
    return sorted(candidates, reverse=True)


class CPTECWRFAdapter(ModelAdapter):
    """Operational CPTEC/INPE WRF 7-km South-America adapter.

    CPTEC publishes hourly GRIB2 files for the South-American WRF integration.
    This adapter deliberately treats WRF as a mesoscale deterministic member,
    not as a replacement for the global models. The first implementation uses
    the public 7-km feed; a future local WRF-GFS/WRF-ECMWF runner can implement
    the same ModelAdapter interface without changing the hazard pipeline.
    """

    name = "WRF-CPTEC-7KM"

    def latest_cycle(self) -> datetime:
        for cycle in _candidate_cycles():
            try:
                hrefs = _directory_hrefs(_cycle_url(cycle))
            except requests.RequestException:
                continue
            init = cycle.strftime("%Y%m%d%H")
            if any(h.startswith(f"WRF_cpt_07KM_{init}_") and h.endswith(".grib2") for h in hrefs):
                return cycle
        raise RuntimeError("Could not discover a recent CPTEC WRF 7-km cycle")

    def discover_forecast_hours(self, cycle: datetime) -> list[int]:
        hrefs = _directory_hrefs(_cycle_url(cycle))
        result: set[int] = set()
        init = cycle.strftime("%Y%m%d%H")
        for href in hrefs:
            name = href.rsplit("/", 1)[-1]
            m = _FILE_RE.match(name)
            if not m or m.group(1) != init:
                continue
            valid = datetime.strptime(m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            hour = int((valid - cycle).total_seconds() // 3600)
            if hour >= 0:
                result.add(hour)
        return sorted(result)

    def file_url(self, cycle: datetime, forecast_hour: int) -> str:
        valid = cycle + timedelta(hours=int(forecast_hour))
        filename = f"WRF_cpt_07KM_{cycle:%Y%m%d%H}_{valid:%Y%m%d%H}.grib2"
        return _cycle_url(cycle) + filename

    def download(self, cycle: datetime, forecast_hours: Iterable[int], destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        result: list[Path] = []
        for forecast_hour in forecast_hours:
            target = destination / f"wrf_cptec_7km_{cycle:%Y%m%d%H}_f{int(forecast_hour):03d}.grib2"
            if not target.exists() or target.stat().st_size < 1024 * 1024:
                url = self.file_url(cycle, int(forecast_hour))
                print(f"Downloading CPTEC WRF: {url}", flush=True)
                with requests.get(url, stream=True, timeout=(20, 240)) as r:
                    r.raise_for_status()
                    with target.open("wb") as fh:
                        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                fh.write(chunk)
            result.append(target)
        return result

    def open_native(self, paths: Iterable[Path]) -> xr.Dataset:
        try:
            import cfgrib
        except ImportError as exc:
            raise RuntimeError("WRF GRIB ingestion requires cfgrib + eccodes") from exc

        groups: list[xr.Dataset] = []
        for path in paths:
            native_groups = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
            groups.extend(_prepare_cfgrib_group(ds) for ds in native_groups)
        if not groups:
            raise RuntimeError("No WRF GRIB datasets were decoded")
        merged = _merge_cfgrib_groups(groups)
        merged.attrs.update(model=self.name, native_grid="CPTEC_WRF_7km")
        return merged

    def standardize(self, dataset: xr.Dataset) -> xr.Dataset:
        ds = super().standardize(dataset)
        ds.attrs.update(model=self.name, native_grid="CPTEC_WRF_7km")
        return ds
