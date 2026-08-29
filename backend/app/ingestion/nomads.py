from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
import time
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOMADS_GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
NOMADS_GFS_PROD = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"


@dataclass(slots=True, frozen=True)
class DomainBox:
    north: float
    south: float
    west: float
    east: float

    def as_nomads(self) -> dict[str, str]:
        # NOMADS accepts 0..360 or -180..180. Use 0..360 to avoid ambiguity.
        def lon360(value: float) -> float:
            return value % 360.0

        return {
            "toplat": f"{self.north:g}",
            "bottomlat": f"{self.south:g}",
            "leftlon": f"{lon360(self.west):g}",
            "rightlon": f"{lon360(self.east):g}",
        }


def build_gfs_filter_url(
    cycle: datetime,
    forecast_hour: int,
    *,
    variables: Iterable[str],
    levels: Iterable[str],
    domain: DomainBox,
) -> str:
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    cycle = cycle.astimezone(timezone.utc)
    date = cycle.strftime("%Y%m%d")
    hour = cycle.strftime("%H")
    params: list[tuple[str, str]] = [
        ("file", f"gfs.t{hour}z.pgrb2.0p25.f{forecast_hour:03d}"),
    ]
    params.extend((f"var_{name}", "on") for name in variables)
    params.extend((f"lev_{name}", "on") for name in levels)
    params.extend(domain.as_nomads().items())
    params.append(("subregion", ""))
    params.append(("dir", f"/gfs.{date}/{hour}/atmos"))
    return f"{NOMADS_GFS_FILTER}?{urlencode(params)}"


def _read_text(url: str, *, timeout: int = 20) -> str:
    req = Request(url, headers={"User-Agent": "BrazilSevereWeatherOutlook/0.2"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def gfs_cycle_directory(cycle: datetime) -> str:
    cycle = cycle.astimezone(timezone.utc)
    return (
        f"{NOMADS_GFS_PROD}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
    )


def cycle_available(cycle: datetime, *, timeout: int = 15) -> bool:
    try:
        text = _read_text(gfs_cycle_directory(cycle), timeout=timeout)
    except Exception:
        return False
    return f"gfs.t{cycle:%H}z.pgrb2.0p25.f000" in text


def discover_latest_gfs_cycle(now: datetime | None = None) -> datetime:
    """Return newest available 00/06/12/18Z GFS cycle on NOMADS.

    The function checks recent cycle directories rather than assuming publication
    delay, which keeps the service safe during partial/late runs.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    base_hour = (current.hour // 6) * 6
    candidate = current.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    for offset in range(0, 36, 6):
        cycle = candidate - timedelta(hours=offset)
        if cycle_available(cycle):
            return cycle
    raise RuntimeError("No recent GFS cycle could be discovered on NOMADS")


def discover_gfs_forecast_hours(cycle: datetime) -> list[int]:
    text = _read_text(gfs_cycle_directory(cycle))
    pattern = rf"gfs\.t{cycle:%H}z\.pgrb2\.0p25\.f(\d{{3}})"
    hours = sorted({int(value) for value in re.findall(pattern, text)})
    return hours


def download_url(
    url: str,
    destination: Path,
    *,
    attempts: int = 4,
    timeout: int = 120,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": "BrazilSevereWeatherOutlook/0.2"})
            with urlopen(req, timeout=timeout) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size < 16:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")
            partial.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error
