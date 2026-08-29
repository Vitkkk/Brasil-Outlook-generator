from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

from app.config import get_config
from app.hazards.gfs_diagnostics_v3 import aggregate_day1_v3, gfs_diagnostics_v3_probabilities
from app.models.gfs import GFSAdapter
from app.outlook.categories import categorical_outlook
from app.outlook.categorical_geojson import categorical_field_to_geojson
from app.outlook.geojson import probability_field_to_geojson

# Reuse the peak-sounding helpers from the live V3 generator so hindcasts and
# operational runs package identical sounding metadata.
from generate_gfs_day1_v3 import _max_location, _scalar_at, _save_peak_soundings, _safe_max, _safe_min


DEFAULT_HOURS = (0, 6, 12, 18, 24)
SOUNDING_PRODUCTS = ("severe", "tornado", "hail", "wind")

# Herbie searches the GRIB inventory before downloading, so this retrieves the
# meteorological fields required by Diagnostics V3 without downloading every GFS
# field in the global pgrb2 file. The result is then cropped to South America.
HERBIE_SEARCH = r":(TMP|SPFH|RH|UGRD|VGRD|HGT|VVEL|DPT|PRES|GUST|PWAT|REFC|HLCY|USTM|VSTM):"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a historical GFS Diagnostics V3 Day-1 hindcast")
    p.add_argument("--cycle", required=True, help="Historical GFS cycle, e.g. 2025-11-07T00:00Z")
    p.add_argument("--hours", default=",".join(map(str, DEFAULT_HOURS)))
    p.add_argument("--data-dir", default="data/gfs-hindcast-v3")
    p.add_argument("--output-dir", default="output/gfs-hindcast-v3")
    return p.parse_args()


def parse_cycle(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def crop_domain(ds: xr.Dataset) -> xr.Dataset:
    cfg = get_config().domain
    out = ds
    if "longitude" in out.coords:
        out = out.sel(longitude=slice(cfg.west, cfg.east))
    if "latitude" in out.coords:
        lat = np.asarray(out.latitude, dtype=float)
        lat_slice = slice(cfg.north, cfg.south) if lat.size > 1 and lat[0] > lat[-1] else slice(cfg.south, cfg.north)
        out = out.sel(latitude=lat_slice)
    return out


def historical_grib(cycle: datetime, hour: int, destination: Path) -> Path:
    try:
        from herbie import Herbie
    except ImportError as exc:
        raise RuntimeError("Historical GFS mode requires the 'herbie-data' package") from exc

    destination.mkdir(parents=True, exist_ok=True)
    # Herbie checks the NOAA/NODD archives (AWS/Google and fallbacks) and uses
    # the GRIB index for byte-range subsetting where available.
    h = Herbie(cycle, model="gfs", product="pgrb2.0p25", fxx=int(hour), save_dir=destination)
    path = h.download(HERBIE_SEARCH, save_dir=destination, overwrite=False)
    return Path(path)


def main() -> None:
    args = parse_args()
    cfg = get_config()
    cycle = parse_cycle(args.cycle)
    hours = [int(x) for x in args.hours.split(",") if x.strip()]
    if not hours:
        raise RuntimeError("No forecast hours requested")

    stamp = cycle.strftime("%Y%m%d%H")
    data_dir = Path(args.data_dir) / stamp
    output_dir = Path(args.output_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = GFSAdapter()
    sampled: list[xr.Dataset] = []
    best_soundings: dict[str, dict] = {}
    sources: dict[str, str] = {}

    for hour in hours:
        path = historical_grib(cycle, hour, data_dir)
        sources[f"f{hour:03d}"] = str(path)
        native = adapter.open_native([path])
        standard = crop_domain(adapter.standardize(native)).load()
        diagnostics = gfs_diagnostics_v3_probabilities(standard)
        sampled.append(diagnostics)

        for product in SOUNDING_PRODUCTS:
            location = _max_location(diagnostics[product])
            if location is None:
                continue
            score, lat, lon = location
            previous = best_soundings.get(product)
            if previous is not None and score <= previous["probability"]:
                continue
            point_standard = standard.sel(latitude=lat, longitude=lon, method="nearest")
            point_diag = diagnostics.sel(latitude=lat, longitude=lon, method="nearest")
            point_dataset = xr.merge([point_standard, point_diag], compat="override", join="outer")
            params = {
                name: _scalar_at(point_diag, name)
                for name in (
                    "mlcape_jkg", "mucape_jkg", "mlcape_0_3km_jkg", "mlcin_jkg",
                    "ml_lcl_agl_m", "effective_inflow_depth_m", "shear_0_1km_ms",
                    "shear_0_6km_ms", "srh_0_1km_proxy_m2s2", "srh_0_3km_proxy_m2s2",
                    "lapse_700_500_c_per_km", "freezing_level_agl_m",
                    "supercell", "qlcs", "tornado", "hail", "wind", "severe",
                )
            }
            best_soundings[product] = {
                "probability": score,
                "forecast_hour": hour,
                "valid_time": (cycle + timedelta(hours=hour)).isoformat(),
                "latitude": lat,
                "longitude": lon,
                "parameters": params,
                "point_dataset": point_dataset,
            }

    fields = aggregate_day1_v3(sampled)
    fields.to_netcdf(output_dir / "gfs_day1_hindcast_v3.nc")
    sounding_summary = _save_peak_soundings(best_soundings, output_dir)

    start, end = cycle, cycle + timedelta(hours=24)
    polygon_cfg = cfg.risk_thresholds["polygonization"]
    simplify = float(polygon_cfg["simplify_tolerance_deg"])
    min_area = float(polygon_cfg["min_area_deg2"])
    bridge_gap = float(polygon_cfg.get("bridge_gap_deg", 0.0))
    land_only = bool(polygon_cfg.get("land_only", True))

    hazard_thresholds = {
        "thunderstorm": [0.10, 0.40, 0.70],
        "severe": [0.05, 0.15, 0.30, 0.45, 0.60],
        "tornado": cfg.risk_thresholds["hazards"]["tornado"]["contours"],
        "hail": cfg.risk_thresholds["hazards"]["hail"]["contours"],
        "wind": cfg.risk_thresholds["hazards"]["wind"]["contours"],
    }
    for name, thresholds in hazard_thresholds.items():
        product = probability_field_to_geojson(
            fields[name], thresholds, product=name, valid_start=start, valid_end=end,
            distance_km=cfg.domain.probability_radius_km,
            simplify_tolerance_deg=simplify, min_area_deg2=min_area,
            bridge_gap_deg=bridge_gap, land_only=land_only,
        )
        product["properties"].update({
            "model": "GFS historical 0.25", "cycle": cycle.isoformat(),
            "sampled_forecast_hours": hours, "calibrated": False,
            "method": "GFS Diagnostics V3.2 historical hindcast",
        })
        (output_dir / f"{name}.geojson").write_text(json.dumps(product, ensure_ascii=False), encoding="utf-8")

    categories = categorical_outlook(fields["severe"], fields["thunderstorm"], cfg.risk_thresholds)
    categorical = categorical_field_to_geojson(
        categories, valid_start=start, valid_end=end,
        simplify_tolerance_deg=simplify, min_area_deg2=min_area,
        bridge_gap_deg=bridge_gap, land_only=land_only,
    )
    categorical["properties"].update({
        "model": "GFS historical 0.25", "cycle": cycle.isoformat(),
        "sampled_forecast_hours": hours, "calibrated": False,
        "method": "GFS Diagnostics V3.2 historical hindcast",
    })
    (output_dir / "categorical.geojson").write_text(json.dumps(categorical, ensure_ascii=False), encoding="utf-8")

    maxima = {
        "severe_probability": _safe_max(fields["severe"]),
        "tornado_probability": _safe_max(fields["tornado"]),
        "hail_probability": _safe_max(fields["hail"]),
        "wind_probability": _safe_max(fields["wind"]),
        "supercell_probability": _safe_max(fields["supercell"]),
        "qlcs_probability": _safe_max(fields["qlcs"]),
        "mlcape_jkg": _safe_max(fields["mlcape_jkg"]),
        "mucape_jkg": _safe_max(fields["mucape_jkg"]),
        "mlcape_0_3km_jkg": _safe_max(fields["mlcape_0_3km_jkg"]),
        "effective_inflow_depth_m": _safe_max(fields["effective_inflow_depth_m"]),
        "shear_0_6km_ms": _safe_max(fields["shear_0_6km_ms"]),
        "abs_srh_0_1km_proxy_m2s2": _safe_max(np.abs(fields["srh_0_1km_proxy_m2s2"])),
    }
    manifest = {
        "mode": "historical_hindcast",
        "model": "GFS 0.25",
        "cycle": cycle.isoformat(),
        "valid_start": start.isoformat(),
        "valid_end": end.isoformat(),
        "sampled_forecast_hours": hours,
        "data_source": "NOAA GFS historical archive retrieved via Herbie",
        "method": "GFS Diagnostics V3.2 historical hindcast",
        "calibrated": False,
        "observations_used": False,
        "post_event_information_used": False,
        "soundings": sounding_summary,
        "maxima": maxima,
        "retrieved_files": sources,
        "warning": "Strict model-only hindcast. No radar, reports, damage data, or Prevots outlook are ingested.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
