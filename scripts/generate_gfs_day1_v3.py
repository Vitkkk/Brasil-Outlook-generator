from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import xarray as xr

from app.config import get_config
from app.hazards.gfs_diagnostics_v3 import (
    aggregate_day1_v3,
    gfs_diagnostics_v3_probabilities,
)
from app.models.gfs import GFSAdapter
from app.outlook.categories import categorical_outlook
from app.outlook.categorical_geojson import categorical_field_to_geojson
from app.outlook.geojson import probability_field_to_geojson


DEFAULT_HOURS = tuple(range(0, 25, 3))
SOUNDING_PRODUCTS = ("severe", "tornado", "hail", "wind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate live GFS Diagnostics V3 Day-1 outlook")
    parser.add_argument("--data-dir", default="data/gfs-v3", help="GRIB cache directory")
    parser.add_argument("--output-dir", default="output/gfs-v3", help="Generated output directory")
    parser.add_argument(
        "--hours",
        default=",".join(str(x) for x in DEFAULT_HOURS),
        help="Comma-separated GFS forecast hours sampled for Day 1",
    )
    return parser.parse_args()


def _safe_max(field) -> float:
    values = np.asarray(field)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def _safe_min(field) -> float:
    values = np.asarray(field)
    return float(np.nanmin(values)) if np.isfinite(values).any() else float("nan")


def _point_field(field: xr.DataArray) -> xr.DataArray:
    work = field
    if "valid_time" in work.dims and work.sizes["valid_time"] == 1:
        work = work.isel(valid_time=0, drop=True)
    return work


def _max_location(field: xr.DataArray) -> tuple[float, float, float] | None:
    work = _point_field(field).transpose("latitude", "longitude")
    values = np.asarray(work, dtype=float)
    if not np.isfinite(values).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(values), values.shape)
    return (
        float(values[iy, ix]),
        float(work.latitude.values[iy]),
        float(work.longitude.values[ix]),
    )


def _scalar_at(point: xr.Dataset, name: str) -> float | None:
    if name not in point:
        return None
    values = np.asarray(point[name], dtype=float)
    if not np.isfinite(values).any():
        return None
    return float(values.reshape(-1)[0])


def _save_peak_soundings(
    best: dict[str, dict],
    output_dir: Path,
) -> list[dict]:
    sounding_dir = output_dir / "soundings"
    sounding_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for product, entry in best.items():
        point = entry.pop("point_dataset")
        point_path = sounding_dir / f"{product}_peak.nc"
        point.to_netcdf(point_path)
        metadata = dict(entry)
        metadata["product"] = product
        metadata["profile_file"] = point_path.name
        metadata_path = sounding_dir / f"{product}_peak.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(metadata)
    return summary


def main() -> None:
    args = parse_args()
    adapter = GFSAdapter()
    cfg = get_config()
    cycle = adapter.latest_cycle()
    available = set(adapter.discover_forecast_hours(cycle))
    requested = [int(value) for value in args.hours.split(",") if value.strip()]
    hours = [hour for hour in requested if hour in available]
    if not hours:
        raise RuntimeError("None of the requested Day-1 hours are available")

    data_dir = Path(args.data_dir) / cycle.strftime("%Y%m%d%H")
    output_dir = Path(args.output_dir) / cycle.strftime("%Y%m%d%H")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled: list[xr.Dataset] = []
    best_soundings: dict[str, dict] = {}

    for hour in hours:
        [path] = adapter.download(cycle, [hour], data_dir)
        native = adapter.open_native([path])
        standard = adapter.standardize(native)
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
                "forecast_hour": int(hour),
                "valid_time": (cycle + timedelta(hours=int(hour))).isoformat(),
                "latitude": lat,
                "longitude": lon,
                "parameters": params,
                "point_dataset": point_dataset,
            }

    fields = aggregate_day1_v3(sampled)
    fields.to_netcdf(output_dir / "gfs_day1_diagnostics_v3.nc")
    sounding_summary = _save_peak_soundings(best_soundings, output_dir)

    start = cycle
    end = cycle + timedelta(hours=24)
    polygon_cfg = cfg.risk_thresholds["polygonization"]
    simplify = float(polygon_cfg["simplify_tolerance_deg"])
    min_area = float(polygon_cfg["min_area_deg2"])
    bridge_gap = float(polygon_cfg.get("bridge_gap_deg", 0.0))
    land_only = bool(polygon_cfg.get("land_only", False))

    hazard_thresholds = {
        "thunderstorm": [0.10, 0.40, 0.70],
        "severe": [0.05, 0.15, 0.30, 0.45, 0.60],
        "tornado": cfg.risk_thresholds["hazards"]["tornado"]["contours"],
        "hail": cfg.risk_thresholds["hazards"]["hail"]["contours"],
        "wind": cfg.risk_thresholds["hazards"]["wind"]["contours"],
        "convective_initiation": [0.10, 0.30, 0.50, 0.70],
        "supercell": [0.10, 0.30, 0.50, 0.70],
        "qlcs": [0.10, 0.30, 0.50, 0.70],
        "sig_tornado_support": [0.30, 0.50, 0.70],
    }

    for name, thresholds in hazard_thresholds.items():
        product = probability_field_to_geojson(
            fields[name],
            thresholds,
            product=name,
            valid_start=start,
            valid_end=end,
            distance_km=cfg.domain.probability_radius_km,
            simplify_tolerance_deg=simplify,
            min_area_deg2=min_area,
            bridge_gap_deg=bridge_gap,
            land_only=land_only,
        )
        product["properties"].update(
            {
                "model": "GFS 0.25",
                "cycle": cycle.isoformat(),
                "sampled_forecast_hours": hours,
                "calibrated": False,
                "method": "GFS Diagnostics V3.2 presentation",
                "native_cape_cin_used_for_hazards": False,
            }
        )
        with (output_dir / f"{name}.geojson").open("w", encoding="utf-8") as handle:
            json.dump(product, handle, ensure_ascii=False)

    categories = categorical_outlook(fields["severe"], fields["thunderstorm"], cfg.risk_thresholds)
    categorical = categorical_field_to_geojson(
        categories,
        valid_start=start,
        valid_end=end,
        simplify_tolerance_deg=simplify,
        min_area_deg2=min_area,
        bridge_gap_deg=bridge_gap,
        land_only=land_only,
    )
    categorical["properties"].update(
        {
            "model": "GFS 0.25",
            "cycle": cycle.isoformat(),
            "sampled_forecast_hours": hours,
            "calibrated": False,
            "method": "GFS Diagnostics V3.2 presentation",
            "native_cape_cin_used_for_hazards": False,
        }
    )
    with (output_dir / "categorical.geojson").open("w", encoding="utf-8") as handle:
        json.dump(categorical, handle, ensure_ascii=False)

    maxima = {
        "severe_probability": _safe_max(fields["severe"]),
        "tornado_probability": _safe_max(fields["tornado"]),
        "hail_probability": _safe_max(fields["hail"]),
        "wind_probability": _safe_max(fields["wind"]),
        "supercell_probability": _safe_max(fields["supercell"]),
        "qlcs_probability": _safe_max(fields["qlcs"]),
        "sbcape_jkg": _safe_max(fields["sbcape_jkg"]),
        "mlcape_jkg": _safe_max(fields["mlcape_jkg"]),
        "mucape_jkg": _safe_max(fields["mucape_jkg"]),
        "mlcape_0_3km_jkg": _safe_max(fields["mlcape_0_3km_jkg"]),
        "least_negative_mlcin_jkg": _safe_max(fields["mlcin_jkg"]),
        "lowest_ml_lcl_agl_m": _safe_min(fields["ml_lcl_agl_m"]),
        "effective_inflow_depth_m": _safe_max(fields["effective_inflow_depth_m"]),
        "shear_0_6km_ms": _safe_max(fields["shear_0_6km_ms"]),
        "abs_srh_0_1km_proxy_m2s2": _safe_max(np.abs(fields["srh_0_1km_proxy_m2s2"])),
    }

    manifest = {
        "model": "GFS 0.25",
        "cycle": cycle.isoformat(),
        "valid_start": start.isoformat(),
        "valid_end": end.isoformat(),
        "sampled_forecast_hours": hours,
        "data_source": "NOAA/NCEP NOMADS",
        "method": "GFS Diagnostics V3.2",
        "calibrated": False,
        "native_cape_cin_used_for_hazards": False,
        "presentation": {
            "land_only": land_only,
            "bridge_gap_deg": bridge_gap,
            "min_area_deg2": min_area,
            "simplify_tolerance_deg": simplify,
            "raw_probability_grid_unchanged": True,
        },
        "soundings": sounding_summary,
        "thermodynamic_reconstruction": {
            "vertical_levels": "1000–200 hPa; mostly 50-hPa spacing with extra near-surface levels",
            "parcels": ["surface-based", "100-hPa mixed-layer", "most-unstable in lowest 300 hPa"],
            "parcel_ascent": "dry adiabatic to Bolton LCL, then pseudoadiabatic log-pressure integration",
            "buoyancy": "virtual-temperature CAPE/CIN integrated on model geopotential-height profile",
            "effective_inflow": "CAPE >=100 J/kg and CIN >=-250 J/kg, sampled every 50 hPa through lowest 300 hPa",
        },
        "maxima": maxima,
        "warning": (
            "Diagnostics V3.2 is live-model engineering guidance. Thermodynamics are reconstructed from "
            "the GFS sounding, but probabilities are not yet historically calibrated."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
