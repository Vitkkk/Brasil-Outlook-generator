from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

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

    sampled = []
    for hour in hours:
        [path] = adapter.download(cycle, [hour], data_dir)
        native = adapter.open_native([path])
        standard = adapter.standardize(native)
        sampled.append(gfs_diagnostics_v3_probabilities(standard))

    fields = aggregate_day1_v3(sampled)
    fields.to_netcdf(output_dir / "gfs_day1_diagnostics_v3.nc")

    start = cycle
    end = cycle + timedelta(hours=24)
    simplify_cfg = cfg.risk_thresholds["polygonization"]
    simplify = float(simplify_cfg["simplify_tolerance_deg"])
    min_area = float(simplify_cfg["min_area_deg2"])

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
        )
        product["properties"].update(
            {
                "model": "GFS 0.25",
                "cycle": cycle.isoformat(),
                "sampled_forecast_hours": hours,
                "calibrated": False,
                "method": "GFS Diagnostics V3",
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
    )
    categorical["properties"].update(
        {
            "model": "GFS 0.25",
            "cycle": cycle.isoformat(),
            "sampled_forecast_hours": hours,
            "calibrated": False,
            "method": "GFS Diagnostics V3",
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
        "method": "GFS Diagnostics V3",
        "calibrated": False,
        "native_cape_cin_used_for_hazards": False,
        "thermodynamic_reconstruction": {
            "vertical_levels": "1000–200 hPa; mostly 50-hPa spacing with extra near-surface levels",
            "parcels": ["surface-based", "100-hPa mixed-layer", "most-unstable in lowest 300 hPa"],
            "parcel_ascent": "dry adiabatic to Bolton LCL, then pseudoadiabatic log-pressure integration",
            "buoyancy": "virtual-temperature CAPE/CIN integrated on model geopotential-height profile",
            "effective_inflow": "CAPE >=100 J/kg and CIN >=-250 J/kg, sampled every 50 hPa through lowest 300 hPa",
        },
        "diagnostic_changes": [
            "reconstructed SBCAPE/SBCIN",
            "reconstructed MLCAPE/MLCIN",
            "reconstructed MUCAPE/MUCIN",
            "0–3 km parcel CAPE",
            "parcel LCL/LFC/EL",
            "sampled effective inflow layer",
            "native GFS CAPE/CIN removed from V3 hazard equations",
            "V2 height-resolved shear/SRH/streamwise diagnostics retained",
        ],
        "maxima": maxima,
        "warning": (
            "Diagnostics V3 is live-model engineering guidance. Thermodynamics are reconstructed from "
            "the GFS sounding, but probabilities are not yet historically calibrated."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
