from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

import numpy as np

from app.hazards.gfs_diagnostics_v3 import aggregate_day1_v3
from app.hazards.model_diagnostics_v3 import model_diagnostics_v3_probabilities
from app.models.ecmwf import ECMWFAdapter
from app.models.gfs import GFSAdapter
from app.ensemble.multimodel_v1 import multimodel_consensus


DEFAULT_HOURS = (0, 6, 12, 18, 24)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate first live deterministic multimodel Day-1 outlook")
    p.add_argument("--data-dir", default="data/multimodel-v1")
    p.add_argument("--output-dir", default="output/multimodel-v1")
    p.add_argument("--hours", default=",".join(map(str, DEFAULT_HOURS)))
    return p.parse_args()


def _safe_max(da) -> float:
    values = np.asarray(da)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def _run_model(adapter, cycle, hours, data_root: Path):
    samples = []
    model_dir = data_root / adapter.name.lower().replace(" ", "_") / cycle.strftime("%Y%m%d%H")
    for hour in hours:
        [path] = adapter.download(cycle, [hour], model_dir)
        native = adapter.open_native([path])
        standard = adapter.standardize(native)
        # Keep only the South-America calculation domain after decoding.
        if "latitude" in standard.coords:
            lat = standard.latitude
            standard = standard.sel(latitude=slice(15, -60) if float(lat[0]) > float(lat[-1]) else slice(-60, 15))
        if "longitude" in standard.coords:
            standard = standard.sel(longitude=slice(-90, -25))
        samples.append(model_diagnostics_v3_probabilities(standard))
    return aggregate_day1_v3(samples)


def main() -> None:
    args = parse_args()
    requested = [int(x) for x in args.hours.split(",") if x.strip()]

    gfs = GFSAdapter()
    ecmwf = ECMWFAdapter()
    gfs_latest = gfs.latest_cycle()
    ecmwf_latest = ecmwf.latest_cycle()

    # Use the latest cycle available from BOTH deterministic systems. Comparing
    # different initialisation times would mix forecast age with model skill.
    cycle = min(gfs_latest, ecmwf_latest)
    hours = [h for h in requested if h in set(gfs.discover_forecast_hours(cycle)) and h in set(ecmwf.discover_forecast_hours(cycle))]
    if not hours:
        raise RuntimeError("No common GFS/ECMWF Day-1 forecast hours")

    data_root = Path(args.data_dir)
    output = Path(args.output_dir) / cycle.strftime("%Y%m%d%H")
    output.mkdir(parents=True, exist_ok=True)

    model_fields = {
        "GFS": _run_model(gfs, cycle, hours, data_root),
        "ECMWF": _run_model(ecmwf, cycle, hours, data_root),
    }
    for name, ds in model_fields.items():
        ds.to_netcdf(output / f"{name.lower()}_day1_v3.nc")

    consensus, summary = multimodel_consensus(model_fields)
    consensus.to_netcdf(output / "multimodel_day1_consensus_v1.nc")

    maxima = {
        name: {
            "severe": _safe_max(ds["severe"]),
            "tornado": _safe_max(ds["tornado"]),
            "hail": _safe_max(ds["hail"]),
            "wind": _safe_max(ds["wind"]),
        }
        for name, ds in model_fields.items()
    }
    maxima["CONSENSUS"] = {
        "severe": _safe_max(consensus["severe"]),
        "tornado": _safe_max(consensus["tornado"]),
        "hail": _safe_max(consensus["hail"]),
        "wind": _safe_max(consensus["wind"]),
        "confidence": _safe_max(consensus["forecast_confidence"]),
    }

    manifest = {
        "cycle": cycle.isoformat(),
        "valid_start": cycle.isoformat(),
        "valid_end": (cycle + timedelta(hours=24)).isoformat(),
        "sampled_forecast_hours": hours,
        "models_requested": ["GFS", "ECMWF", "ICON", "WRF"],
        "models_used": summary.models,
        "models_pending": {
            "ICON": "adapter/live DWD global pressure-profile ingestion pending",
            "WRF": "local-run adapter/domain configuration pending",
        },
        "method": summary.method,
        "calibrated": False,
        "maxima": maxima,
        "warning": "Pilot multimodel engineering guidance. No historical skill weighting yet.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
