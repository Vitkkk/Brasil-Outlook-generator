from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np

from app.hazards.gfs_diagnostics_v3 import aggregate_day1_v3
from app.hazards.model_diagnostics_v3 import model_diagnostics_v3_probabilities
from app.models.ecmwf import ECMWFAdapter
from app.models.gfs import GFSAdapter
from app.ensemble.multimodel_v1 import multimodel_consensus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PREVOTS-style 12Z-to-12Z deterministic multimodel guidance"
    )
    p.add_argument("--date", help="Target UTC date YYYY-MM-DD; defaults to current UTC date")
    p.add_argument("--data-dir", default="data/prevots-mode")
    p.add_argument("--output-dir", default="output/prevots-mode")
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
        if "latitude" in standard.coords:
            lat = standard.latitude
            standard = standard.sel(
                latitude=slice(15, -60) if float(lat[0]) > float(lat[-1]) else slice(-60, 15)
            )
        if "longitude" in standard.coords:
            standard = standard.sel(longitude=slice(-90, -25))
        samples.append(model_diagnostics_v3_probabilities(standard))
    return aggregate_day1_v3(samples)


def main() -> None:
    args = parse_args()
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(timezone.utc).date()

    valid_start = datetime(
        target_date.year, target_date.month, target_date.day, 12, tzinfo=timezone.utc
    )
    valid_end = valid_start + timedelta(hours=24)

    gfs = GFSAdapter()
    ecmwf = ECMWFAdapter()
    latest_common = min(gfs.latest_cycle(), ecmwf.latest_cycle())

    # PREVOTS-style guidance uses a fixed 12Z-to-12Z window. If a newer cycle
    # already exists (e.g. 18Z), keep the 12Z cycle so the product validity
    # remains comparable to the human-issued map. Before 12Z is available, use
    # the latest common earlier cycle and forecast forward into the same window.
    cycle = min(latest_common, valid_start)

    start_hour = int((valid_start - cycle).total_seconds() // 3600)
    end_hour = int((valid_end - cycle).total_seconds() // 3600)
    requested = list(range(start_hour, end_hour + 1, 3))

    gfs_hours = set(gfs.discover_forecast_hours(cycle))
    ec_hours = set(ecmwf.discover_forecast_hours(cycle))
    hours = [h for h in requested if h in gfs_hours and h in ec_hours]
    if not hours:
        raise RuntimeError(
            f"No common GFS/ECMWF PREVOTS-window hours for cycle={cycle.isoformat()} "
            f"target={valid_start.isoformat()}..{valid_end.isoformat()}"
        )

    data_root = Path(args.data_dir)
    output = Path(args.output_dir) / valid_start.strftime("%Y%m%d12")
    output.mkdir(parents=True, exist_ok=True)

    model_fields = {
        "GFS": _run_model(gfs, cycle, hours, data_root),
        "ECMWF": _run_model(ecmwf, cycle, hours, data_root),
    }
    model_fields["GFS"].to_netcdf(output / "gfs_prevots_day1_v3.nc")
    model_fields["ECMWF"].to_netcdf(output / "ecmwf_prevots_day1_v3.nc")

    consensus, summary = multimodel_consensus(model_fields)
    consensus.to_netcdf(output / "multimodel_prevots_consensus_v1.nc")

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
        "valid_start": valid_start.isoformat(),
        "valid_end": valid_end.isoformat(),
        "sampled_forecast_hours": hours,
        "sampling_interval_hours": 3,
        "models_used": summary.models,
        "method": summary.method,
        "product_mode": "PREVOTS_STYLE_AUTOMATIC",
        "calibrated": False,
        "maxima": maxima,
        "warning": (
            "Automatic experimental PREVOTS-style engineering guidance. "
            "Not an official PREVOTS product; thresholds are not historically calibrated."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
