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
from app.models.wrf_cptec import CPTECWRFAdapter
from app.ensemble.multimodel_v1 import multimodel_consensus

DEFAULT_HOURS = (0, 6, 12, 18, 24)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GFS + ECMWF + CPTEC WRF Day-1 guidance")
    p.add_argument("--data-dir", default="data/multimodel-wrf-v2")
    p.add_argument("--output-dir", default="output/multimodel-wrf-v2")
    p.add_argument("--hours", default=",".join(map(str, DEFAULT_HOURS)))
    return p.parse_args()


def _safe_max(da) -> float:
    values = np.asarray(da)
    return float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")


def _crop(ds):
    if "latitude" in ds.coords and ds["latitude"].ndim == 1:
        lat = ds.latitude
        ds = ds.sel(latitude=slice(15, -60) if float(lat[0]) > float(lat[-1]) else slice(-60, 15))
    if "longitude" in ds.coords and ds["longitude"].ndim == 1:
        ds = ds.sel(longitude=slice(-90, -25))
    return ds


def _run_model(adapter, cycle, hours, data_root: Path):
    samples = []
    model_dir = data_root / adapter.name.lower().replace(" ", "_").replace("/", "_") / cycle.strftime("%Y%m%d%H")
    for hour in hours:
        [path] = adapter.download(cycle, [hour], model_dir)
        native = adapter.open_native([path])
        standard = _crop(adapter.standardize(native))
        missing = adapter.validate(standard)
        if missing:
            print(f"{adapter.name} standardized missing critical variables: {missing}", flush=True)
        samples.append(model_diagnostics_v3_probabilities(standard))
    return aggregate_day1_v3(samples)


def main() -> None:
    args = parse_args()
    requested = [int(x) for x in args.hours.split(",") if x.strip()]

    adapters = [GFSAdapter(), ECMWFAdapter(), CPTECWRFAdapter()]
    latest = {a.name: a.latest_cycle() for a in adapters}
    cycle = min(latest.values())
    print("Latest model cycles:", {k: v.isoformat() for k, v in latest.items()}, flush=True)
    print("Common cycle:", cycle.isoformat(), flush=True)

    hour_sets = [set(a.discover_forecast_hours(cycle)) for a in adapters]
    hours = [h for h in requested if all(h in s for s in hour_sets)]
    if not hours:
        raise RuntimeError("No common GFS/ECMWF/WRF Day-1 forecast hours")

    data_root = Path(args.data_dir)
    output = Path(args.output_dir) / cycle.strftime("%Y%m%d%H")
    output.mkdir(parents=True, exist_ok=True)

    model_fields = {}
    for adapter in adapters:
        print(f"=== {adapter.name} ===", flush=True)
        ds = _run_model(adapter, cycle, hours, data_root)
        model_fields[adapter.name] = ds
        safe_name = adapter.name.lower().replace("-", "_").replace(" ", "_")
        ds.to_netcdf(output / f"{safe_name}_day1_v3.nc")

    consensus, summary = multimodel_consensus(model_fields)
    consensus.to_netcdf(output / "multimodel_day1_consensus_v2_wrf.nc")

    maxima = {}
    for name, ds in model_fields.items():
        maxima[name] = {k: _safe_max(ds[k]) for k in ("severe", "tornado", "hail", "wind", "supercell", "convective_initiation")}
    maxima["CONSENSUS"] = {k: _safe_max(consensus[k]) for k in ("severe", "tornado", "hail", "wind")}
    if "forecast_confidence" in consensus:
        maxima["CONSENSUS"]["confidence"] = _safe_max(consensus["forecast_confidence"])

    manifest = {
        "cycle": cycle.isoformat(),
        "valid_start": cycle.isoformat(),
        "valid_end": (cycle + timedelta(hours=24)).isoformat(),
        "sampled_forecast_hours": hours,
        "latest_cycles": {k: v.isoformat() for k, v in latest.items()},
        "models_used": summary.models,
        "method": summary.method,
        "wrf_source": "CPTEC/INPE operational WRF 7-km South America GRIB2",
        "wrf_role": "mesoscale realization member; simulated reflectivity enters V3 convective initiation when available",
        "calibrated": False,
        "maxima": maxima,
        "warning": "Engineering guidance. CPTEC WRF 7-km is not the future local 3-km WRF configuration and skill weights are not yet historically calibrated.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
