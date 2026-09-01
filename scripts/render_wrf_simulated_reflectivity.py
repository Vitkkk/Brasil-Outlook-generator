from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from app.models.wrf_cptec import CPTECWRFAdapter


def _find_reflectivity(groups: list[xr.Dataset]) -> xr.DataArray:
    candidates = []
    for ds in groups:
        for name, da in ds.data_vars.items():
            attrs = {k: str(v).lower() for k, v in da.attrs.items()}
            text = " ".join([name.lower(), attrs.get("grib_shortname", ""), attrs.get("long_name", ""), attrs.get("standard_name", "")])
            score = 0
            if name.lower() in {"refc", "refd", "dbz", "reflectivity"}: score += 10
            if "reflectiv" in text: score += 8
            if "dbz" in text: score += 3
            if "composite" in text: score += 2
            if score:
                candidates.append((score, da))
    if not candidates:
        raise RuntimeError("No reflectivity field found in CPTEC WRF GRIB groups")
    candidates.sort(key=lambda x: x[0], reverse=True)
    da = candidates[0][1]
    # collapse any non-horizontal dimensions conservatively using max reflectivity
    for dim in tuple(da.dims):
        if dim.lower() in {"latitude", "longitude"}:
            continue
        if dim in da.coords and da.sizes.get(dim, 1) > 1:
            da = da.max(dim=dim, skipna=True)
        else:
            da = da.isel({dim: 0}, drop=True)
    return da


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--hours", default="12,15,18,21,24,27")
    args = p.parse_args()

    adapter = CPTECWRFAdapter()
    cycle = adapter.latest_cycle()
    hours = [int(x) for x in args.hours.split(",")]
    available = set(adapter.discover_forecast_hours(cycle))
    hours = [h for h in hours if h in available]
    if not hours:
        raise RuntimeError("Requested forecast hours unavailable")

    work = Path("wrf_reflectivity_raw")
    work.mkdir(exist_ok=True)
    paths = adapter.download(cycle, hours, work)

    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError("cfgrib required") from exc

    fields = []
    for h, path in zip(hours, paths):
        groups = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        da = _find_reflectivity(groups)
        # normalize coordinate names
        if "latitude" not in da.coords and "lat" in da.coords:
            da = da.rename({"lat": "latitude"})
        if "longitude" not in da.coords and "lon" in da.coords:
            da = da.rename({"lon": "longitude"})
        fields.append((h, da))

    n = len(fields)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    levels = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

    for ax, (h, da) in zip(axes, fields):
        lon = da["longitude"].values
        lat = da["latitude"].values
        z = np.asarray(da.values, dtype=float)
        if z.ndim != 2:
            z = np.squeeze(z)
        # South America focus
        mask_lon = (lon >= -70) & (lon <= -35)
        mask_lat = (lat >= -40) & (lat <= -15)
        if z.shape == (lat.size, lon.size):
            z = z[np.ix_(mask_lat, mask_lon)]
            lonp = lon[mask_lon]; latp = lat[mask_lat]
        else:
            lonp = lon; latp = lat
        cf = ax.contourf(lonp, latp, z, levels=levels, extend="max")
        ax.contour(lonp, latp, z, levels=[20,35,50], linewidths=[0.5,0.8,1.1])
        valid = cycle + __import__('datetime').timedelta(hours=h)
        ax.set_title(f"F{h:03d} • VALID {valid:%d/%m %HZ}", fontsize=12, weight="bold")
        ax.set_xlim(-70, -35); ax.set_ylim(-40, -15)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)

    for ax in axes[n:]:
        ax.axis("off")
    cbar = fig.colorbar(cf, ax=axes[:n].tolist(), orientation="horizontal", fraction=0.035, pad=0.03)
    cbar.set_label("Simulated / model reflectivity (dBZ)")
    fig.suptitle(f"CPTEC/INPE WRF 7 km — SIMULATED REFLECTIVITY • INIT {cycle:%d/%m/%Y %HZ}", fontsize=18, weight="bold")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
