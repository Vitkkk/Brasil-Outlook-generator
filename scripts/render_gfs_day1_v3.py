from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from app.config import get_config
from app.outlook.categories import categorical_outlook


CAT_COLORS = {
    1: "#cdeec8",
    2: "#58b957",
    3: "#f5e84a",
    4: "#f3a43a",
    5: "#e85b55",
    6: "#d94fd5",
}
CAT_NAMES = {1: "TSTM", 2: "MRGL", 3: "SLGT", 4: "ENH", 5: "MDT", 6: "HIGH"}
HAZARD_COLORS = {
    0.02: "#58b957",
    0.05: "#9f6b4a",
    0.10: "#f0cf32",
    0.15: "#f0a333",
    0.30: "#e65345",
    0.45: "#8853d1",
    0.60: "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("netcdf")
    parser.add_argument("output")
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--thermo-output")
    return parser.parse_args()


def lon_lat(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(da["longitude"])
    lat = np.asarray(da["latitude"])
    return np.meshgrid(lon, lat)


def setup_axis(ax, *, left_labels: bool, bottom_labels: bool):
    ax.set_extent([-68, -34, -42, -14], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f5f6f7", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.65, edgecolor="#555555", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.65, edgecolor="#666666", zorder=5)
    try:
        states = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )
        ax.add_feature(states, linewidth=0.40, edgecolor="#808080", zorder=5)
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#c8ccd2", alpha=0.65)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def draw_category(ax, severe: xr.DataArray, thunderstorm: xr.DataArray, cfg):
    cat = categorical_outlook(severe, thunderstorm, cfg.risk_thresholds)
    lon2d, lat2d = lon_lat(cat)
    values = np.asarray(cat)
    for code in range(1, 7):
        mask = np.where(values >= code, 1.0, np.nan)
        if np.isfinite(mask).any():
            ax.contourf(
                lon2d, lat2d, mask,
                levels=[0.5, 1.5], colors=[CAT_COLORS[code]], alpha=0.72,
                transform=ccrs.PlateCarree(), zorder=1 + code * 0.05,
            )
            ax.contour(
                lon2d, lat2d, values,
                levels=[code - 0.5], colors=[CAT_COLORS[code]], linewidths=1.25,
                transform=ccrs.PlateCarree(), zorder=4,
            )
    handles = [
        Patch(facecolor=CAT_COLORS[c], edgecolor="#555555", label=CAT_NAMES[c])
        for c in range(1, 7) if np.any(values == c)
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", ncol=3, fontsize=7,
                  title="Risk Category", title_fontsize=8, framealpha=0.95)


def draw_hazard(ax, field: xr.DataArray, thresholds: list[float]):
    lon2d, lat2d = lon_lat(field)
    values = np.asarray(field)
    reached = [t for t in thresholds if np.isfinite(values).any() and np.nanmax(values) >= t]
    for idx, thr in enumerate(reached):
        ax.contourf(
            lon2d, lat2d, values,
            levels=[thr, 1.0], colors=[HAZARD_COLORS[thr]], alpha=0.66,
            transform=ccrs.PlateCarree(), zorder=1 + idx * 0.05,
        )
        ax.contour(
            lon2d, lat2d, values,
            levels=[thr], colors=[HAZARD_COLORS[thr]], linewidths=1.2,
            transform=ccrs.PlateCarree(), zorder=4,
        )
    handles = [Patch(facecolor=HAZARD_COLORS[t], edgecolor="#555555", label=f"{int(t * 100)}%") for t in reached]
    if handles:
        ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=7,
                  title="Probability within 40 km", title_fontsize=8, framealpha=0.95)


def _manifest(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary(data: dict) -> str:
    maxima = data.get("maxima", {})
    mlcape = maxima.get("mlcape_jkg")
    mucape = maxima.get("mucape_jkg")
    lowcape = maxima.get("mlcape_0_3km_jkg")
    eff = maxima.get("effective_inflow_depth_m")
    if any(value is None for value in (mlcape, mucape, lowcape, eff)):
        return ""
    return (
        f"DAY-1 THERMODYNAMIC MAXIMA: MLCAPE {mlcape:.0f} J/kg • MUCAPE {mucape:.0f} J/kg • "
        f"0–3 km MLCAPE {lowcape:.0f} J/kg • effective inflow depth {eff:.0f} m"
    )


def render_outlook(ds: xr.Dataset, args: argparse.Namespace, cfg, manifest: dict) -> None:
    fig = plt.figure(figsize=(14, 14), dpi=155, facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.955, bottom=0.095, top=0.815, wspace=0.10, hspace=0.12)
    projection = ccrs.PlateCarree()
    products = [
        ("CATEGORICAL CONVECTIVE OUTLOOK", "categorical", None),
        ("TORNADO PROBABILITY", "tornado", [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]),
        ("HAIL PROBABILITY", "hail", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
        ("WIND PROBABILITY", "wind", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
    ]
    for i, (title, name, thresholds) in enumerate(products):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col], projection=projection)
        setup_axis(ax, left_labels=(col == 0), bottom_labels=(row == 1))
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=6)
        if name == "categorical":
            draw_category(ax, ds["severe"], ds["thunderstorm"], cfg)
        else:
            draw_hazard(ax, ds[name], thresholds)

    fig.suptitle("BRAZIL SEVERE WEATHER OUTLOOK — GFS DIAGNOSTICS V3", fontsize=21.5, fontweight="bold", y=0.955)
    fig.text(0.5, 0.920, f"GFS 0.25° • CYCLE {args.cycle} • VALID {args.valid}", ha="center", fontsize=10.8)
    fig.text(
        0.5, 0.887,
        "LIVE GFS • RECONSTRUCTED SB/ML/MU PARCELS • HEIGHT-RESOLVED KINEMATICS • NOT YET CALIBRATED",
        ha="center", fontsize=9.0, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#fff0b8", edgecolor="#987a14"),
    )
    summary = _summary(manifest)
    if summary:
        fig.text(0.5, 0.852, summary, ha="center", fontsize=8.5, color="#444444")
    fig.text(
        0.055, 0.040,
        "Source: NOAA/NCEP GFS via NOMADS. V3 reconstructs surface-based, 100-hPa mixed-layer and most-unstable parcel thermodynamics from pressure-level T/q/height profiles; native GFS CAPE/CIN are not used by the V3 hazard equations. Probabilities remain engineering guidance pending historical calibration.",
        fontsize=7.6, color="#50545a", wrap=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def render_thermodynamics(ds: xr.Dataset, output_path: str, args: argparse.Namespace) -> None:
    fig = plt.figure(figsize=(14, 14), dpi=155, facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.955, bottom=0.095, top=0.84, wspace=0.10, hspace=0.12)
    projection = ccrs.PlateCarree()
    products = [
        ("MAX MIXED-LAYER CAPE (J/kg)", "mlcape_jkg", [250, 500, 1000, 1500, 2000, 3000, 4000]),
        ("MAX MOST-UNSTABLE CAPE (J/kg)", "mucape_jkg", [250, 500, 1000, 1500, 2000, 3000, 4000]),
        ("MAX 0–3 km MLCAPE (J/kg)", "mlcape_0_3km_jkg", [25, 50, 100, 150, 250, 400]),
        ("MAX EFFECTIVE INFLOW DEPTH (m)", "effective_inflow_depth_m", [100, 300, 600, 900, 1200, 1600]),
    ]
    for i, (title, name, levels) in enumerate(products):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col], projection=projection)
        setup_axis(ax, left_labels=(col == 0), bottom_labels=(row == 1))
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=6)
        field = ds[name]
        lon2d, lat2d = lon_lat(field)
        values = np.asarray(field)
        finite = np.isfinite(values)
        if finite.any():
            vmax = max(float(np.nanmax(values)), levels[1])
            usable = [level for level in levels if level < vmax]
            if len(usable) < 2:
                usable = [0.0, vmax]
            else:
                usable = [0.0] + usable + [max(vmax, usable[-1] + 1.0)]
            cf = ax.contourf(lon2d, lat2d, values, levels=usable, transform=ccrs.PlateCarree(), extend="max")
            cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.045, shrink=0.88)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle("BRAZIL GFS THERMODYNAMICS — DIAGNOSTICS V3", fontsize=21.5, fontweight="bold", y=0.955)
    fig.text(0.5, 0.920, f"GFS 0.25° • CYCLE {args.cycle} • VALID {args.valid}", ha="center", fontsize=10.8)
    fig.text(
        0.5, 0.887,
        "PROFILE-RECONSTRUCTED PARCELS • DAY-1 EXTREMA • NOT NATIVE GFS CAPE/CIN",
        ha="center", fontsize=9.0, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#e9f4ff", edgecolor="#4d7697"),
    )
    fig.text(
        0.055, 0.040,
        "Thermodynamics are calculated from pressure-level temperature, specific humidity, geopotential height, 2-m T/Td and surface pressure. Effective inflow uses CAPE ≥100 J/kg and CIN ≥−250 J/kg sampled every 50 hPa through the lowest 300 hPa.",
        fontsize=7.6, color="#50545a", wrap=True,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ds = xr.open_dataset(args.netcdf)
    manifest = _manifest(args.manifest)
    render_outlook(ds, args, cfg, manifest)
    if args.thermo_output:
        render_thermodynamics(ds, args.thermo_output, args)
    print(args.output)


if __name__ == "__main__":
    main()
