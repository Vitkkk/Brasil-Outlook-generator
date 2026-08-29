from __future__ import annotations

import argparse
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
    return parser.parse_args()


def lon_lat(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(da["longitude"])
    lat = np.asarray(da["latitude"])
    return np.meshgrid(lon, lat)


def setup_axis(ax):
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
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def draw_category(ax, field: xr.DataArray, cfg):
    cat = categorical_outlook(field, field * 0 + 1.0, cfg.risk_thresholds)
    lon2d, lat2d = lon_lat(cat)
    values = np.asarray(cat)
    for code in range(1, 7):
        mask = np.where(values >= code, 1.0, np.nan)
        if np.isfinite(mask).any():
            ax.contourf(
                lon2d,
                lat2d,
                mask,
                levels=[0.5, 1.5],
                colors=[CAT_COLORS[code]],
                alpha=0.72,
                transform=ccrs.PlateCarree(),
                zorder=1 + code * 0.05,
            )
            ax.contour(
                lon2d,
                lat2d,
                values,
                levels=[code - 0.5],
                colors=[CAT_COLORS[code]],
                linewidths=1.25,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
    handles = [
        Patch(facecolor=CAT_COLORS[c], edgecolor="#555555", label=CAT_NAMES[c])
        for c in range(1, 7)
        if np.any(values == c)
    ]
    ax.legend(handles=handles, loc="lower right", ncol=3, fontsize=7, title="Risk Category", title_fontsize=8)


def draw_hazard(ax, field: xr.DataArray, thresholds: list[float]):
    lon2d, lat2d = lon_lat(field)
    values = np.asarray(field)
    reached = [t for t in thresholds if np.nanmax(values) >= t]
    for idx, thr in enumerate(reached):
        ax.contourf(
            lon2d,
            lat2d,
            values,
            levels=[thr, 1.0],
            colors=[HAZARD_COLORS[thr]],
            alpha=0.66,
            transform=ccrs.PlateCarree(),
            zorder=1 + idx * 0.05,
        )
        ax.contour(
            lon2d,
            lat2d,
            values,
            levels=[thr],
            colors=[HAZARD_COLORS[thr]],
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    handles = [
        Patch(facecolor=HAZARD_COLORS[t], edgecolor="#555555", label=f"{int(t*100)}%")
        for t in reached
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            fontsize=7,
            title="Probability within 40 km",
            title_fontsize=8,
        )


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ds = xr.open_dataset(args.netcdf)

    fig = plt.figure(figsize=(18, 12), dpi=160)
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.985, bottom=0.075, top=0.84, wspace=0.05, hspace=0.08)
    projection = ccrs.PlateCarree()
    products = [
        ("CATEGORICAL CONVECTIVE OUTLOOK", "severe", None),
        ("TORNADO PROBABILITY", "tornado", [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]),
        ("HAIL PROBABILITY", "hail", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
        ("WIND PROBABILITY", "wind", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
    ]

    for i, (title, name, thresholds) in enumerate(products):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection=projection)
        setup_axis(ax)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
        if thresholds is None:
            draw_category(ax, ds[name], cfg)
        else:
            draw_hazard(ax, ds[name], thresholds)

    fig.suptitle("BRAZIL SEVERE WEATHER OUTLOOK — GFS PROXY V1", fontsize=22, fontweight="bold", y=0.96)
    fig.text(0.5, 0.92, f"GFS 0.25° • CYCLE {args.cycle} • VALID {args.valid}", ha="center", fontsize=11)
    fig.text(
        0.5,
        0.888,
        "FIRST LIVE-GFS SMOKE TEST • PROXY PROBABILITIES NOT YET CALIBRATED",
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#fff0b8", edgecolor="#987a14"),
    )
    fig.text(
        0.04,
        0.035,
        "Source: NOAA/NCEP GFS via NOMADS subset ingestion. Proxy v1 uses native CAPE/CIN, pressure-level wind/temperature/height, helicity, PWAT, reflectivity and gust features.",
        fontsize=8.2,
        color="#50545a",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
