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
    return parser.parse_args()


def lon_lat(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(da["longitude"])
    lat = np.asarray(da["latitude"])
    return np.meshgrid(lon, lat)


def setup_axis(ax, *, left_labels: bool, bottom_labels: bool):
    ax.set_extent([-68, -34, -42, -14], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f5f6f7", zorder=0)
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.65,
        edgecolor="#555555",
        zorder=5,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.65,
        edgecolor="#666666",
        zorder=5,
    )
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
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=3,
            fontsize=7,
            title="Risk Category",
            title_fontsize=8,
            framealpha=0.95,
        )


def draw_hazard(ax, field: xr.DataArray, thresholds: list[float]):
    lon2d, lat2d = lon_lat(field)
    values = np.asarray(field)
    reached = [t for t in thresholds if np.isfinite(values).any() and np.nanmax(values) >= t]

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
        Patch(facecolor=HAZARD_COLORS[t], edgecolor="#555555", label=f"{int(t * 100)}%")
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
            framealpha=0.95,
        )


def _manifest_summary(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        maxima = data.get("maxima", {})
        shear = maxima.get("shear_0_6km_ms")
        srh = maxima.get("abs_srh_0_1km_proxy_m2s2")
        if shear is None or srh is None:
            return ""
        return f"DAY-1 DIAGNOSTIC MAXIMA: 0–6 km shear {shear:.1f} m/s • |SRH 0–1 km proxy| {srh:.0f} m²/s²"
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ds = xr.open_dataset(args.netcdf)

    # Square canvas + fixed margins prevents the right-hand panels from being
    # clipped in mobile previews and downstream image viewers.
    fig = plt.figure(figsize=(14, 14), dpi=155, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.955,
        bottom=0.095,
        top=0.815,
        wspace=0.10,
        hspace=0.12,
    )

    projection = ccrs.PlateCarree()
    products = [
        ("CATEGORICAL CONVECTIVE OUTLOOK", "categorical", None),
        ("TORNADO PROBABILITY", "tornado", [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]),
        ("HAIL PROBABILITY", "hail", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
        ("WIND PROBABILITY", "wind", [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]),
    ]

    for i, (title, name, thresholds) in enumerate(products):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=projection)
        setup_axis(ax, left_labels=(col == 0), bottom_labels=(row == 1))
        ax.set_title(title, fontsize=11.2, fontweight="bold", pad=6)

        if name == "categorical":
            draw_category(ax, ds["severe"], ds["thunderstorm"], cfg)
        else:
            draw_hazard(ax, ds[name], thresholds)

    fig.suptitle(
        "BRAZIL SEVERE WEATHER OUTLOOK — GFS DIAGNOSTICS V2",
        fontsize=21.5,
        fontweight="bold",
        y=0.955,
    )
    fig.text(
        0.5,
        0.920,
        f"GFS 0.25° • CYCLE {args.cycle} • VALID {args.valid}",
        ha="center",
        fontsize=10.8,
    )
    fig.text(
        0.5,
        0.887,
        "LIVE GFS • HEIGHT-RESOLVED KINEMATICS • 3-HOURLY DAY-1 SAMPLING • NOT YET CALIBRATED",
        ha="center",
        fontsize=9.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#fff0b8", edgecolor="#987a14"),
    )

    summary = _manifest_summary(args.manifest)
    if summary:
        fig.text(0.5, 0.852, summary, ha="center", fontsize=8.7, color="#444444")

    fig.text(
        0.055,
        0.040,
        "Source: NOAA/NCEP GFS via NOMADS. V2 interpolates wind/height profiles to AGL layers and adds SRH, LCL, lapse-rate, freezing-level, storm-relative-flow, supercell and QLCS diagnostics. Probabilities remain engineering guidance pending historical calibration.",
        fontsize=7.8,
        color="#50545a",
        wrap=True,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Intentionally no bbox_inches='tight': Cartopy label artists can make tight
    # bounding boxes extend beyond the intended canvas and clip the right column.
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
