from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

from app.config import get_config
from render_gfs_day1_v3 import CAT_COLORS, CAT_NAMES, HAZARD_COLORS, draw_category, draw_hazard, setup_axis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render historical GFS V3 Day-1 hindcast")
    p.add_argument("netcdf")
    p.add_argument("output")
    p.add_argument("--manifest", required=True)
    return p.parse_args()


def panel_label(fig, ax, title: str) -> None:
    box = ax.get_position()
    fig.text((box.x0 + box.x1) / 2, box.y1 + 0.010, title,
             ha="center", va="bottom", fontsize=11.4, fontweight="bold")


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ds = xr.open_dataset(args.netcdf)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

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
        if name == "categorical":
            draw_category(ax, ds["severe"], ds["thunderstorm"], cfg)
        else:
            draw_hazard(ax, ds[name], thresholds, cfg)
        panel_label(fig, ax, title)

    maxima = manifest.get("maxima", {})
    summary = (
        f"DAY-1 MAXIMA: severe {100*maxima.get('severe_probability', np.nan):.1f}% • "
        f"tornado {100*maxima.get('tornado_probability', np.nan):.1f}% • "
        f"hail {100*maxima.get('hail_probability', np.nan):.1f}% • "
        f"wind {100*maxima.get('wind_probability', np.nan):.1f}%"
    )
    cycle = manifest.get("cycle", "")
    valid_start = manifest.get("valid_start", "")
    valid_end = manifest.get("valid_end", "")

    fig.suptitle("BRAZIL SEVERE WEATHER OUTLOOK — GFS HISTORICAL HINDCAST V3.2",
                 fontsize=20.5, fontweight="bold", y=0.955)
    fig.text(0.5, 0.920,
             f"GFS 0.25° • CYCLE {cycle} • VALID {valid_start} – {valid_end}",
             ha="center", fontsize=10.0)
    fig.text(0.5, 0.887,
             "STRICT MODEL-ONLY HINDCAST • NO RADAR / REPORTS / DAMAGE / PREVOTS INPUT • NOT YET CALIBRATED",
             ha="center", fontsize=8.8, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.34", facecolor="#fff0b8", edgecolor="#987a14"))
    fig.text(0.5, 0.852, summary, ha="center", fontsize=8.6, color="#444444")
    fig.text(0.055, 0.040,
             "Source: historical NOAA/NCEP GFS archive retrieved via Herbie. Diagnostics use reconstructed SB/ML/MU parcel thermodynamics and height-resolved kinematics. This is a retrospective model-only test generated without ingesting observations from the event.",
             fontsize=7.6, color="#50545a", wrap=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    ds.close()
    print(out)


if __name__ == "__main__":
    main()
