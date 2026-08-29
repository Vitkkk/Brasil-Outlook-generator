from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr
from metpy.plots import Hodograph, SkewT
from metpy.units import units


TAG_COLORS = {
    "TOR": "#d73027",
    "HAIL": "#8e44ad",
    "WIND": "#2878b5",
    "MULTI": "#b8860b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render V3 peak-risk Skew-T/hodograph soundings")
    parser.add_argument("soundings_dir")
    parser.add_argument("output_dir")
    return parser.parse_args()


def _scalar(ds: xr.Dataset, name: str, default=np.nan) -> float:
    if name not in ds:
        return float(default)
    values = np.asarray(ds[name], dtype=float)
    values = values[np.isfinite(values)]
    return float(values[0]) if values.size else float(default)


def _profile_dim(ds: xr.Dataset) -> str:
    for dim in ("isobaricInhPa", "isobaricInPa"):
        if dim in ds.dims:
            return dim
    raise KeyError("No pressure-level dimension in sounding dataset")


def _dewpoint_from_e_hpa(e_hpa: np.ndarray) -> np.ndarray:
    ln_ratio = np.log(np.maximum(e_hpa, 1e-8) / 6.112)
    tc = 243.5 * ln_ratio / np.maximum(17.67 - ln_ratio, 1e-6)
    return tc + 273.15


def _dewpoint_profile(ds: xr.Dataset, temperature_k: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    if "specific_humidity" in ds:
        q = np.asarray(ds["specific_humidity"], dtype=float).reshape(-1)
        q = np.clip(q, 1e-8, 0.10)
        e = q * pressure_hpa / np.maximum(0.622 + 0.378 * q, 1e-8)
        return _dewpoint_from_e_hpa(e)
    if "relative_humidity" in ds:
        rh = np.asarray(ds["relative_humidity"], dtype=float).reshape(-1)
        tc = temperature_k - 273.15
        es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
        return _dewpoint_from_e_hpa(es * np.clip(rh, 0.0, 100.0) / 100.0)
    return np.full_like(temperature_k, np.nan)


def _prepare_profile(ds: xr.Dataset):
    work = ds
    if "valid_time" in work.dims:
        work = work.isel(valid_time=0, drop=True)
    dim = _profile_dim(work)
    p = np.asarray(work[dim], dtype=float)
    if dim == "isobaricInPa":
        p = p / 100.0
    t = np.asarray(work["air_temperature"], dtype=float).reshape(-1)
    u = np.asarray(work["eastward_wind"], dtype=float).reshape(-1)
    v = np.asarray(work["northward_wind"], dtype=float).reshape(-1)
    z = np.asarray(work["geopotential_height"], dtype=float).reshape(-1)
    td = _dewpoint_profile(work, t, p)

    psfc = _scalar(work, "surface_air_pressure", 101325.0)
    if psfc > 2000.0:
        psfc /= 100.0
    t2 = _scalar(work, "temperature_2m")
    td2 = _scalar(work, "dewpoint_2m")
    u10 = _scalar(work, "u_wind_10m")
    v10 = _scalar(work, "v_wind_10m")
    terrain = _scalar(work, "terrain_height", 0.0)

    valid = np.isfinite(p) & np.isfinite(t) & np.isfinite(td) & np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    valid &= p <= psfc + 2.0
    p, t, td, u, v, z = p[valid], t[valid], td[valid], u[valid], v[valid], z[valid]
    order = np.argsort(p)[::-1]
    p, t, td, u, v, z = [x[order] for x in (p, t, td, u, v, z)]

    if all(np.isfinite(x) for x in (psfc, t2, td2, u10, v10)):
        p = np.concatenate(([psfc], p))
        t = np.concatenate(([t2], t))
        td = np.concatenate(([min(td2, t2)], td))
        u = np.concatenate(([u10], u))
        v = np.concatenate(([v10], v))
        z = np.concatenate(([terrain + 10.0], z))

    keep = np.concatenate(([True], np.diff(p) < -0.5))
    return p[keep], t[keep], td[keep], u[keep], v[keep], z[keep], terrain


def _fmt(value, unit="", precision=0):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{precision}f}{unit}"


def _parameter_rows(meta: dict):
    p = meta.get("parameters", {})
    return [
        ("MLCAPE", _fmt(p.get("mlcape_jkg"), " J/kg"), ["MULTI"]),
        ("MUCAPE", _fmt(p.get("mucape_jkg"), " J/kg"), ["HAIL", "MULTI"]),
        ("0–3 km MLCAPE", _fmt(p.get("mlcape_0_3km_jkg"), " J/kg"), ["TOR"]),
        ("MLCIN", _fmt(p.get("mlcin_jkg"), " J/kg"), ["MULTI"]),
        ("ML LCL", _fmt(p.get("ml_lcl_agl_m"), " m AGL"), ["TOR"]),
        ("Effective inflow depth", _fmt(p.get("effective_inflow_depth_m"), " m"), ["TOR", "HAIL"]),
        ("0–1 km shear", _fmt(p.get("shear_0_1km_ms"), " m/s", 1), ["TOR"]),
        ("0–6 km shear", _fmt(p.get("shear_0_6km_ms"), " m/s", 1), ["TOR", "HAIL"]),
        ("|SRH 0–1 km|", _fmt(abs(p.get("srh_0_1km_proxy_m2s2", np.nan)), " m²/s²"), ["TOR"]),
        ("|SRH 0–3 km|", _fmt(abs(p.get("srh_0_3km_proxy_m2s2", np.nan)), " m²/s²"), ["TOR", "HAIL"]),
        ("700–500 lapse rate", _fmt(p.get("lapse_700_500_c_per_km"), " °C/km", 1), ["HAIL"]),
        ("Freezing level", _fmt(p.get("freezing_level_agl_m"), " m AGL"), ["HAIL"]),
        ("Supercell probability", _fmt(100.0 * p.get("supercell", np.nan), "%"), ["TOR", "HAIL"]),
        ("QLCS probability", _fmt(100.0 * p.get("qlcs", np.nan), "%"), ["WIND", "TOR"]),
        ("Tornado probability", _fmt(100.0 * p.get("tornado", np.nan), "%"), ["TOR"]),
        ("Hail probability", _fmt(100.0 * p.get("hail", np.nan), "%"), ["HAIL"]),
        ("Wind probability", _fmt(100.0 * p.get("wind", np.nan), "%"), ["WIND"]),
    ]


def _draw_parameter_panel(ax, meta: dict):
    ax.axis("off")
    ax.text(0.02, 0.97, "DIAGNOSTIC PARAMETERS", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")
    ax.text(0.02, 0.925, "Color key:", transform=ax.transAxes, fontsize=9, va="top")
    x = 0.20
    for tag, label in (("TOR", "tornado"), ("HAIL", "hail"), ("WIND", "wind"), ("MULTI", "multi-hazard")):
        ax.add_patch(Rectangle((x, 0.912), 0.022, 0.022, transform=ax.transAxes,
                               facecolor=TAG_COLORS[tag], edgecolor="none"))
        ax.text(x + 0.028, 0.924, label, transform=ax.transAxes, fontsize=8, va="center")
        x += 0.19

    rows = _parameter_rows(meta)
    y = 0.875
    dy = 0.048
    for label, value, tags in rows:
        bar_x = 0.02
        bar_w = 0.010
        for tag in tags:
            ax.add_patch(Rectangle((bar_x, y - 0.013), bar_w, 0.030, transform=ax.transAxes,
                                   facecolor=TAG_COLORS[tag], edgecolor="none"))
            bar_x += bar_w + 0.004
        ax.text(0.065, y, label, transform=ax.transAxes, fontsize=9.4, va="center")
        ax.text(0.98, y, value, transform=ax.transAxes, fontsize=9.4, fontweight="bold",
                ha="right", va="center")
        y -= dy

    ax.text(0.02, 0.035,
            "Colors indicate which hazard the parameter most directly supports; they are not standalone forecast thresholds.",
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom", wrap=True)


def render_one(nc_path: Path, json_path: Path, output_path: Path):
    ds = xr.open_dataset(nc_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    p, t, td, u, v, z, terrain = _prepare_profile(ds)

    fig = plt.figure(figsize=(18, 10), dpi=150, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.1], left=0.05, right=0.97,
                          bottom=0.08, top=0.86, wspace=0.18)

    skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
    skew.plot(p * units.hPa, (t - 273.15) * units.degC, color="#d62728", linewidth=2.0, label="Temperature")
    skew.plot(p * units.hPa, (td - 273.15) * units.degC, color="#2ca02c", linewidth=2.0, label="Dewpoint")
    barb_step = max(1, len(p) // 18)
    skew.plot_barbs(p[::barb_step] * units.hPa, u[::barb_step] * units("m/s"), v[::barb_step] * units("m/s"))
    skew.ax.set_ylim(1000, 200)
    skew.ax.set_xlim(-40, 45)
    skew.plot_dry_adiabats(alpha=0.25)
    skew.plot_moist_adiabats(alpha=0.25)
    skew.plot_mixing_lines(alpha=0.20)
    skew.ax.set_title("SKEW-T / LOG-P", fontsize=14, fontweight="bold", pad=10)
    skew.ax.legend(loc="best", fontsize=8)

    ax_h = fig.add_subplot(gs[0, 1])
    component_range = max(25.0, float(np.nanmax(np.hypot(u, v))) + 5.0)
    h = Hodograph(ax_h, component_range=component_range)
    h.add_grid(increment=10)
    heights_agl = np.maximum(z - terrain, 0.0)
    h.plot_colormapped(u * units("m/s"), v * units("m/s"), heights_agl * units.m)
    ax_h.set_title("HODOGRAPH", fontsize=14, fontweight="bold", pad=10)
    ax_h.set_xlabel("u (m/s)")
    ax_h.set_ylabel("v (m/s)")

    ax_p = fig.add_subplot(gs[0, 2])
    _draw_parameter_panel(ax_p, meta)

    product = meta.get("product", "severe").upper()
    lat, lon = meta.get("latitude"), meta.get("longitude")
    valid = meta.get("valid_time", "")
    prob = 100.0 * float(meta.get("probability", 0.0))
    fig.suptitle(f"GFS V3.2 PEAK {product} SOUNDING", fontsize=22, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
             f"{lat:.2f}°, {lon:.2f}° • VALID {valid} • {product} PEAK {prob:.1f}%",
             ha="center", fontsize=11)
    fig.text(0.5, 0.895,
             "Environmental sounding used by the same GFS run that generated the Day-1 outlook",
             ha="center", fontsize=9.5, color="#555555")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)
    ds.close()


def main() -> None:
    args = parse_args()
    source = Path(args.soundings_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(source.glob("*_peak.json")):
        product = json_path.stem.replace("_peak", "")
        nc_path = source / f"{product}_peak.nc"
        if not nc_path.exists():
            continue
        target = output / f"sounding_{product}_peak.png"
        render_one(nc_path, json_path, target)
        print(target)


if __name__ == "__main__":
    main()
