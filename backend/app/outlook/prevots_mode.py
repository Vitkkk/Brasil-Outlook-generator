from __future__ import annotations

import numpy as np
import xarray as xr


def _pair_median(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    a, b = xr.align(a, b, join="inner")
    return xr.concat([a, b], dim="model_pair").median("model_pair", skipna=True)


def _pair_max(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    a, b = xr.align(a, b, join="inner")
    return xr.apply_ufunc(np.maximum, a, b)


def prevots_severity_levels(
    consensus: xr.Dataset,
    gfs: xr.Dataset,
    ecmwf: xr.Dataset,
) -> xr.DataArray:
    """PREVOTS-like automatic severity scale (0..5).

    This is a dedicated South-America presentation heuristic, not a remapping of
    SPC categories. It uses the multimodel severe guidance together with
    instability, deep/low-level shear, helicity proxies and expected storm mode.

    Codes:
      0 none
      1 thunderstorms
      2 Level 1
      3 Level 2
      4 Level 3
      5 Level 4

    The thresholds are intentionally conservative for Levels 1-4 so broad
    marginal severe probabilities do not paint most of Brazil yellow. They are
    still engineering thresholds and require historical calibration.
    """
    consensus, gfs, ecmwf = xr.align(consensus, gfs, ecmwf, join="inner")

    thunder = consensus["thunderstorm"]
    severe = consensus["severe"]
    tornado = consensus["tornado"]
    hail = consensus["hail"]
    wind = consensus["wind"]

    cape_med = _pair_median(gfs["mucape_jkg"], ecmwf["mucape_jkg"])
    cape_max = _pair_max(gfs["mucape_jkg"], ecmwf["mucape_jkg"])
    shear06_med = _pair_median(gfs["shear_0_6km_ms"], ecmwf["shear_0_6km_ms"])
    shear01_med = _pair_median(gfs["shear_0_1km_ms"], ecmwf["shear_0_1km_ms"])
    srh03_max = _pair_max(
        np.abs(gfs["srh_0_3km_proxy_m2s2"]),
        np.abs(ecmwf["srh_0_3km_proxy_m2s2"]),
    )
    supercell_med = _pair_median(gfs["supercell"], ecmwf["supercell"])
    qlcs_med = _pair_median(gfs["qlcs"], ecmwf["qlcs"])

    # Broad thunder area, similar to the green "Tempestades" envelope.
    tstm = thunder >= 0.18

    # Level 1 requires a meaningful severe signal OR a physically supportive
    # CAPE/deep-shear overlap. This deliberately avoids mapping a 5% SPC-like
    # MRGL contour directly to Level 1.
    level1 = tstm & (
        (severe >= 0.085)
        | ((cape_med >= 550.0) & (shear06_med >= 15.0))
        | ((cape_max >= 200.0) & (shear06_med >= 15.0) & (thunder >= 0.35))
    )

    # Level 2: organized severe convection is plausible over a coherent area.
    level2 = tstm & (
        (severe >= 0.15)
        | (
            (cape_med >= 1200.0)
            & (shear06_med >= 18.0)
            & ((supercell_med >= 0.07) | (qlcs_med >= 0.07))
        )
    )

    # Level 3: substantial instability/shear plus a strong organized-storm
    # signal. The QLCS branch captures high-end line events; the supercell branch
    # captures conditional discrete development.
    strong_organization = (supercell_med >= 0.18) | (qlcs_med >= 0.09)
    level3 = tstm & (
        (severe >= 0.28)
        | (
            (cape_med >= 1700.0)
            & (shear06_med >= 20.0)
            & strong_organization
            & ((shear01_med >= 10.0) | (srh03_max >= 100.0))
        )
    )

    # Level 4 is intentionally rare and requires both very high severe guidance
    # and a significant hazard, or an extreme, strongly rotating environment.
    level4 = tstm & (
        (
            (severe >= 0.48)
            & ((tornado >= 0.12) | (hail >= 0.42) | (wind >= 0.40))
        )
        | (
            (cape_med >= 2500.0)
            & (shear06_med >= 24.0)
            & (supercell_med >= 0.30)
            & (srh03_max >= 200.0)
            & (tornado >= 0.08)
        )
    )

    out = xr.zeros_like(severe, dtype=np.int8)
    out = xr.where(tstm, 1, out)
    out = xr.where(level1, 2, out)
    out = xr.where(level2, 3, out)
    out = xr.where(level3, 4, out)
    out = xr.where(level4, 5, out)
    out.name = "prevots_severity_level"
    out.attrs.update(
        level_names="0:NONE,1:Tempestades,2:Nivel 1,3:Nivel 2,4:Nivel 3,5:Nivel 4",
        calibrated=False,
        method=(
            "Dedicated PREVOTS-like heuristic using consensus hazards plus GFS/ECMWF "
            "CAPE, shear, helicity proxies and storm-mode diagnostics"
        ),
    )
    return out
