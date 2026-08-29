from __future__ import annotations

import numpy as np
import xarray as xr

from app.diagnostics.thermodynamics_v3 import calculate_gfs_thermodynamics_v3
from app.hazards.gfs_diagnostics_v2 import gfs_diagnostics_v2_probabilities


def _clip01(value: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.clip, value, 0.0, 1.0)


def _norm(value: xr.DataArray, low: float, high: float) -> xr.DataArray:
    return _clip01((value - low) / max(high - low, 1e-6))


def _sigmoid(value: xr.DataArray, midpoint: float = 0.5, steepness: float = 7.0) -> xr.DataArray:
    return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))


def _profile_dim(da: xr.DataArray) -> str:
    for dim in ("isobaricInhPa", "isobaricInPa"):
        if dim in da.dims:
            return dim
    raise KeyError(f"No pressure-level dimension found for {da.name}")


def _select_pressure(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    dim = _profile_dim(da)
    target = level_hpa if dim == "isobaricInhPa" else level_hpa * 100.0
    selected = da.sel({dim: target}, method="nearest")
    if dim in selected.coords and dim not in selected.dims:
        selected = selected.drop_vars(dim)
    return selected


def _collapse_scalar(da: xr.DataArray, reduction: str = "first") -> xr.DataArray:
    work = da
    for dim in tuple(work.dims):
        if dim in {"valid_time", "latitude", "longitude"}:
            continue
        if reduction == "max":
            work = work.max(dim=dim, skipna=True)
        else:
            work = work.isel({dim: 0}, drop=True)
    return work


def _field_or_zeros(ds: xr.Dataset, name: str, like: xr.DataArray) -> xr.DataArray:
    if name not in ds:
        return xr.zeros_like(like)
    return _collapse_scalar(ds[name])


def _clean(value: xr.DataArray, fill: float = 0.0) -> xr.DataArray:
    return xr.where(np.isfinite(value), value, fill)


def gfs_diagnostics_v3_probabilities(ds: xr.Dataset) -> xr.Dataset:
    """Thermodynamics-driven third-generation GFS severe-weather guidance.

    V3 keeps the height-resolved V2 kinematics, but the instability, cap, LCL and
    low-level buoyancy features are reconstructed from the vertical sounding.
    Native GFS CAPE/CIN no longer enter the V3 hazard equations.
    """
    thermo = calculate_gfs_thermodynamics_v3(ds)
    kin = gfs_diagnostics_v2_probabilities(ds)

    mlcape = thermo["mlcape_jkg"]
    mucape = thermo["mucape_jkg"]
    mlcin = thermo["mlcin_jkg"]
    lowcape = thermo["mlcape_0_3km_jkg"]
    lcl = thermo["ml_lcl_agl_m"]
    eff_depth = _clean(thermo["effective_inflow_depth_m"])
    eff_cape = _clean(thermo["effective_cape_jkg"])

    s01 = kin["shear_0_1km_ms"]
    s03 = kin["shear_0_3km_ms"]
    s06 = kin["shear_0_6km_ms"]
    h01 = np.abs(kin["srh_0_1km_proxy_m2s2"])
    h03 = np.abs(kin["srh_0_3km_proxy_m2s2"])
    streamwise = kin["streamwise_fraction_proxy"]
    lapse03 = kin["lapse_0_3km_c_per_km"]
    lapse_mid = kin["lapse_700_500_c_per_km"]
    freezing = kin["freezing_level_agl_m"]
    sr850 = kin["storm_relative_850_ms"]

    pwat = _field_or_zeros(ds, "precipitable_water", mlcape)
    gust = _field_or_zeros(ds, "surface_gust", mlcape)
    refc = _field_or_zeros(ds, "composite_reflectivity", mlcape)
    omega700 = (
        _select_pressure(ds["lagrangian_tendency_of_air_pressure"], 700.0)
        if "lagrangian_tendency_of_air_pressure" in ds
        else xr.zeros_like(mlcape)
    )
    u850 = _select_pressure(ds["eastward_wind"], 850.0)
    v850 = _select_pressure(ds["northward_wind"], 850.0)
    u700 = _select_pressure(ds["eastward_wind"], 700.0)
    v700 = _select_pressure(ds["northward_wind"], 700.0)
    midwind = xr.apply_ufunc(np.maximum, np.hypot(u850, v850), np.hypot(u700, v700))

    mlcape_f = _norm(_clean(mlcape), 250.0, 3000.0)
    mucape_f = _norm(_clean(mucape), 400.0, 3500.0)
    lowcape_f = _norm(_clean(lowcape), 40.0, 300.0)
    cin_f = 1.0 - _norm(np.abs(_clean(mlcin, -300.0)), 25.0, 220.0)
    lcl_f = 1.0 - _norm(_clean(lcl, 2500.0), 700.0, 1800.0)
    eff_depth_f = _norm(eff_depth, 100.0, 1400.0)
    eff_cape_f = _norm(eff_cape, 250.0, 2600.0)

    s01_f = _norm(s01, 5.0, 18.0)
    s03_f = _norm(s03, 10.0, 25.0)
    s06_f = _norm(s06, 14.0, 30.0)
    h01_f = _norm(h01, 60.0, 300.0)
    h03_f = _norm(h03, 100.0, 450.0)
    lr03_f = _norm(lapse03, 5.5, 8.5)
    lrmid_f = _norm(lapse_mid, 5.5, 8.0)
    sr850_f = _norm(sr850, 8.0, 28.0)
    pwat_f = _norm(pwat, 18.0, 48.0)
    gust_f = _norm(gust, 17.0, 35.0)
    midwind_f = _norm(midwind, 12.0, 30.0)
    forcing_f = _norm(-omega700, 0.03, 0.45)
    convective_signal = xr.where(refc >= 20.0, 1.0, 0.0)
    freeze_f = 1.0 - _norm(_clean(freezing, 5000.0), 3500.0, 5200.0)

    # In V3 initiation depends on reconstructed mixed-layer instability and CIN,
    # not the model-provided CAPE/CIN grids.
    initiation_score = (
        0.26 * mlcape_f
        + 0.22 * cin_f
        + 0.12 * pwat_f
        + 0.16 * forcing_f
        + 0.24 * convective_signal
    )
    initiation = _clip01(_sigmoid(initiation_score, midpoint=0.40, steepness=6.5))

    supercell_score = (
        0.25 * s06_f
        + 0.17 * h03_f
        + 0.14 * eff_depth_f
        + 0.13 * eff_cape_f
        + 0.11 * lowcape_f
        + 0.09 * streamwise
        + 0.11 * mlcape_f
    )
    supercell = _clip01(
        initiation * _sigmoid(supercell_score, midpoint=0.46, steepness=7.8)
    )

    qlcs_score = (
        0.29 * s03_f
        + 0.20 * forcing_f
        + 0.20 * midwind_f
        + 0.15 * gust_f
        + 0.16 * initiation
    )
    qlcs = _clip01(initiation * _sigmoid(qlcs_score, midpoint=0.48, steepness=7.0))

    tornado_score = (
        0.18 * h01_f
        + 0.12 * h03_f
        + 0.13 * s01_f
        + 0.10 * s06_f
        + 0.14 * lcl_f
        + 0.11 * lowcape_f
        + 0.09 * eff_depth_f
        + 0.06 * streamwise
        + 0.07 * cin_f
    )
    tornado = _clip01(
        0.34
        * initiation
        * (0.38 + 0.62 * supercell)
        * _sigmoid(tornado_score, midpoint=0.49, steepness=8.7)
    )

    sig_tornado_score = (
        0.23 * h01_f
        + 0.18 * s06_f
        + 0.14 * s01_f
        + 0.14 * lcl_f
        + 0.11 * lowcape_f
        + 0.10 * eff_depth_f
        + 0.06 * mlcape_f
        + 0.04 * streamwise
    )
    sig_tornado = _clip01(
        supercell * _sigmoid(sig_tornado_score, midpoint=0.57, steepness=9.2)
    )

    hail_score = (
        0.26 * mucape_f
        + 0.23 * s06_f
        + 0.18 * lrmid_f
        + 0.11 * supercell
        + 0.09 * freeze_f
        + 0.07 * sr850_f
        + 0.06 * eff_cape_f
    )
    hail = _clip01(
        0.64 * initiation * _sigmoid(hail_score, midpoint=0.46, steepness=7.7)
    )

    wind_score = (
        0.21 * gust_f
        + 0.20 * lr03_f
        + 0.18 * midwind_f
        + 0.15 * s03_f
        + 0.12 * mlcape_f
        + 0.08 * cin_f
        + 0.06 * qlcs
    )
    wind = _clip01(
        0.63 * initiation * _sigmoid(wind_score, midpoint=0.46, steepness=7.3)
    )

    max_hazard = xr.apply_ufunc(np.maximum, xr.apply_ufunc(np.maximum, hail, wind), tornado)
    union = 1.0 - (1.0 - hail) * (1.0 - wind) * (1.0 - tornado)
    severe = _clip01(max_hazard + 0.35 * (union - max_hazard))
    thunder = _clip01(0.08 + 0.92 * initiation)

    variables: dict[str, xr.DataArray] = {
        "thunderstorm": thunder,
        "severe": severe,
        "tornado": tornado,
        "hail": hail,
        "wind": wind,
        "convective_initiation": initiation,
        "supercell": supercell,
        "qlcs": qlcs,
        "sig_tornado_support": sig_tornado,
        "shear_0_1km_ms": s01,
        "shear_0_3km_ms": s03,
        "shear_0_6km_ms": s06,
        "srh_0_1km_proxy_m2s2": kin["srh_0_1km_proxy_m2s2"],
        "srh_0_3km_proxy_m2s2": kin["srh_0_3km_proxy_m2s2"],
        "streamwise_fraction_proxy": streamwise,
        "lapse_0_3km_c_per_km": lapse03,
        "lapse_700_500_c_per_km": lapse_mid,
        "freezing_level_agl_m": freezing,
        "storm_relative_850_ms": sr850,
    }
    variables.update({name: thermo[name] for name in thermo.data_vars})

    result = xr.Dataset({name: _clean(field) for name, field in variables.items()})
    # Preserve NaN semantics for geometric parcel levels where no LFC/EL/effective
    # layer exists; hazard features already converted those missing values safely.
    for name in (
        "sb_lfc_agl_m", "ml_lfc_agl_m", "mu_lfc_agl_m",
        "sb_el_agl_m", "ml_el_agl_m", "mu_el_agl_m",
        "effective_inflow_bottom_agl_m", "effective_inflow_top_agl_m",
        "effective_inflow_depth_m", "effective_cape_jkg",
    ):
        result[name] = thermo[name]

    result.attrs.update(
        source="GFS_0p25_diagnostics_v3",
        calibrated=False,
        probability_radius_km=40,
        native_cape_cin_used_for_v3_hazards=False,
        warning=(
            "GFS Diagnostics V3 engineering guidance. SB/ML/MU CAPE/CIN and parcel levels are "
            "reconstructed from the sounding, but hazard probabilities are not historically calibrated."
        ),
    )
    return result


def _collapse_single_valid_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" not in ds.dims:
        return ds
    if ds.sizes["valid_time"] == 1:
        return ds.isel(valid_time=0, drop=True)
    return ds.max("valid_time", skipna=True)


def aggregate_day1_v3(hourly: list[xr.Dataset]) -> xr.Dataset:
    if not hourly:
        raise ValueError("At least one forecast dataset is required")

    samples = [
        _collapse_single_valid_time(ds).expand_dims(forecast_sample=[i])
        for i, ds in enumerate(hourly)
    ]
    stacked = xr.concat(
        samples,
        dim="forecast_sample",
        join="exact",
        compat="override",
        coords="minimal",
    )

    max_fields = (
        "thunderstorm", "severe", "tornado", "hail", "wind",
        "convective_initiation", "supercell", "qlcs", "sig_tornado_support",
        "shear_0_1km_ms", "shear_0_3km_ms", "shear_0_6km_ms",
        "streamwise_fraction_proxy", "lapse_0_3km_c_per_km",
        "lapse_700_500_c_per_km", "storm_relative_850_ms",
        "sbcape_jkg", "mlcape_jkg", "mucape_jkg",
        "sbcape_0_3km_jkg", "mlcape_0_3km_jkg", "mucape_0_3km_jkg",
        "effective_inflow_depth_m", "effective_cape_jkg",
    )
    output = {
        name: stacked[name].max("forecast_sample", skipna=True)
        for name in max_fields
    }
    output["srh_0_1km_proxy_m2s2"] = np.abs(stacked["srh_0_1km_proxy_m2s2"]).max(
        "forecast_sample", skipna=True
    )
    output["srh_0_3km_proxy_m2s2"] = np.abs(stacked["srh_0_3km_proxy_m2s2"]).max(
        "forecast_sample", skipna=True
    )

    # For CIN/LCL/LFC, the most permissive Day-1 environment is the least
    # negative CIN and the lowest condensation/free-convection height.
    for name in ("sbcin_jkg", "mlcin_jkg", "mucin_jkg"):
        output[name] = stacked[name].max("forecast_sample", skipna=True)
    for name in (
        "sb_lcl_agl_m", "ml_lcl_agl_m", "mu_lcl_agl_m",
        "sb_lfc_agl_m", "ml_lfc_agl_m", "mu_lfc_agl_m",
        "freezing_level_agl_m", "effective_inflow_bottom_agl_m",
    ):
        output[name] = stacked[name].min("forecast_sample", skipna=True)
    for name in (
        "sb_el_agl_m", "ml_el_agl_m", "mu_el_agl_m",
        "effective_inflow_top_agl_m",
    ):
        output[name] = stacked[name].max("forecast_sample", skipna=True)
    output["mu_parcel_pressure_hpa"] = stacked["mu_parcel_pressure_hpa"].max(
        "forecast_sample", skipna=True
    )

    result = xr.Dataset(output)
    result.attrs.update(hourly[0].attrs)
    result.attrs["aggregation"] = "Day-1 extrema across 3-hourly forecast samples"
    return result
