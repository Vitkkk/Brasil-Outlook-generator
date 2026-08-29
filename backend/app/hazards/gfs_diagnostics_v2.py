from __future__ import annotations

import numpy as np
import xarray as xr


HORIZONTAL_DIMS = {"valid_time", "latitude", "longitude"}


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
    level = level_hpa if dim == "isobaricInhPa" else level_hpa * 100.0
    return da.sel({dim: level}, method="nearest")


def _collapse_scalar(
    da: xr.DataArray,
    *,
    reduction: str = "first",
) -> xr.DataArray:
    """Collapse non-horizontal layer dimensions from scalar GRIB products."""
    work = da
    for dim in tuple(work.dims):
        if dim in HORIZONTAL_DIMS:
            continue
        if reduction == "max":
            work = work.max(dim=dim, skipna=True)
        elif reduction == "min":
            work = work.min(dim=dim, skipna=True)
        elif reduction == "maxabs":
            work = np.abs(work).max(dim=dim, skipna=True)
        else:
            work = work.isel({dim: 0}, drop=True)
    return work


def _field_or_zeros(ds: xr.Dataset, name: str, like: xr.DataArray) -> xr.DataArray:
    if name not in ds:
        return xr.zeros_like(like)
    return _collapse_scalar(ds[name])


def _terrain(ds: xr.Dataset, like: xr.DataArray) -> xr.DataArray:
    if "terrain_height" not in ds:
        return xr.zeros_like(like)
    terrain = _collapse_scalar(ds["terrain_height"])
    # Defensive unit handling: GFS HGT is normally metres, but reject impossible
    # magnitudes rather than allowing a malformed terrain group to break AGL interpolation.
    return xr.where(np.abs(terrain) < 9000.0, terrain, 0.0)


def _interp_profile_np(
    z_m: np.ndarray,
    value: np.ndarray,
    terrain_m: np.ndarray,
    target_agl_m: float,
) -> np.ndarray:
    """Vectorized linear interpolation of a profile to one AGL height.

    xarray moves the pressure-level core dimension to the final axis before this
    function is called. All remaining dimensions are flattened and solved in one
    NumPy operation, avoiding Python loops over tens of thousands of grid cells.
    """
    z = np.asarray(z_m, dtype=float)
    v = np.asarray(value, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)

    if z.shape != v.shape:
        raise ValueError("Height and value profile shapes must match")

    nlev = z.shape[-1]
    flat_z = (z - terrain[..., None]).reshape(-1, nlev)
    flat_v = v.reshape(-1, nlev)

    valid = np.isfinite(flat_z) & np.isfinite(flat_v)
    zsort_source = np.where(valid, flat_z, np.inf)
    order = np.argsort(zsort_source, axis=1)
    zs = np.take_along_axis(flat_z, order, axis=1)
    vs = np.take_along_axis(flat_v, order, axis=1)
    valid_sorted = np.take_along_axis(valid, order, axis=1)

    count = valid_sorted.sum(axis=1)
    zs = np.where(valid_sorted, zs, np.nan)
    vs = np.where(valid_sorted, vs, np.nan)

    npoints = zs.shape[0]
    rows = np.arange(npoints)
    below_count = np.sum(valid_sorted & (zs <= target_agl_m), axis=1)
    k0 = np.clip(below_count - 1, 0, max(nlev - 2, 0))
    k1 = np.clip(k0 + 1, 0, max(nlev - 1, 0))

    z0 = zs[rows, k0]
    z1 = zs[rows, k1]
    v0 = vs[rows, k0]
    v1 = vs[rows, k1]

    zmin = np.nanmin(zs, axis=1)
    zmax = np.nanmax(zs, axis=1)
    inside = (count >= 2) & (target_agl_m >= zmin) & (target_agl_m <= zmax)
    denom = z1 - z0
    weight = np.where(np.abs(denom) > 1e-6, (target_agl_m - z0) / denom, 0.0)
    out = v0 + weight * (v1 - v0)
    out = np.where(inside & np.isfinite(out), out, np.nan)
    return out.reshape(z.shape[:-1])


def _interp_to_agl(
    z: xr.DataArray,
    value: xr.DataArray,
    terrain: xr.DataArray,
    target_agl_m: float,
) -> xr.DataArray:
    level_dim = _profile_dim(z)
    if _profile_dim(value) != level_dim:
        raise ValueError("Profile variables use different pressure dimensions")

    result = xr.apply_ufunc(
        _interp_profile_np,
        z,
        value,
        terrain,
        input_core_dims=[[level_dim], [level_dim], []],
        output_core_dims=[[]],
        kwargs={"target_agl_m": float(target_agl_m)},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
    )
    return result


def _freezing_level_np(
    z_m: np.ndarray,
    temperature_k: np.ndarray,
    terrain_m: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z_m, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)
    nlev = z.shape[-1]

    flat_z = (z - terrain[..., None]).reshape(-1, nlev)
    flat_t = t.reshape(-1, nlev)
    valid = np.isfinite(flat_z) & np.isfinite(flat_t)
    source = np.where(valid, flat_z, np.inf)
    order = np.argsort(source, axis=1)
    zs = np.take_along_axis(flat_z, order, axis=1)
    ts = np.take_along_axis(flat_t, order, axis=1)
    ok = np.take_along_axis(valid, order, axis=1)

    zs = np.where(ok, zs, np.nan)
    ts = np.where(ok, ts, np.nan)
    warm = ts >= 273.15
    cold = ts < 273.15
    crossing = warm[:, :-1] & cold[:, 1:] & ok[:, :-1] & ok[:, 1:]
    has_cross = crossing.any(axis=1)
    idx = crossing.argmax(axis=1)
    rows = np.arange(zs.shape[0])

    z0 = zs[rows, idx]
    z1 = zs[rows, idx + 1]
    t0 = ts[rows, idx]
    t1 = ts[rows, idx + 1]
    denom = t1 - t0
    frac = np.where(np.abs(denom) > 1e-6, (273.15 - t0) / denom, 0.0)
    level = z0 + frac * (z1 - z0)

    # Entirely subfreezing profiles get a zero-metre freezing level; entirely
    # warm profiles remain NaN because the freezing level is above the profile.
    all_cold = np.any(ok, axis=1) & np.all((~ok) | cold, axis=1)
    out = np.where(has_cross, level, np.where(all_cold, 0.0, np.nan))
    return out.reshape(z.shape[:-1])


def _freezing_level_agl(
    z: xr.DataArray,
    temperature: xr.DataArray,
    terrain: xr.DataArray,
) -> xr.DataArray:
    level_dim = _profile_dim(z)
    return xr.apply_ufunc(
        _freezing_level_np,
        z,
        temperature,
        terrain,
        input_core_dims=[[level_dim], [level_dim], []],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
    )


def _wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.hypot(u, v)


def _srh_segment(
    u0: xr.DataArray,
    v0: xr.DataArray,
    u1: xr.DataArray,
    v1: xr.DataArray,
    storm_u: xr.DataArray,
    storm_v: xr.DataArray,
) -> xr.DataArray:
    # Discrete hodograph-area form of storm-relative helicity for one segment.
    return (u1 - storm_u) * (v0 - storm_v) - (u0 - storm_u) * (v1 - storm_v)


def _lcl_height_proxy(t2m_k: xr.DataArray, td2m_k: xr.DataArray) -> xr.DataArray:
    # Bolton-style practical approximation: ~125 m per degree C of dewpoint depression.
    depression = xr.apply_ufunc(np.maximum, t2m_k - td2m_k, 0.0)
    return depression * 125.0


def _surface_fields(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    missing = [name for name in ("temperature_2m", "dewpoint_2m", "u_wind_10m", "v_wind_10m") if name not in ds]
    if missing:
        raise KeyError(f"GFS V2 requires surface fields: {', '.join(missing)}")
    return (
        _collapse_scalar(ds["temperature_2m"]),
        _collapse_scalar(ds["dewpoint_2m"]),
        _collapse_scalar(ds["u_wind_10m"]),
        _collapse_scalar(ds["v_wind_10m"]),
    )


def _native_instability(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    if "native_cape" not in ds or "native_cin" not in ds:
        raise KeyError("GFS V2 requires native CAPE/CIN until full parcel reconstruction lands")
    cape = _collapse_scalar(ds["native_cape"], reduction="max")
    cin = _collapse_scalar(ds["native_cin"], reduction="min")
    return cape, cin


def gfs_diagnostics_v2_probabilities(ds: xr.Dataset) -> xr.Dataset:
    """Second-generation GFS severe-convective diagnostic engine.

    V2 removes the 10 m-to-500 hPa shear shortcut used by V1. Pressure-level wind
    and geopotential-height profiles are interpolated to 0.5/1/3/6 km AGL, which
    allows explicit 0-1/0-3/0-6 km bulk shear, a storm-motion-relative SRH proxy,
    low-level lapse rate, LCL height and freezing-level diagnostics.

    Probabilities remain *uncalibrated engineering guidance*. Native GFS CAPE/CIN
    are still used as instability features; parcel reconstruction and historical
    statistical calibration are later stages.
    """
    required_profiles = (
        "air_temperature",
        "eastward_wind",
        "northward_wind",
        "geopotential_height",
    )
    missing = [name for name in required_profiles if name not in ds]
    if missing:
        raise KeyError(f"GFS V2 missing pressure-profile fields: {', '.join(missing)}")

    cape, cin = _native_instability(ds)
    t2m, td2m, u0, v0 = _surface_fields(ds)
    terrain = _terrain(ds, t2m)

    z = ds["geopotential_height"]
    u = ds["eastward_wind"]
    v = ds["northward_wind"]
    temp = ds["air_temperature"]

    u500m = _interp_to_agl(z, u, terrain, 500.0)
    v500m = _interp_to_agl(z, v, terrain, 500.0)
    u1km = _interp_to_agl(z, u, terrain, 1000.0)
    v1km = _interp_to_agl(z, v, terrain, 1000.0)
    u3km = _interp_to_agl(z, u, terrain, 3000.0)
    v3km = _interp_to_agl(z, v, terrain, 3000.0)
    u6km = _interp_to_agl(z, u, terrain, 6000.0)
    v6km = _interp_to_agl(z, v, terrain, 6000.0)
    t3km = _interp_to_agl(z, temp, terrain, 3000.0)

    shear01_u = u1km - u0
    shear01_v = v1km - v0
    shear03_u = u3km - u0
    shear03_v = v3km - v0
    shear06_u = u6km - u0
    shear06_v = v6km - v0
    shear01 = _wind_speed(shear01_u, shear01_v)
    shear03 = _wind_speed(shear03_u, shear03_v)
    shear06 = _wind_speed(shear06_u, shear06_v)

    storm_u = _field_or_zeros(ds, "storm_motion_u", cape)
    storm_v = _field_or_zeros(ds, "storm_motion_v", cape)
    # If the native storm-motion product is absent, fall back to the 0-6 km mean
    # vector proxy instead of treating storm motion as zero.
    if "storm_motion_u" not in ds or "storm_motion_v" not in ds:
        storm_u = 0.5 * (u0 + u6km)
        storm_v = 0.5 * (v0 + v6km)

    srh_0_1 = (
        _srh_segment(u0, v0, u500m, v500m, storm_u, storm_v)
        + _srh_segment(u500m, v500m, u1km, v1km, storm_u, storm_v)
    )
    srh_0_3 = srh_0_1 + _srh_segment(u1km, v1km, u3km, v3km, storm_u, storm_v)

    sr_u = u0 - storm_u
    sr_v = v0 - storm_v
    sr_flow_surface = _wind_speed(sr_u, sr_v)
    # Horizontal-vorticity vector is approximately (-dv/dz, du/dz); alignment
    # with storm-relative inflow gives a compact streamwise-vorticity fraction.
    streamwise_num = np.abs(sr_u * (-shear01_v) + sr_v * shear01_u)
    streamwise_den = xr.apply_ufunc(np.maximum, sr_flow_surface * shear01, 1e-6)
    streamwise_fraction = _clip01(streamwise_num / streamwise_den)

    lcl = _lcl_height_proxy(t2m, td2m)
    lapse03 = (t2m - t3km) / 3.0
    t700 = _select_pressure(temp, 700.0)
    t500 = _select_pressure(temp, 500.0)
    z700 = _select_pressure(z, 700.0)
    z500 = _select_pressure(z, 500.0)
    dz_km = xr.where(np.abs(z500 - z700) > 100.0, np.abs(z500 - z700) / 1000.0, np.nan)
    lapse_700_500 = (t700 - t500) / dz_km
    freezing_level = _freezing_level_agl(z, temp, terrain)

    u850 = _select_pressure(u, 850.0)
    v850 = _select_pressure(v, 850.0)
    u700 = _select_pressure(u, 700.0)
    v700 = _select_pressure(v, 700.0)
    midlevel_wind = xr.apply_ufunc(
        np.maximum,
        _wind_speed(u850, v850),
        _wind_speed(u700, v700),
    )
    storm_relative_850 = _wind_speed(u850 - storm_u, v850 - storm_v)

    pwat = _field_or_zeros(ds, "precipitable_water", cape)
    gust = _field_or_zeros(ds, "surface_gust", cape)
    refc = _field_or_zeros(ds, "composite_reflectivity", cape)
    omega700 = xr.zeros_like(cape)
    if "lagrangian_tendency_of_air_pressure" in ds:
        omega700 = _select_pressure(ds["lagrangian_tendency_of_air_pressure"], 700.0)

    cape_f = _norm(cape, 300.0, 2600.0)
    cin_f = 1.0 - _norm(np.abs(cin), 40.0, 220.0)
    pwat_f = _norm(pwat, 18.0, 48.0)
    shear01_f = _norm(shear01, 5.0, 18.0)
    shear03_f = _norm(shear03, 10.0, 25.0)
    shear06_f = _norm(shear06, 14.0, 30.0)
    srh01_f = _norm(np.abs(srh_0_1), 60.0, 300.0)
    srh03_f = _norm(np.abs(srh_0_3), 100.0, 450.0)
    lapse03_f = _norm(lapse03, 5.5, 8.5)
    lapse_mid_f = _norm(lapse_700_500, 5.5, 8.0)
    lcl_f = 1.0 - _norm(lcl, 800.0, 1800.0)
    dewpoint_dep_f = 1.0 - _norm(t2m - td2m, 5.0, 18.0)
    streamwise_f = _clip01(streamwise_fraction)
    sr850_f = _norm(storm_relative_850, 8.0, 28.0)
    midwind_f = _norm(midlevel_wind, 12.0, 30.0)
    gust_f = _norm(gust, 17.0, 35.0)
    forcing_f = _norm(-omega700, 0.03, 0.45)
    convective_signal = xr.where(refc >= 20.0, 1.0, 0.0)

    freezing_survival_f = 1.0 - _norm(freezing_level, 3500.0, 5200.0)
    freezing_survival_f = xr.where(np.isfinite(freezing_survival_f), freezing_survival_f, 0.35)

    surface_inflow = _clip01(
        0.34 * cin_f
        + 0.25 * lcl_f
        + 0.21 * dewpoint_dep_f
        + 0.20 * cape_f
    )

    initiation_score = (
        0.26 * cape_f
        + 0.18 * cin_f
        + 0.12 * pwat_f
        + 0.18 * forcing_f
        + 0.26 * convective_signal
    )
    initiation = _clip01(_sigmoid(initiation_score, midpoint=0.40, steepness=6.5))

    supercell_score = (
        0.34 * shear06_f
        + 0.18 * shear03_f
        + 0.18 * srh03_f
        + 0.12 * streamwise_f
        + 0.18 * cape_f
    )
    supercell = _clip01(initiation * _sigmoid(supercell_score, midpoint=0.47, steepness=7.5))

    qlcs_score = (
        0.30 * shear03_f
        + 0.20 * forcing_f
        + 0.20 * midwind_f
        + 0.15 * gust_f
        + 0.15 * initiation
    )
    qlcs = _clip01(initiation * _sigmoid(qlcs_score, midpoint=0.48, steepness=7.0))

    tornado_score = (
        0.19 * srh01_f
        + 0.14 * srh03_f
        + 0.14 * shear01_f
        + 0.13 * shear06_f
        + 0.12 * lcl_f
        + 0.10 * streamwise_f
        + 0.10 * surface_inflow
        + 0.08 * cape_f
    )
    tornado = _clip01(
        0.32
        * initiation
        * (0.45 + 0.55 * supercell)
        * _sigmoid(tornado_score, midpoint=0.50, steepness=8.5)
    )

    sig_tornado_support = _clip01(
        supercell
        * _sigmoid(
            0.27 * srh01_f
            + 0.22 * shear06_f
            + 0.18 * shear01_f
            + 0.16 * lcl_f
            + 0.10 * cape_f
            + 0.07 * streamwise_f,
            midpoint=0.58,
            steepness=9.0,
        )
    )

    hail_score = (
        0.28 * cape_f
        + 0.24 * shear06_f
        + 0.18 * lapse_mid_f
        + 0.12 * supercell
        + 0.10 * freezing_survival_f
        + 0.08 * sr850_f
    )
    hail = _clip01(0.62 * initiation * _sigmoid(hail_score, midpoint=0.47, steepness=7.5))

    wind_score = (
        0.22 * gust_f
        + 0.20 * lapse03_f
        + 0.18 * midwind_f
        + 0.15 * shear03_f
        + 0.12 * cape_f
        + 0.13 * qlcs
    )
    wind = _clip01(0.62 * initiation * _sigmoid(wind_score, midpoint=0.46, steepness=7.2))

    # Severe-any remains deliberately conservative relative to a literal union of
    # highly correlated hazard probabilities. A multi-hazard overlap bonus is
    # applied without allowing correlated hazards to sum directly to 100%.
    max_hazard = xr.apply_ufunc(np.maximum, hail, wind)
    max_hazard = xr.apply_ufunc(np.maximum, max_hazard, tornado)
    union = 1.0 - (1.0 - hail) * (1.0 - wind) * (1.0 - tornado)
    severe = _clip01(max_hazard + 0.35 * (union - max_hazard))
    thunder = _clip01(0.08 + 0.92 * initiation)

    result = xr.Dataset(
        {
            "thunderstorm": thunder,
            "severe": severe,
            "tornado": tornado,
            "hail": hail,
            "wind": wind,
            "convective_initiation": initiation,
            "supercell": supercell,
            "qlcs": qlcs,
            "sig_tornado_support": sig_tornado_support,
            "shear_0_1km_ms": shear01,
            "shear_0_3km_ms": shear03,
            "shear_0_6km_ms": shear06,
            "srh_0_1km_proxy_m2s2": srh_0_1,
            "srh_0_3km_proxy_m2s2": srh_0_3,
            "streamwise_fraction_proxy": streamwise_fraction,
            "lcl_height_proxy_m": lcl,
            "lapse_0_3km_c_per_km": lapse03,
            "lapse_700_500_c_per_km": lapse_700_500,
            "freezing_level_agl_m": freezing_level,
            "storm_relative_850_ms": storm_relative_850,
            "surface_inflow_quality": surface_inflow,
        }
    )
    result.attrs.update(
        source="GFS_0p25_diagnostics_v2",
        calibrated=False,
        probability_radius_km=40,
        warning=(
            "GFS Diagnostics V2 uses height-interpolated kinematics and expanded thermodynamic "
            "features, but hazard probabilities are not yet statistically calibrated. Native GFS "
            "CAPE/CIN remain temporary instability inputs pending full parcel reconstruction."
        ),
    )
    return result


def _collapse_single_valid_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" not in ds.dims:
        return ds
    if ds.sizes["valid_time"] == 1:
        return ds.isel(valid_time=0, drop=True)
    return ds.max("valid_time", skipna=True)


def aggregate_day1_v2(hourly: list[xr.Dataset]) -> xr.Dataset:
    if not hourly:
        raise ValueError("At least one forecast dataset is required")

    normalized: list[xr.Dataset] = []
    for sample_index, ds in enumerate(hourly):
        work = _collapse_single_valid_time(ds)
        normalized.append(work.expand_dims(forecast_sample=[sample_index]))

    stacked = xr.concat(
        normalized,
        dim="forecast_sample",
        join="exact",
        compat="override",
        coords="minimal",
    )

    probability_products = (
        "thunderstorm",
        "severe",
        "tornado",
        "hail",
        "wind",
        "convective_initiation",
        "supercell",
        "qlcs",
        "sig_tornado_support",
    )
    diagnostic_max = (
        "shear_0_1km_ms",
        "shear_0_3km_ms",
        "shear_0_6km_ms",
        "streamwise_fraction_proxy",
        "lapse_0_3km_c_per_km",
        "lapse_700_500_c_per_km",
        "storm_relative_850_ms",
        "surface_inflow_quality",
    )
    diagnostic_absmax = (
        "srh_0_1km_proxy_m2s2",
        "srh_0_3km_proxy_m2s2",
    )
    diagnostic_min = (
        "lcl_height_proxy_m",
        "freezing_level_agl_m",
    )

    output_vars: dict[str, xr.DataArray] = {}
    for name in probability_products + diagnostic_max:
        output_vars[name] = stacked[name].max("forecast_sample", skipna=True)
    for name in diagnostic_absmax:
        idx = np.abs(stacked[name]).argmax("forecast_sample", skipna=True)
        output_vars[name] = stacked[name].isel(forecast_sample=idx)
    for name in diagnostic_min:
        output_vars[name] = stacked[name].min("forecast_sample", skipna=True)

    output = xr.Dataset(output_vars)
    output.attrs.update(hourly[0].attrs)
    output.attrs["aggregation"] = "Day-1 extrema across sampled forecast hours"
    return output
