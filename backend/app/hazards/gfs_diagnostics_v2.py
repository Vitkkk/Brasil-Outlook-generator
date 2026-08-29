from __future__ import annotations

import numpy as np
import xarray as xr


HORIZONTAL_DIMS = {"valid_time", "latitude", "longitude"}


def _clip01(x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.clip, x, 0.0, 1.0)


def _norm(x: xr.DataArray, low: float, high: float) -> xr.DataArray:
    return _clip01((x - low) / max(high - low, 1e-6))


def _sigmoid(x: xr.DataArray, midpoint: float = 0.5, steepness: float = 7.0) -> xr.DataArray:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def _clean(x: xr.DataArray) -> xr.DataArray:
    """Drop scalar GRIB metadata coordinates while preserving dimension indexes."""
    return x.reset_coords(drop=True)


def _profile_dim(da: xr.DataArray) -> str:
    for dim in ("isobaricInhPa", "isobaricInPa"):
        if dim in da.dims:
            return dim
    raise KeyError(f"No pressure-level dimension found for {da.name}")


def _select_pressure(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    dim = _profile_dim(da)
    level = level_hpa if dim == "isobaricInhPa" else level_hpa * 100.0
    return _clean(da.sel({dim: level}, method="nearest"))


def _collapse_scalar(da: xr.DataArray, reduction: str = "first") -> xr.DataArray:
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
    return _clean(work)


def _field_or_zeros(ds: xr.Dataset, name: str, like: xr.DataArray) -> xr.DataArray:
    return _collapse_scalar(ds[name]) if name in ds else xr.zeros_like(like)


def _terrain(ds: xr.Dataset, like: xr.DataArray) -> xr.DataArray:
    if "terrain_height" not in ds:
        return xr.zeros_like(like)
    terrain = _collapse_scalar(ds["terrain_height"])
    return xr.where(np.abs(terrain) < 9000.0, terrain, 0.0)


def _interp_profile_np(z_m, value, terrain_m, target_agl_m: float):
    z = np.asarray(z_m, dtype=float)
    v = np.asarray(value, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)
    if z.shape != v.shape:
        raise ValueError("Height and profile-value shapes must match")

    nlev = z.shape[-1]
    flat_z = (z - terrain[..., None]).reshape(-1, nlev)
    flat_v = v.reshape(-1, nlev)
    valid = np.isfinite(flat_z) & np.isfinite(flat_v)
    order = np.argsort(np.where(valid, flat_z, np.inf), axis=1)
    zs = np.take_along_axis(flat_z, order, axis=1)
    vs = np.take_along_axis(flat_v, order, axis=1)
    ok = np.take_along_axis(valid, order, axis=1)
    zs = np.where(ok, zs, np.nan)
    vs = np.where(ok, vs, np.nan)

    rows = np.arange(zs.shape[0])
    count = ok.sum(axis=1)
    below = np.sum(ok & (zs <= target_agl_m), axis=1)
    k0 = np.clip(below - 1, 0, max(nlev - 2, 0))
    k1 = np.clip(k0 + 1, 0, max(nlev - 1, 0))
    z0, z1 = zs[rows, k0], zs[rows, k1]
    v0, v1 = vs[rows, k0], vs[rows, k1]
    zmin, zmax = np.nanmin(zs, axis=1), np.nanmax(zs, axis=1)
    inside = (count >= 2) & (target_agl_m >= zmin) & (target_agl_m <= zmax)
    denom = z1 - z0
    w = np.where(np.abs(denom) > 1e-6, (target_agl_m - z0) / denom, 0.0)
    out = v0 + w * (v1 - v0)
    out = np.where(inside & np.isfinite(out), out, np.nan)
    return out.reshape(z.shape[:-1])


def _interp_to_agl(z, value, terrain, target_agl_m: float) -> xr.DataArray:
    dim = _profile_dim(z)
    result = xr.apply_ufunc(
        _interp_profile_np,
        z,
        value,
        terrain,
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[]],
        kwargs={"target_agl_m": float(target_agl_m)},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
    )
    return _clean(result)


def _freezing_level_np(z_m, temperature_k, terrain_m):
    z = np.asarray(z_m, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)
    nlev = z.shape[-1]
    flat_z = (z - terrain[..., None]).reshape(-1, nlev)
    flat_t = t.reshape(-1, nlev)
    valid = np.isfinite(flat_z) & np.isfinite(flat_t)
    order = np.argsort(np.where(valid, flat_z, np.inf), axis=1)
    zs = np.take_along_axis(flat_z, order, axis=1)
    ts = np.take_along_axis(flat_t, order, axis=1)
    ok = np.take_along_axis(valid, order, axis=1)
    zs, ts = np.where(ok, zs, np.nan), np.where(ok, ts, np.nan)

    warm, cold = ts >= 273.15, ts < 273.15
    crossing = warm[:, :-1] & cold[:, 1:] & ok[:, :-1] & ok[:, 1:]
    has = crossing.any(axis=1)
    idx = crossing.argmax(axis=1)
    rows = np.arange(zs.shape[0])
    z0, z1 = zs[rows, idx], zs[rows, idx + 1]
    t0, t1 = ts[rows, idx], ts[rows, idx + 1]
    frac = np.where(np.abs(t1 - t0) > 1e-6, (273.15 - t0) / (t1 - t0), 0.0)
    level = z0 + frac * (z1 - z0)
    all_cold = np.any(ok, axis=1) & np.all((~ok) | cold, axis=1)
    out = np.where(has, level, np.where(all_cold, 0.0, np.nan))
    return out.reshape(z.shape[:-1])


def _freezing_level_agl(z, temperature, terrain) -> xr.DataArray:
    dim = _profile_dim(z)
    result = xr.apply_ufunc(
        _freezing_level_np,
        z,
        temperature,
        terrain,
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float],
    )
    return _clean(result)


def _speed(u, v):
    return np.hypot(u, v)


def _srh_segment(u0, v0, u1, v1, storm_u, storm_v):
    return (u1 - storm_u) * (v0 - storm_v) - (u0 - storm_u) * (v1 - storm_v)


def _lcl_proxy(t2m, td2m):
    return xr.apply_ufunc(np.maximum, t2m - td2m, 0.0) * 125.0


def _surface_fields(ds):
    names = ("temperature_2m", "dewpoint_2m", "u_wind_10m", "v_wind_10m")
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"GFS V2 requires surface fields: {', '.join(missing)}")
    return tuple(_collapse_scalar(ds[name]) for name in names)


def _native_instability(ds):
    if "native_cape" not in ds or "native_cin" not in ds:
        raise KeyError("GFS V2 requires native CAPE/CIN pending full parcel reconstruction")
    return (
        _collapse_scalar(ds["native_cape"], "max"),
        _collapse_scalar(ds["native_cin"], "min"),
    )


def gfs_diagnostics_v2_probabilities(ds: xr.Dataset) -> xr.Dataset:
    """Live GFS V2 diagnostics with height-resolved kinematics.

    V2 replaces the V1 10-m-to-500-hPa shear shortcut with wind/geopotential
    interpolation at 0.5/1/3/6 km AGL. It adds SRH, streamwise-flow, LCL,
    low-level lapse rate, freezing level, storm-relative 850-hPa flow,
    supercell and QLCS features. Hazard probabilities remain uncalibrated.
    """
    required = ("air_temperature", "eastward_wind", "northward_wind", "geopotential_height")
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"GFS V2 missing profile fields: {', '.join(missing)}")

    cape, cin = _native_instability(ds)
    t2m, td2m, u0, v0 = _surface_fields(ds)
    terrain = _terrain(ds, t2m)
    z, u, v, temp = (
        ds["geopotential_height"],
        ds["eastward_wind"],
        ds["northward_wind"],
        ds["air_temperature"],
    )

    u05, v05 = _interp_to_agl(z, u, terrain, 500.0), _interp_to_agl(z, v, terrain, 500.0)
    u1, v1 = _interp_to_agl(z, u, terrain, 1000.0), _interp_to_agl(z, v, terrain, 1000.0)
    u3, v3 = _interp_to_agl(z, u, terrain, 3000.0), _interp_to_agl(z, v, terrain, 3000.0)
    u6, v6 = _interp_to_agl(z, u, terrain, 6000.0), _interp_to_agl(z, v, terrain, 6000.0)
    t3 = _interp_to_agl(z, temp, terrain, 3000.0)

    du01, dv01 = u1 - u0, v1 - v0
    du03, dv03 = u3 - u0, v3 - v0
    du06, dv06 = u6 - u0, v6 - v0
    shear01, shear03, shear06 = _speed(du01, dv01), _speed(du03, dv03), _speed(du06, dv06)

    if "storm_motion_u" in ds and "storm_motion_v" in ds:
        storm_u = _collapse_scalar(ds["storm_motion_u"])
        storm_v = _collapse_scalar(ds["storm_motion_v"])
    else:
        storm_u, storm_v = 0.5 * (u0 + u6), 0.5 * (v0 + v6)

    srh01 = _srh_segment(u0, v0, u05, v05, storm_u, storm_v) + _srh_segment(
        u05, v05, u1, v1, storm_u, storm_v
    )
    srh03 = srh01 + _srh_segment(u1, v1, u3, v3, storm_u, storm_v)

    sr_u, sr_v = u0 - storm_u, v0 - storm_v
    sr_flow = _speed(sr_u, sr_v)
    stream_num = np.abs(sr_u * (-dv01) + sr_v * du01)
    stream_den = xr.apply_ufunc(np.maximum, sr_flow * shear01, 1e-6)
    streamwise = _clip01(stream_num / stream_den)

    lcl = _lcl_proxy(t2m, td2m)
    lapse03 = (t2m - t3) / 3.0
    t700, t500 = _select_pressure(temp, 700.0), _select_pressure(temp, 500.0)
    z700, z500 = _select_pressure(z, 700.0), _select_pressure(z, 500.0)
    dz_km = xr.where(np.abs(z500 - z700) > 100.0, np.abs(z500 - z700) / 1000.0, np.nan)
    lapse_mid = _clean((t700 - t500) / dz_km)
    freezing = _freezing_level_agl(z, temp, terrain)

    u850, v850 = _select_pressure(u, 850.0), _select_pressure(v, 850.0)
    u700, v700 = _select_pressure(u, 700.0), _select_pressure(v, 700.0)
    midwind = xr.apply_ufunc(np.maximum, _speed(u850, v850), _speed(u700, v700))
    sr850 = _speed(u850 - storm_u, v850 - storm_v)

    pwat = _field_or_zeros(ds, "precipitable_water", cape)
    gust = _field_or_zeros(ds, "surface_gust", cape)
    refc = _field_or_zeros(ds, "composite_reflectivity", cape)
    omega700 = (
        _select_pressure(ds["lagrangian_tendency_of_air_pressure"], 700.0)
        if "lagrangian_tendency_of_air_pressure" in ds
        else xr.zeros_like(cape)
    )

    cape_f = _norm(cape, 300.0, 2600.0)
    cin_f = 1.0 - _norm(np.abs(cin), 40.0, 220.0)
    pwat_f = _norm(pwat, 18.0, 48.0)
    s01_f, s03_f, s06_f = _norm(shear01, 5.0, 18.0), _norm(shear03, 10.0, 25.0), _norm(shear06, 14.0, 30.0)
    h01_f, h03_f = _norm(np.abs(srh01), 60.0, 300.0), _norm(np.abs(srh03), 100.0, 450.0)
    lr03_f, lrmid_f = _norm(lapse03, 5.5, 8.5), _norm(lapse_mid, 5.5, 8.0)
    lcl_f = 1.0 - _norm(lcl, 800.0, 1800.0)
    dew_f = 1.0 - _norm(t2m - td2m, 5.0, 18.0)
    sr850_f, midwind_f, gust_f = _norm(sr850, 8.0, 28.0), _norm(midwind, 12.0, 30.0), _norm(gust, 17.0, 35.0)
    forcing_f = _norm(-omega700, 0.03, 0.45)
    conv_signal = xr.where(refc >= 20.0, 1.0, 0.0)
    freeze_f = 1.0 - _norm(freezing, 3500.0, 5200.0)
    freeze_f = xr.where(np.isfinite(freeze_f), freeze_f, 0.35)

    surface_inflow = _clip01(0.34 * cin_f + 0.25 * lcl_f + 0.21 * dew_f + 0.20 * cape_f)
    initiation = _clip01(_sigmoid(
        0.26 * cape_f + 0.18 * cin_f + 0.12 * pwat_f + 0.18 * forcing_f + 0.26 * conv_signal,
        midpoint=0.40,
        steepness=6.5,
    ))
    supercell = _clip01(initiation * _sigmoid(
        0.34 * s06_f + 0.18 * s03_f + 0.18 * h03_f + 0.12 * streamwise + 0.18 * cape_f,
        midpoint=0.47,
        steepness=7.5,
    ))
    qlcs = _clip01(initiation * _sigmoid(
        0.30 * s03_f + 0.20 * forcing_f + 0.20 * midwind_f + 0.15 * gust_f + 0.15 * initiation,
        midpoint=0.48,
        steepness=7.0,
    ))

    tornado_score = (
        0.19 * h01_f + 0.14 * h03_f + 0.14 * s01_f + 0.13 * s06_f
        + 0.12 * lcl_f + 0.10 * streamwise + 0.10 * surface_inflow + 0.08 * cape_f
    )
    tornado = _clip01(
        0.32 * initiation * (0.45 + 0.55 * supercell)
        * _sigmoid(tornado_score, midpoint=0.50, steepness=8.5)
    )
    sig_tornado = _clip01(supercell * _sigmoid(
        0.27 * h01_f + 0.22 * s06_f + 0.18 * s01_f + 0.16 * lcl_f
        + 0.10 * cape_f + 0.07 * streamwise,
        midpoint=0.58,
        steepness=9.0,
    ))

    hail = _clip01(0.62 * initiation * _sigmoid(
        0.28 * cape_f + 0.24 * s06_f + 0.18 * lrmid_f + 0.12 * supercell
        + 0.10 * freeze_f + 0.08 * sr850_f,
        midpoint=0.47,
        steepness=7.5,
    ))
    wind = _clip01(0.62 * initiation * _sigmoid(
        0.22 * gust_f + 0.20 * lr03_f + 0.18 * midwind_f + 0.15 * s03_f
        + 0.12 * cape_f + 0.13 * qlcs,
        midpoint=0.46,
        steepness=7.2,
    ))

    max_hazard = xr.apply_ufunc(np.maximum, xr.apply_ufunc(np.maximum, hail, wind), tornado)
    union = 1.0 - (1.0 - hail) * (1.0 - wind) * (1.0 - tornado)
    severe = _clip01(max_hazard + 0.35 * (union - max_hazard))
    thunder = _clip01(0.08 + 0.92 * initiation)

    variables = {
        "thunderstorm": thunder,
        "severe": severe,
        "tornado": tornado,
        "hail": hail,
        "wind": wind,
        "convective_initiation": initiation,
        "supercell": supercell,
        "qlcs": qlcs,
        "sig_tornado_support": sig_tornado,
        "shear_0_1km_ms": shear01,
        "shear_0_3km_ms": shear03,
        "shear_0_6km_ms": shear06,
        "srh_0_1km_proxy_m2s2": srh01,
        "srh_0_3km_proxy_m2s2": srh03,
        "streamwise_fraction_proxy": streamwise,
        "lcl_height_proxy_m": lcl,
        "lapse_0_3km_c_per_km": lapse03,
        "lapse_700_500_c_per_km": lapse_mid,
        "freezing_level_agl_m": freezing,
        "storm_relative_850_ms": sr850,
        "surface_inflow_quality": surface_inflow,
    }
    result = xr.Dataset({name: _clean(field) for name, field in variables.items()})
    result.attrs.update(
        source="GFS_0p25_diagnostics_v2",
        calibrated=False,
        probability_radius_km=40,
        warning=(
            "Height-resolved GFS V2 engineering guidance; probabilities are not statistically calibrated. "
            "Native GFS CAPE/CIN remain temporary instability inputs pending parcel reconstruction."
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
    samples = [
        _collapse_single_valid_time(ds).expand_dims(forecast_sample=[i])
        for i, ds in enumerate(hourly)
    ]
    stacked = xr.concat(samples, dim="forecast_sample", join="exact", compat="override", coords="minimal")

    max_fields = (
        "thunderstorm", "severe", "tornado", "hail", "wind", "convective_initiation",
        "supercell", "qlcs", "sig_tornado_support", "shear_0_1km_ms", "shear_0_3km_ms",
        "shear_0_6km_ms", "streamwise_fraction_proxy", "lapse_0_3km_c_per_km",
        "lapse_700_500_c_per_km", "storm_relative_850_ms", "surface_inflow_quality",
    )
    output = {name: stacked[name].max("forecast_sample", skipna=True) for name in max_fields}
    output["srh_0_1km_proxy_m2s2"] = np.abs(stacked["srh_0_1km_proxy_m2s2"]).max("forecast_sample", skipna=True)
    output["srh_0_3km_proxy_m2s2"] = np.abs(stacked["srh_0_3km_proxy_m2s2"]).max("forecast_sample", skipna=True)
    output["lcl_height_proxy_m"] = stacked["lcl_height_proxy_m"].min("forecast_sample", skipna=True)
    output["freezing_level_agl_m"] = stacked["freezing_level_agl_m"].min("forecast_sample", skipna=True)

    result = xr.Dataset(output)
    result.attrs.update(hourly[0].attrs)
    result.attrs["aggregation"] = "Day-1 extrema across sampled forecast hours"
    return result
