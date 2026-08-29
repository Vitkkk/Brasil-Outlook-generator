from __future__ import annotations

import numpy as np
import xarray as xr


RD = 287.05
RV = 461.5
CP = 1004.0
G = 9.80665
LV = 2.5e6
EPS = RD / RV
KAPPA = RD / CP


OUTPUT_NAMES = (
    "sbcape_jkg",
    "sbcin_jkg",
    "mlcape_jkg",
    "mlcin_jkg",
    "mucape_jkg",
    "mucin_jkg",
    "sbcape_0_3km_jkg",
    "mlcape_0_3km_jkg",
    "mucape_0_3km_jkg",
    "sb_lcl_agl_m",
    "ml_lcl_agl_m",
    "mu_lcl_agl_m",
    "sb_lfc_agl_m",
    "ml_lfc_agl_m",
    "mu_lfc_agl_m",
    "sb_el_agl_m",
    "ml_el_agl_m",
    "mu_el_agl_m",
    "effective_inflow_bottom_agl_m",
    "effective_inflow_top_agl_m",
    "effective_inflow_depth_m",
    "effective_cape_jkg",
    "mu_parcel_pressure_hpa",
)


def _profile_dim(da: xr.DataArray) -> str:
    for dim in ("isobaricInhPa", "isobaricInPa"):
        if dim in da.dims:
            return dim
    raise KeyError(f"No pressure-level dimension found for {da.name}")


def _collapse_scalar(da: xr.DataArray) -> xr.DataArray:
    work = da
    for dim in tuple(work.dims):
        if dim in {"valid_time", "latitude", "longitude"}:
            continue
        work = work.isel({dim: 0}, drop=True)
    return work


def _terrain(ds: xr.Dataset, like: xr.DataArray) -> xr.DataArray:
    if "terrain_height" not in ds:
        return xr.zeros_like(like)
    terrain = _collapse_scalar(ds["terrain_height"])
    return xr.where(np.isfinite(terrain) & (np.abs(terrain) < 9000.0), terrain, 0.0)


def _surface_pressure_hpa(ds: xr.Dataset, terrain: xr.DataArray) -> xr.DataArray:
    if "surface_air_pressure" in ds:
        pressure = _collapse_scalar(ds["surface_air_pressure"])
        # GFS PRES is normally Pa. Keep the path tolerant of already-converted data.
        return xr.where(pressure > 2000.0, pressure / 100.0, pressure)

    # Defensive fallback only. The V3 NOMADS manifest requests PRES explicitly,
    # but a standard-atmosphere estimate prevents one missing GRIB group from
    # destroying an otherwise useful engineering run.
    base = xr.apply_ufunc(np.maximum, 1.0 - 2.25577e-5 * terrain, 0.20)
    return 1013.25 * base ** 5.2559


def _saturation_vapor_pressure_hpa(temperature_k: xr.DataArray) -> xr.DataArray:
    tc = temperature_k - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def _specific_humidity_profile(ds: xr.Dataset, temperature: xr.DataArray) -> xr.DataArray:
    if "specific_humidity" in ds:
        return xr.where(ds["specific_humidity"] >= 0.0, ds["specific_humidity"], np.nan)
    if "relative_humidity" not in ds:
        raise KeyError("V3 thermodynamics requires pressure-level specific humidity or RH")

    dim = _profile_dim(temperature)
    pressure = temperature[dim]
    p_hpa = pressure / 100.0 if dim == "isobaricInPa" else pressure
    es = _saturation_vapor_pressure_hpa(temperature)
    e = xr.apply_ufunc(np.minimum, es * ds["relative_humidity"] / 100.0, 0.99 * p_hpa)
    return EPS * e / (p_hpa - (1.0 - EPS) * e)


def _sat_vapor_pressure_np(temperature_k: np.ndarray) -> np.ndarray:
    tc = temperature_k - 273.15
    exponent = np.clip(17.67 * tc / np.maximum(tc + 243.5, 1.0), -40.0, 40.0)
    return 6.112 * np.exp(exponent)


def _qsat_np(temperature_k: np.ndarray, pressure_hpa: np.ndarray | float) -> np.ndarray:
    p = np.asarray(pressure_hpa, dtype=float)
    es = np.minimum(_sat_vapor_pressure_np(temperature_k), 0.99 * p)
    return EPS * es / np.maximum(p - (1.0 - EPS) * es, 1e-6)


def _dewpoint_from_q_np(q: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-8, 0.20)
    e = q * pressure_hpa / np.maximum(EPS + (1.0 - EPS) * q, 1e-8)
    ln_ratio = np.log(np.maximum(e, 1e-8) / 6.112)
    tc = 243.5 * ln_ratio / np.maximum(17.67 - ln_ratio, 1e-6)
    return tc + 273.15


def _q_from_dewpoint_np(dewpoint_k: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    e = np.minimum(_sat_vapor_pressure_np(dewpoint_k), 0.99 * pressure_hpa)
    return EPS * e / np.maximum(pressure_hpa - (1.0 - EPS) * e, 1e-6)


def _lcl_np(
    temperature_k: np.ndarray,
    dewpoint_k: np.ndarray,
    pressure_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    td = np.minimum(dewpoint_k, temperature_k)
    tlcl = 1.0 / (
        1.0 / np.maximum(td - 56.0, 1.0)
        + np.log(np.maximum(temperature_k / np.maximum(td, 150.0), 1e-6)) / 800.0
    ) + 56.0
    plcl = pressure_hpa * (tlcl / temperature_k) ** (CP / RD)
    return tlcl, np.clip(plcl, 50.0, pressure_hpa)


def _theta_e_np(
    temperature_k: np.ndarray,
    dewpoint_k: np.ndarray,
    pressure_hpa: np.ndarray,
    specific_humidity: np.ndarray,
) -> np.ndarray:
    tlcl, _ = _lcl_np(temperature_k, dewpoint_k, pressure_hpa)
    r = specific_humidity / np.maximum(1.0 - specific_humidity, 1e-6)
    theta = temperature_k * (1000.0 / pressure_hpa) ** (0.2854 * (1.0 - 0.28 * r))
    exponent = np.clip((3376.0 / tlcl - 2.54) * r * (1.0 + 0.81 * r), -10.0, 10.0)
    return theta * np.exp(exponent)


def _moist_dtemp_dlnp_np(temperature_k: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    qsat = _qsat_np(temperature_k, pressure_hpa)
    rs = qsat / np.maximum(1.0 - qsat, 1e-6)
    numerator = RD * temperature_k * (1.0 + LV * rs / (RD * temperature_k))
    denominator = CP + (LV**2 * rs * EPS) / (RD * temperature_k**2)
    return numerator / np.maximum(denominator, 1e-6)


def _parcel_diagnostics_flat(
    pressure_levels_hpa: np.ndarray,
    environment_temperature_k: np.ndarray,
    environment_q: np.ndarray,
    geopotential_height_m: np.ndarray,
    terrain_m: np.ndarray,
    parcel_pressure_hpa: np.ndarray,
    parcel_temperature_k: np.ndarray,
    parcel_dewpoint_k: np.ndarray,
    parcel_start_agl_m: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Vectorized pseudoadiabatic parcel ascent for an entire horizontal grid."""
    p = np.asarray(pressure_levels_hpa, dtype=float)
    t_env = np.asarray(environment_temperature_k, dtype=float)
    q_env = np.asarray(environment_q, dtype=float)
    z = np.asarray(geopotential_height_m, dtype=float)
    terrain = np.asarray(terrain_m, dtype=float)
    p0 = np.asarray(parcel_pressure_hpa, dtype=float)
    t0 = np.asarray(parcel_temperature_k, dtype=float)
    td0 = np.asarray(parcel_dewpoint_k, dtype=float)
    z0 = np.asarray(parcel_start_agl_m, dtype=float)

    npoints, nlev = t_env.shape
    start_ok = (
        np.isfinite(p0)
        & np.isfinite(t0)
        & np.isfinite(td0)
        & np.isfinite(z0)
        & (p0 >= p.min() - 1.0)
        & (p0 <= 1100.0)
        & (t0 > 180.0)
    )

    td0 = np.minimum(td0, t0)
    q0 = _q_from_dewpoint_np(td0, p0)
    tlcl, plcl = _lcl_np(t0, td0, p0)
    lcl_rise = (RD * (t0 + tlcl) * 0.5 / G) * np.log(
        np.maximum(p0, 1.0) / np.maximum(plcl, 1.0)
    )
    lcl_agl = z0 + np.maximum(lcl_rise, 0.0)

    t_parcel = np.full((npoints, nlev), np.nan, dtype=float)
    q_parcel = np.full((npoints, nlev), np.nan, dtype=float)
    last_t = tlcl.copy()
    last_p = plcl.copy()

    for i, level_p in enumerate(p):
        z_agl_i = z[:, i] - terrain
        valid = (
            start_ok
            & (level_p <= p0 + 0.1)
            & np.isfinite(t_env[:, i])
            & np.isfinite(q_env[:, i])
            & np.isfinite(z[:, i])
            & (z_agl_i >= z0 - 120.0)
        )
        dry = valid & (level_p >= plcl)
        moist = valid & (level_p < plcl)

        if np.any(dry):
            t_parcel[dry, i] = t0[dry] * (level_p / p0[dry]) ** KAPPA
            q_parcel[dry, i] = q0[dry]

        if np.any(moist):
            delta_ln_p = np.log(level_p / last_p[moist])
            slope_1 = _moist_dtemp_dlnp_np(last_t[moist], last_p[moist])
            predicted = last_t[moist] + slope_1 * delta_ln_p
            slope_2 = _moist_dtemp_dlnp_np(predicted, np.full(predicted.shape, level_p))
            new_t = last_t[moist] + 0.5 * (slope_1 + slope_2) * delta_ln_p
            t_parcel[moist, i] = new_t
            q_parcel[moist, i] = _qsat_np(new_t, level_p)
            last_t[moist] = new_t
            last_p[moist] = level_p

    tv_env = t_env * (1.0 + 0.61 * q_env)
    tv_parcel = t_parcel * (1.0 + 0.61 * q_parcel)
    buoyancy = G * (tv_parcel - tv_env) / tv_env
    z_agl = z - terrain[:, None]
    valid = (
        np.isfinite(buoyancy)
        & np.isfinite(z_agl)
        & (z_agl >= z0[:, None] - 120.0)
        & (p[None, :] <= p0[:, None] + 0.1)
    )

    lfc_candidate = valid & (buoyancy > 0.0) & (z_agl >= lcl_agl[:, None] - 100.0)
    has_lfc = lfc_candidate.any(axis=1) & start_ok
    lfc_idx = np.where(has_lfc, lfc_candidate.argmax(axis=1), nlev)

    indices = np.arange(nlev)[None, :]
    el_candidate = valid & (indices > lfc_idx[:, None]) & (buoyancy <= 0.0)
    has_el = el_candidate.any(axis=1)
    first_el = el_candidate.argmax(axis=1)
    last_valid = np.max(np.where(valid, indices, -1), axis=1)
    el_idx = np.where(has_el, first_el, np.maximum(last_valid, 0))

    b0 = buoyancy[:, :-1]
    b1 = buoyancy[:, 1:]
    dz = np.maximum(z_agl[:, 1:] - z_agl[:, :-1], 0.0)
    segment_index = np.arange(1, nlev)[None, :]

    positive_area = 0.5 * (np.maximum(b0, 0.0) + np.maximum(b1, 0.0)) * dz
    cape_layer = (
        has_lfc[:, None]
        & (segment_index >= lfc_idx[:, None])
        & (segment_index <= el_idx[:, None])
        & valid[:, :-1]
        & valid[:, 1:]
    )
    cape = np.nansum(np.where(cape_layer, positive_area, 0.0), axis=1)

    midpoint_agl = 0.5 * (z_agl[:, :-1] + z_agl[:, 1:])
    cape_03 = np.nansum(
        np.where(cape_layer & (midpoint_agl <= 3000.0), positive_area, 0.0), axis=1
    )

    negative_area = 0.5 * (np.minimum(b0, 0.0) + np.minimum(b1, 0.0)) * dz
    before_lfc = (
        (segment_index < lfc_idx[:, None])
        & valid[:, :-1]
        & valid[:, 1:]
    )
    cin = np.nansum(np.where(before_lfc, negative_area, 0.0), axis=1)

    no_lfc_layer = valid[:, :-1] & valid[:, 1:]
    no_lfc_cin = np.nansum(np.where(no_lfc_layer, negative_area, 0.0), axis=1)
    cin = np.where(has_lfc, cin, no_lfc_cin)

    rows = np.arange(npoints)
    safe_lfc = np.clip(lfc_idx, 0, nlev - 1)
    safe_el = np.clip(el_idx, 0, nlev - 1)
    lfc_agl = np.where(has_lfc, z_agl[rows, safe_lfc], np.nan)
    el_agl = np.where(has_lfc, z_agl[rows, safe_el], np.nan)

    cape = np.where(start_ok, np.maximum(cape, 0.0), np.nan)
    cape_03 = np.where(start_ok, np.maximum(cape_03, 0.0), np.nan)
    cin = np.where(start_ok, np.minimum(cin, 0.0), np.nan)
    lcl_agl = np.where(start_ok, lcl_agl, np.nan)
    return cape, cin, cape_03, lcl_agl, lfc_agl, el_agl


def _interp_at_pressure_flat(
    pressure_levels_hpa: np.ndarray,
    values: np.ndarray,
    target_pressure_hpa: np.ndarray,
) -> np.ndarray:
    """Log-pressure interpolation for one target pressure at every grid point."""
    p = np.asarray(pressure_levels_hpa, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_pressure_hpa, dtype=float)
    npoints, nlev = values.shape

    count_higher = np.sum(p[None, :] > target[:, None], axis=1)
    upper_idx = np.clip(count_higher - 1, 0, nlev - 1)
    lower_idx = np.clip(count_higher, 0, nlev - 1)
    rows = np.arange(npoints)

    p_hi = p[upper_idx]
    p_lo = p[lower_idx]
    v_hi = values[rows, upper_idx]
    v_lo = values[rows, lower_idx]
    denom = np.log(np.maximum(p_lo, 1.0) / np.maximum(p_hi, 1.0))
    weight = np.where(
        np.abs(denom) > 1e-10,
        np.log(np.maximum(target, 1.0) / np.maximum(p_hi, 1.0)) / denom,
        0.0,
    )
    result = v_hi + weight * (v_lo - v_hi)
    inside = (target <= p.max() + 0.1) & (target >= p.min() - 0.1)
    return np.where(inside & np.isfinite(v_hi) & np.isfinite(v_lo), result, np.nan)


def _thermodynamics_np(
    pressure_levels: np.ndarray,
    environment_temperature_k: np.ndarray,
    environment_q: np.ndarray,
    geopotential_height_m: np.ndarray,
    surface_pressure_hpa: np.ndarray,
    temperature_2m_k: np.ndarray,
    dewpoint_2m_k: np.ndarray,
    terrain_m: np.ndarray,
) -> tuple[np.ndarray, ...]:
    pressure = np.asarray(pressure_levels, dtype=float)
    if np.nanmax(pressure) > 2000.0:
        pressure = pressure / 100.0

    order = np.argsort(pressure)[::-1]
    pressure = pressure[order]
    t_env = np.asarray(environment_temperature_k, dtype=float)[..., order]
    q_env = np.asarray(environment_q, dtype=float)[..., order]
    z_env = np.asarray(geopotential_height_m, dtype=float)[..., order]

    horizontal_shape = t_env.shape[:-1]
    nlev = t_env.shape[-1]
    t_env = t_env.reshape(-1, nlev)
    q_env = q_env.reshape(-1, nlev)
    z_env = z_env.reshape(-1, nlev)
    psfc = np.asarray(surface_pressure_hpa, dtype=float).reshape(-1)
    t2m = np.asarray(temperature_2m_k, dtype=float).reshape(-1)
    td2m = np.asarray(dewpoint_2m_k, dtype=float).reshape(-1)
    terrain = np.asarray(terrain_m, dtype=float).reshape(-1)

    psfc = np.where(psfc > 2000.0, psfc / 100.0, psfc)
    td2m = np.minimum(td2m, t2m)
    q_env = np.where((q_env >= 0.0) & (q_env < 0.08), q_env, np.nan)

    p2d = np.broadcast_to(pressure[None, :], t_env.shape)
    above_ground = p2d <= psfc[:, None] + 1.0
    q_surface = _q_from_dewpoint_np(td2m, psfc)

    # Surface-based parcel.
    sb = _parcel_diagnostics_flat(
        pressure, t_env, q_env, z_env, terrain, psfc, t2m, td2m, np.zeros_like(psfc)
    )

    # 100-hPa mixed-layer parcel. Average potential temperature and moisture in
    # pressure space and include the observed/model 2-m parcel as the bottom sample.
    theta_env = t_env * (1000.0 / p2d) ** KAPPA
    theta_surface = t2m * (1000.0 / psfc) ** KAPPA
    ml_mask = above_ground & (p2d >= psfc[:, None] - 100.0) & np.isfinite(theta_env) & np.isfinite(q_env)
    ml_count = ml_mask.sum(axis=1).astype(float)
    theta_sum = np.nansum(np.where(ml_mask, theta_env, 0.0), axis=1) + theta_surface
    q_sum = np.nansum(np.where(ml_mask, q_env, 0.0), axis=1) + q_surface
    theta_ml = theta_sum / np.maximum(ml_count + 1.0, 1.0)
    q_ml = q_sum / np.maximum(ml_count + 1.0, 1.0)
    t_ml = theta_ml * (psfc / 1000.0) ** KAPPA
    td_ml = _dewpoint_from_q_np(q_ml, psfc)
    ml = _parcel_diagnostics_flat(
        pressure, t_env, q_env, z_env, terrain, psfc, t_ml, td_ml, np.zeros_like(psfc)
    )

    # Most-unstable parcel from the lowest 300 hPa, selected by Bolton theta-e.
    td_env = _dewpoint_from_q_np(q_env, p2d)
    thetae_env = _theta_e_np(t_env, td_env, p2d, q_env)
    mu_mask = above_ground & (p2d >= psfc[:, None] - 300.0) & np.isfinite(thetae_env)
    score_env = np.where(mu_mask, thetae_env, -np.inf)
    env_idx = np.argmax(score_env, axis=1)
    rows = np.arange(t_env.shape[0])
    env_score = score_env[rows, env_idx]
    thetae_surface = _theta_e_np(t2m, td2m, psfc, q_surface)
    use_surface = thetae_surface >= env_score

    p_mu = np.where(use_surface, psfc, pressure[env_idx])
    t_mu = np.where(use_surface, t2m, t_env[rows, env_idx])
    td_mu = np.where(use_surface, td2m, td_env[rows, env_idx])
    z_mu = np.where(use_surface, 0.0, z_env[rows, env_idx] - terrain)
    mu = _parcel_diagnostics_flat(
        pressure, t_env, q_env, z_env, terrain, p_mu, t_mu, td_mu, z_mu
    )

    # Sample the formal effective-inflow CAPE/CIN thresholds every 50 hPa from
    # the surface through 300 hPa depth. This is much closer to the operational
    # effective-layer concept than the V2 surface-inflow proxy while remaining
    # computationally tractable on the national GFS grid.
    effective_bottom = np.full_like(psfc, np.nan)
    effective_top = np.full_like(psfc, np.nan)
    effective_cape = np.full_like(psfc, np.nan)
    started = np.zeros(psfc.shape, dtype=bool)
    ended = np.zeros(psfc.shape, dtype=bool)

    candidate_offsets = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    for offset in candidate_offsets:
        if offset == 0.0:
            cand_p = psfc
            cand_t = t2m
            cand_td = td2m
            cand_z = np.zeros_like(psfc)
            cand_cape, cand_cin = sb[0], sb[1]
            valid_candidate = np.isfinite(cand_cape) & np.isfinite(cand_cin)
        else:
            cand_p = psfc - offset
            cand_t = _interp_at_pressure_flat(pressure, t_env, cand_p)
            cand_q = _interp_at_pressure_flat(pressure, q_env, cand_p)
            cand_z_abs = _interp_at_pressure_flat(pressure, z_env, cand_p)
            cand_td = _dewpoint_from_q_np(cand_q, cand_p)
            cand_z = cand_z_abs - terrain
            valid_candidate = (
                np.isfinite(cand_t)
                & np.isfinite(cand_q)
                & np.isfinite(cand_z)
                & (cand_p >= pressure.min())
                & (cand_z >= -100.0)
            )
            cand_diag = _parcel_diagnostics_flat(
                pressure,
                t_env,
                q_env,
                z_env,
                terrain,
                cand_p,
                cand_t,
                cand_td,
                cand_z,
            )
            cand_cape, cand_cin = cand_diag[0], cand_diag[1]

        qualifies = valid_candidate & (cand_cape >= 100.0) & (cand_cin >= -250.0)
        new_start = (~started) & (~ended) & qualifies
        effective_bottom[new_start] = cand_z[new_start]
        effective_top[new_start] = cand_z[new_start]
        effective_cape[new_start] = cand_cape[new_start]
        started |= new_start

        continuing = started & (~ended) & qualifies
        effective_top[continuing] = cand_z[continuing]
        effective_cape[continuing] = np.fmax(effective_cape[continuing], cand_cape[continuing])

        fail_after_start = started & (~ended) & valid_candidate & (~qualifies)
        ended |= fail_after_start

    effective_depth = effective_top - effective_bottom

    raw_outputs = (
        sb[0], sb[1], ml[0], ml[1], mu[0], mu[1],
        sb[2], ml[2], mu[2],
        sb[3], ml[3], mu[3],
        sb[4], ml[4], mu[4],
        sb[5], ml[5], mu[5],
        effective_bottom, effective_top, effective_depth, effective_cape, p_mu,
    )
    return tuple(np.asarray(value).reshape(horizontal_shape) for value in raw_outputs)


def calculate_gfs_thermodynamics_v3(ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct parcel thermodynamics directly from the GFS vertical profile.

    The calculation uses pressure-level T/q/geopotential plus 2-m T/Td and surface
    pressure. Parcel ascent is dry adiabatic to the Bolton LCL and then integrated
    pseudoadiabatically in log-pressure coordinates. CAPE/CIN are integrated from
    virtual-temperature buoyancy on the model height profile.
    """
    required = ("air_temperature", "geopotential_height", "temperature_2m", "dewpoint_2m")
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"GFS V3 thermodynamics missing fields: {', '.join(missing)}")

    temperature = ds["air_temperature"]
    height = ds["geopotential_height"]
    dim = _profile_dim(temperature)
    if _profile_dim(height) != dim:
        raise ValueError("Temperature and geopotential profiles use different pressure dimensions")

    q = _specific_humidity_profile(ds, temperature)
    t2m = _collapse_scalar(ds["temperature_2m"])
    td2m = _collapse_scalar(ds["dewpoint_2m"])
    terrain = _terrain(ds, t2m)
    psfc = _surface_pressure_hpa(ds, terrain)
    pressure = temperature[dim]

    outputs = xr.apply_ufunc(
        _thermodynamics_np,
        pressure,
        temperature,
        q,
        height,
        psfc,
        t2m,
        td2m,
        terrain,
        input_core_dims=[[dim], [dim], [dim], [dim], [], [], [], []],
        output_core_dims=[[] for _ in OUTPUT_NAMES],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[float for _ in OUTPUT_NAMES],
    )

    result = xr.Dataset({name: value for name, value in zip(OUTPUT_NAMES, outputs, strict=True)})
    for name in ("sbcape_jkg", "mlcape_jkg", "mucape_jkg", "sbcape_0_3km_jkg", "mlcape_0_3km_jkg", "mucape_0_3km_jkg", "effective_cape_jkg"):
        result[name] = xr.where(np.isfinite(result[name]), xr.apply_ufunc(np.maximum, result[name], 0.0), np.nan)
    for name in ("sbcin_jkg", "mlcin_jkg", "mucin_jkg"):
        result[name] = xr.where(np.isfinite(result[name]), xr.apply_ufunc(np.minimum, result[name], 0.0), np.nan)

    result.attrs.update(
        source="GFS_0p25_thermodynamics_v3",
        method=(
            "profile-reconstructed SB/ML/MU parcels; dry ascent to Bolton LCL; "
            "pseudoadiabatic log-pressure integration; virtual-temperature CAPE/CIN"
        ),
        mixed_layer_depth_hpa=100,
        most_unstable_search_depth_hpa=300,
        effective_inflow_thresholds="CAPE >= 100 J/kg and CIN >= -250 J/kg, sampled every 50 hPa",
        native_cape_cin_used=False,
    )
    return result
