from __future__ import annotations

import numpy as np
import xarray as xr


def _clip01(value: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.clip, value, 0.0, 1.0)


def _norm(value: xr.DataArray, low: float, high: float) -> xr.DataArray:
    return _clip01((value - low) / max(high - low, 1e-6))


def _sigmoid(value: xr.DataArray, midpoint: float = 0.5, steepness: float = 7.0) -> xr.DataArray:
    return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))


def _select_pressure(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    for dim in ("isobaricInhPa", "isobaricInPa"):
        if dim in da.dims:
            level = level_hpa if dim == "isobaricInhPa" else level_hpa * 100.0
            return da.sel({dim: level}, method="nearest")
    raise KeyError(f"No pressure-level dimension found for {da.name}")


def _wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.hypot(u, v)


def _deep_layer_shear_proxy(ds: xr.Dataset) -> xr.DataArray:
    """Approximate deep-layer shear using 10 m to 500-hPa vector difference.

    It is intentionally named a proxy: production diagnostics will replace this
    with height-interpolated 0-6 km/effective bulk shear from reconstructed
    vertical profiles.
    """
    u500 = _select_pressure(ds["eastward_wind"], 500)
    v500 = _select_pressure(ds["northward_wind"], 500)
    return np.hypot(u500 - ds["u_wind_10m"], v500 - ds["v_wind_10m"])


def _midlevel_lapse_proxy(ds: xr.Dataset) -> xr.DataArray:
    t700 = _select_pressure(ds["air_temperature"], 700)
    t500 = _select_pressure(ds["air_temperature"], 500)
    z700 = _select_pressure(ds["geopotential_height"], 700)
    z500 = _select_pressure(ds["geopotential_height"], 500)
    dz_km = xr.where(np.abs(z500 - z700) > 100.0, np.abs(z500 - z700) / 1000.0, np.nan)
    return (t700 - t500) / dz_km


def _field_or(ds: xr.Dataset, name: str, fallback: xr.DataArray) -> xr.DataArray:
    return ds[name] if name in ds else xr.zeros_like(fallback)


def gfs_proxy_probabilities(ds: xr.Dataset) -> xr.Dataset:
    """Create the first GFS-only severe-weather probability proxies.

    These are NOT calibrated climatological probabilities. They are a transparent
    bridge between raw GFS fields and the future calibrated multi-model hazard
    system, useful for smoke-testing real-data ingestion and map generation.
    """
    cape = ds["native_cape"]
    cin = ds["native_cin"]

    # Native CAPE may contain multiple layer types after cfgrib merge. If a layer
    # dimension survived, prefer the largest physically relevant value at each
    # point rather than silently choosing one arbitrary layer.
    for dim in tuple(cape.dims):
        if dim not in {"valid_time", "latitude", "longitude"}:
            try:
                cape = cape.max(dim=dim, skipna=True)
                cin = cin.min(dim=dim, skipna=True) if dim in cin.dims else cin
            except Exception:
                pass

    shear = _deep_layer_shear_proxy(ds)
    lapse = _midlevel_lapse_proxy(ds)
    pwat = ds["precipitable_water"]
    helicity = _field_or(ds, "native_helicity", cape)
    if helicity.name == cape.name:
        helicity = xr.zeros_like(cape)
    else:
        for dim in tuple(helicity.dims):
            if dim not in {"valid_time", "latitude", "longitude"}:
                try:
                    helicity = helicity.max(dim=dim, skipna=True)
                except Exception:
                    pass

    refc = _field_or(ds, "composite_reflectivity", cape)
    if refc.name == cape.name:
        refc = xr.zeros_like(cape)
    gust = _field_or(ds, "surface_gust", cape)
    if gust.name == cape.name:
        gust = xr.zeros_like(cape)

    cape_f = _norm(cape, 250.0, 2500.0)
    shear_f = _norm(shear, 10.0, 30.0)
    lapse_f = _norm(lapse, 5.5, 8.0)
    helicity_f = _norm(np.abs(helicity), 50.0, 350.0)
    pwat_f = _norm(pwat, 20.0, 50.0)
    cin_f = 1.0 - _norm(np.abs(cin), 25.0, 200.0)
    convective_signal = xr.where(refc > 20.0, 1.0, 0.0)

    # Initiation/coverage proxy: instability and weak cap matter, but explicit
    # modeled convection receives a large boost. PWAT acts only as a supporting
    # moisture feature rather than a decision threshold.
    initiation_score = (
        0.34 * cape_f
        + 0.22 * cin_f
        + 0.14 * pwat_f
        + 0.30 * convective_signal
    )
    initiation = _clip01(_sigmoid(initiation_score, midpoint=0.42, steepness=6.0))

    supercell_score = 0.50 * shear_f + 0.28 * cape_f + 0.22 * helicity_f
    supercell = _clip01(initiation * _sigmoid(supercell_score, midpoint=0.48, steepness=7.0))

    tornado_score = (
        0.31 * helicity_f
        + 0.26 * shear_f
        + 0.20 * cape_f
        + 0.13 * cin_f
        + 0.10 * pwat_f
    )
    tornado = _clip01(0.18 * supercell * _sigmoid(tornado_score, midpoint=0.52, steepness=8.0))

    hail_score = 0.36 * cape_f + 0.30 * shear_f + 0.24 * lapse_f + 0.10 * supercell
    hail = _clip01(0.55 * initiation * _sigmoid(hail_score, midpoint=0.48, steepness=7.0))

    gust_f = _norm(gust, 18.0, 35.0)
    wind_score = 0.28 * cape_f + 0.24 * shear_f + 0.20 * lapse_f + 0.28 * gust_f
    wind = _clip01(0.55 * initiation * _sigmoid(wind_score, midpoint=0.46, steepness=7.0))

    severe = xr.apply_ufunc(np.maximum, hail, wind)
    severe = xr.apply_ufunc(np.maximum, severe, tornado * 2.2)
    thunder = _clip01(0.10 + 0.90 * initiation)

    result = xr.Dataset(
        {
            "thunderstorm": thunder,
            "severe": severe,
            "tornado": tornado,
            "hail": hail,
            "wind": wind,
            "convective_initiation": initiation,
            "supercell": supercell,
            "deep_shear_proxy_ms": shear,
            "lapse_700_500_c_per_km": lapse,
        }
    )
    result.attrs.update(
        source="GFS_0p25_proxy_v1",
        calibrated=False,
        probability_radius_km=40,
        warning=(
            "First live GFS proxy. Hazard fields are not yet statistically calibrated "
            "and deep-layer shear is approximated by 10 m-to-500 hPa vector difference."
        ),
    )
    return result


def aggregate_day1_max(hourly: list[xr.Dataset]) -> xr.Dataset:
    if not hourly:
        raise ValueError("At least one forecast dataset is required")
    stacked = xr.concat(hourly, dim="forecast_sample")
    products = [
        "thunderstorm",
        "severe",
        "tornado",
        "hail",
        "wind",
        "convective_initiation",
        "supercell",
    ]
    output = xr.Dataset({name: stacked[name].max("forecast_sample", skipna=True) for name in products})
    output.attrs.update(hourly[0].attrs)
    output.attrs["aggregation"] = "maximum across sampled Day-1 forecast hours"
    return output
