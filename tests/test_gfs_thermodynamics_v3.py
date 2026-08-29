from __future__ import annotations

import numpy as np
import xarray as xr

from app.diagnostics.thermodynamics_v3 import calculate_gfs_thermodynamics_v3
from app.hazards.gfs_diagnostics_v3 import gfs_diagnostics_v3_probabilities


def _dataset() -> xr.Dataset:
    levels = np.array(
        [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0,
         650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0]
    )
    lat = np.array([-27.5])
    lon = np.array([-53.0, -47.0])

    heights = np.array(
        [120.0, 330.0, 550.0, 780.0, 1020.0, 1510.0, 2030.0, 2580.0, 3170.0,
         3800.0, 4470.0, 5180.0, 5950.0, 6790.0, 7710.0, 8720.0, 9840.0, 11100.0, 12600.0]
    )
    strong_t = np.array(
        [301.0, 299.5, 298.0, 296.5, 295.0, 291.5, 288.0, 284.0, 280.0,
         276.0, 271.5, 267.0, 262.0, 256.5, 250.5, 244.0, 236.5, 228.0, 218.0]
    )
    weak_t = np.array(
        [294.0, 292.8, 291.5, 290.2, 289.0, 286.5, 284.0, 281.0, 278.0,
         274.5, 270.5, 266.0, 261.0, 255.5, 249.5, 243.0, 235.5, 227.0, 217.0]
    )
    strong_q = np.array(
        [0.0175, 0.0170, 0.0163, 0.0155, 0.0147, 0.0128, 0.0105, 0.0083, 0.0064,
         0.0049, 0.0038, 0.0030, 0.0023, 0.0017, 0.0012, 0.0008, 0.0005, 0.0003, 0.00015]
    )
    weak_q = np.array(
        [0.0090, 0.0085, 0.0080, 0.0074, 0.0068, 0.0058, 0.0048, 0.0040, 0.0033,
         0.0028, 0.0023, 0.0019, 0.0015, 0.0011, 0.0008, 0.0006, 0.0004, 0.00025, 0.00012]
    )

    z = np.broadcast_to(heights[:, None, None], (levels.size, lat.size, lon.size)).copy()
    t = np.stack([strong_t, weak_t], axis=-1)[:, None, :]
    q = np.stack([strong_q, weak_q], axis=-1)[:, None, :]

    u_strong = np.linspace(5.0, 42.0, levels.size)
    v_strong = np.linspace(5.0, 25.0, levels.size)
    u_weak = np.linspace(4.0, 11.0, levels.size)
    v_weak = np.linspace(3.0, 8.0, levels.size)
    u = np.stack([u_strong, u_weak], axis=-1)[:, None, :]
    v = np.stack([v_strong, v_weak], axis=-1)[:, None, :]
    omega = np.broadcast_to(np.linspace(-0.08, 0.0, levels.size)[:, None, None], z.shape).copy()

    return xr.Dataset(
        {
            "air_temperature": (("isobaricInhPa", "latitude", "longitude"), t),
            "specific_humidity": (("isobaricInhPa", "latitude", "longitude"), q),
            "eastward_wind": (("isobaricInhPa", "latitude", "longitude"), u),
            "northward_wind": (("isobaricInhPa", "latitude", "longitude"), v),
            "geopotential_height": (("isobaricInhPa", "latitude", "longitude"), z),
            "lagrangian_tendency_of_air_pressure": (("isobaricInhPa", "latitude", "longitude"), omega),
            "surface_air_pressure": (("latitude", "longitude"), np.array([[99500.0, 99500.0]])),
            "temperature_2m": (("latitude", "longitude"), np.array([[303.0, 294.0]])),
            "dewpoint_2m": (("latitude", "longitude"), np.array([[295.0, 281.0]])),
            "u_wind_10m": (("latitude", "longitude"), np.array([[4.0, 4.0]])),
            "v_wind_10m": (("latitude", "longitude"), np.array([[5.0, 3.0]])),
            "terrain_height": (("latitude", "longitude"), np.array([[100.0, 100.0]])),
            "storm_motion_u": (("latitude", "longitude"), np.array([[14.0, 7.0]])),
            "storm_motion_v": (("latitude", "longitude"), np.array([[7.0, 5.0]])),
            "precipitable_water": (("latitude", "longitude"), np.array([[40.0, 22.0]])),
            "surface_gust": (("latitude", "longitude"), np.array([[29.0, 14.0]])),
            "composite_reflectivity": (("latitude", "longitude"), np.array([[45.0, 5.0]])),
            # Retained only because V3 reuses the independent V2 kinematic engine.
            # The V3 hazard equations do not use these native instability fields.
            "native_cape": (("latitude", "longitude"), np.array([[1000.0, 1000.0]])),
            "native_cin": (("latitude", "longitude"), np.array([[-100.0, -100.0]])),
        },
        coords={"isobaricInhPa": levels, "latitude": lat, "longitude": lon},
    )


def test_v3_reconstructs_parcel_thermodynamics():
    fields = calculate_gfs_thermodynamics_v3(_dataset())
    strong = dict(latitude=-27.5, longitude=-53.0)
    weak = dict(latitude=-27.5, longitude=-47.0)

    for name in (
        "sbcape_jkg", "mlcape_jkg", "mucape_jkg",
        "sbcin_jkg", "mlcin_jkg", "mucin_jkg",
        "mlcape_0_3km_jkg", "ml_lcl_agl_m", "mu_parcel_pressure_hpa",
    ):
        assert np.isfinite(np.asarray(fields[name])).all(), name

    assert float(fields["mlcape_jkg"].sel(**strong)) > float(fields["mlcape_jkg"].sel(**weak))
    assert float(fields["mucape_jkg"].sel(**strong)) > float(fields["mucape_jkg"].sel(**weak))
    assert float(fields["mlcape_0_3km_jkg"].sel(**strong)) >= 0.0
    assert float(fields["ml_lcl_agl_m"].sel(**strong)) < float(fields["ml_lcl_agl_m"].sel(**weak))


def test_v3_hazards_use_reconstructed_instability():
    fields = gfs_diagnostics_v3_probabilities(_dataset())
    strong = dict(latitude=-27.5, longitude=-53.0)
    weak = dict(latitude=-27.5, longitude=-47.0)

    for name in ("tornado", "hail", "wind", "severe", "supercell", "mlcape_jkg", "mucape_jkg"):
        assert np.isfinite(np.asarray(fields[name])).all(), name

    assert fields.attrs["native_cape_cin_used_for_v3_hazards"] is False
    assert float(fields["supercell"].sel(**strong)) > float(fields["supercell"].sel(**weak))
    assert float(fields["hail"].sel(**strong)) > float(fields["hail"].sel(**weak))
    assert float(fields["tornado"].sel(**strong)) > float(fields["tornado"].sel(**weak))
