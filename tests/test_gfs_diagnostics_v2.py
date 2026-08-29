from __future__ import annotations

import numpy as np
import xarray as xr

from app.hazards.gfs_diagnostics_v2 import gfs_diagnostics_v2_probabilities


def _dataset() -> xr.Dataset:
    levels = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 250.0])
    lat = np.array([-27.5])
    lon = np.array([-53.0, -47.0])

    heights = np.array([120.0, 760.0, 1500.0, 3050.0, 5650.0, 9250.0, 10400.0])
    temperatures = np.array([298.0, 294.0, 289.0, 278.0, 260.0, 233.0, 222.0])

    z = np.broadcast_to(heights[:, None, None], (levels.size, lat.size, lon.size)).copy()
    t = np.broadcast_to(temperatures[:, None, None], z.shape).copy()

    # First longitude: strongly sheared/curved severe environment.
    u_strong = np.array([5.0, 8.0, 12.0, 18.0, 28.0, 38.0, 42.0])
    v_strong = np.array([6.0, 9.0, 13.0, 17.0, 20.0, 23.0, 24.0])
    # Second longitude: weakly sheared environment.
    u_weak = np.array([5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 11.0])
    v_weak = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 7.5])

    u = np.stack([u_strong, u_weak], axis=-1)[:, None, :]
    v = np.stack([v_strong, v_weak], axis=-1)[:, None, :]

    omega = np.broadcast_to(np.array([-0.05, -0.08, -0.12, -0.25, -0.15, -0.05, 0.0])[:, None, None], z.shape).copy()

    return xr.Dataset(
        {
            "air_temperature": (("isobaricInhPa", "latitude", "longitude"), t),
            "eastward_wind": (("isobaricInhPa", "latitude", "longitude"), u),
            "northward_wind": (("isobaricInhPa", "latitude", "longitude"), v),
            "geopotential_height": (("isobaricInhPa", "latitude", "longitude"), z),
            "lagrangian_tendency_of_air_pressure": (("isobaricInhPa", "latitude", "longitude"), omega),
            "native_cape": (("latitude", "longitude"), np.array([[2600.0, 450.0]])),
            "native_cin": (("latitude", "longitude"), np.array([[-35.0, -180.0]])),
            "temperature_2m": (("latitude", "longitude"), np.array([[300.0, 294.0]])),
            "dewpoint_2m": (("latitude", "longitude"), np.array([[294.0, 282.0]])),
            "u_wind_10m": (("latitude", "longitude"), np.array([[4.0, 4.0]])),
            "v_wind_10m": (("latitude", "longitude"), np.array([[5.0, 3.5]])),
            "terrain_height": (("latitude", "longitude"), np.array([[100.0, 100.0]])),
            "storm_motion_u": (("latitude", "longitude"), np.array([[14.0, 7.0]])),
            "storm_motion_v": (("latitude", "longitude"), np.array([[7.0, 5.0]])),
            "precipitable_water": (("latitude", "longitude"), np.array([[38.0, 24.0]])),
            "surface_gust": (("latitude", "longitude"), np.array([[28.0, 14.0]])),
            "composite_reflectivity": (("latitude", "longitude"), np.array([[42.0, 5.0]])),
        },
        coords={"isobaricInhPa": levels, "latitude": lat, "longitude": lon},
    )


def test_v2_diagnostics_are_finite_and_discriminate_environment():
    fields = gfs_diagnostics_v2_probabilities(_dataset())

    for name in (
        "tornado",
        "hail",
        "wind",
        "severe",
        "supercell",
        "shear_0_1km_ms",
        "shear_0_6km_ms",
        "lcl_height_proxy_m",
        "freezing_level_agl_m",
    ):
        assert np.isfinite(np.asarray(fields[name])).all(), name

    strong = dict(latitude=-27.5, longitude=-53.0)
    weak = dict(latitude=-27.5, longitude=-47.0)

    assert float(fields["shear_0_6km_ms"].sel(**strong)) > float(fields["shear_0_6km_ms"].sel(**weak))
    assert float(fields["supercell"].sel(**strong)) > float(fields["supercell"].sel(**weak))
    assert float(fields["hail"].sel(**strong)) > float(fields["hail"].sel(**weak))
    assert float(fields["tornado"].sel(**strong)) > float(fields["tornado"].sel(**weak))
