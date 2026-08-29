from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from metpy.calc import (
    cape_cin,
    lcl,
    lfc,
    mixed_layer,
    most_unstable_parcel,
    parcel_profile,
)
from metpy.units import units


@dataclass(slots=True)
class ThermodynamicPoint:
    sbcape_jkg: float
    sbcin_jkg: float
    mlcape_jkg: float
    mlcin_jkg: float
    mucape_jkg: float
    mucin_jkg: float
    lcl_m: float | None
    lfc_hpa: float | None


def _finite_profile(pressure_hpa, temperature_c, dewpoint_c):
    p = np.asarray(pressure_hpa, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    td = np.asarray(dewpoint_c, dtype=float)
    mask = np.isfinite(p) & np.isfinite(t) & np.isfinite(td)
    p, t, td = p[mask], t[mask], td[mask]
    if p.size < 4:
        raise ValueError("At least four valid vertical levels are required")
    order = np.argsort(p)[::-1]
    return p[order] * units.hPa, t[order] * units.degC, td[order] * units.degC


def calculate_point_thermodynamics(
    pressure_hpa,
    temperature_c,
    dewpoint_c,
    mixed_layer_depth_hpa: float = 100.0,
) -> ThermodynamicPoint:
    """Calculate core parcel diagnostics for a single vertical profile.

    This intentionally works at point/sounding level. Grid-wide execution should
    be wrapped with xarray/dask so chunks can be processed lazily rather than
    loading a continental 4-D field into memory.
    """

    p, t, td = _finite_profile(pressure_hpa, temperature_c, dewpoint_c)

    sb_profile = parcel_profile(p, t[0], td[0])
    sbcape, sbcin = cape_cin(p, t, td, sb_profile)

    ml_p, ml_t, ml_td = mixed_layer(
        p,
        t,
        td,
        depth=mixed_layer_depth_hpa * units.hPa,
    )
    ml_profile = parcel_profile(p, ml_t, ml_td)
    mlcape, mlcin = cape_cin(p, t, td, ml_profile)

    mu_p, mu_t, mu_td, _ = most_unstable_parcel(p, t, td, depth=300 * units.hPa)
    mu_profile = parcel_profile(p[p <= mu_p], mu_t, mu_td)
    mucape, mucin = cape_cin(p[p <= mu_p], t[p <= mu_p], td[p <= mu_p], mu_profile)

    lcl_p, lcl_t = lcl(p[0], t[0], td[0])
    lfc_p, _ = lfc(p, t, td, parcel_temperature_profile=sb_profile)

    # Hypsometric conversion is handled elsewhere when model geopotential is
    # available. This rough pressure-derived height is sufficient only as a
    # diagnostic placeholder for the MVP.
    lcl_m = 44330.0 * (1.0 - (float(lcl_p.m) / float(p[0].m)) ** 0.1903)

    return ThermodynamicPoint(
        sbcape_jkg=float(sbcape.to("joule / kilogram").m),
        sbcin_jkg=float(sbcin.to("joule / kilogram").m),
        mlcape_jkg=float(mlcape.to("joule / kilogram").m),
        mlcin_jkg=float(mlcin.to("joule / kilogram").m),
        mucape_jkg=float(mucape.to("joule / kilogram").m),
        mucin_jkg=float(mucin.to("joule / kilogram").m),
        lcl_m=lcl_m,
        lfc_hpa=None if np.ma.is_masked(lfc_p.m) else float(lfc_p.to("hPa").m),
    )
