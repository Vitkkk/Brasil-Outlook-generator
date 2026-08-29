from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from metpy.calc import bunkers_storm_motion, bulk_shear, storm_relative_helicity
from metpy.units import units


@dataclass(slots=True)
class KinematicPoint:
    shear_0_1km_ms: float
    shear_0_3km_ms: float
    shear_0_6km_ms: float
    srh_0_500m_m2s2: float
    srh_0_1km_m2s2: float
    srh_0_3km_m2s2: float
    bunkers_right_u_ms: float
    bunkers_right_v_ms: float
    bunkers_left_u_ms: float
    bunkers_left_v_ms: float


def _mag(u, v) -> float:
    return float(np.hypot(u.to("m/s").m, v.to("m/s").m))


def calculate_point_kinematics(
    pressure_hpa,
    height_m,
    u_ms,
    v_ms,
) -> KinematicPoint:
    """Calculate common shear/SRH diagnostics from one vertical wind profile."""

    p = np.asarray(pressure_hpa, dtype=float)
    z = np.asarray(height_m, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    mask = np.isfinite(p) & np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
    p, z, u, v = p[mask], z[mask], u[mask], v[mask]
    if p.size < 4:
        raise ValueError("At least four valid wind-profile levels are required")

    order = np.argsort(z)
    p = p[order] * units.hPa
    z = z[order] * units.m
    u = u[order] * units("m/s")
    v = v[order] * units("m/s")

    u1, v1 = bulk_shear(p, u, v, height=z, depth=1 * units.km)
    u3, v3 = bulk_shear(p, u, v, height=z, depth=3 * units.km)
    u6, v6 = bulk_shear(p, u, v, height=z, depth=6 * units.km)

    right, left, _mean = bunkers_storm_motion(p, u, v, z)

    def srh(depth):
        positive, negative, total = storm_relative_helicity(
            z,
            u,
            v,
            depth=depth,
            storm_u=right[0],
            storm_v=right[1],
        )
        return float(total.to("meter ** 2 / second ** 2").m)

    return KinematicPoint(
        shear_0_1km_ms=_mag(u1, v1),
        shear_0_3km_ms=_mag(u3, v3),
        shear_0_6km_ms=_mag(u6, v6),
        srh_0_500m_m2s2=srh(500 * units.m),
        srh_0_1km_m2s2=srh(1 * units.km),
        srh_0_3km_m2s2=srh(3 * units.km),
        bunkers_right_u_ms=float(right[0].to("m/s").m),
        bunkers_right_v_ms=float(right[1].to("m/s").m),
        bunkers_left_u_ms=float(left[0].to("m/s").m),
        bunkers_left_v_ms=float(left[1].to("m/s").m),
    )
