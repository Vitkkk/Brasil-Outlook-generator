from __future__ import annotations

import xarray as xr

from app.hazards.gfs_diagnostics_v3 import gfs_diagnostics_v3_probabilities


def model_diagnostics_v3_probabilities(ds: xr.Dataset) -> xr.Dataset:
    """Run the V3 diagnostics on any standardized deterministic model dataset.

    The legacy V2 kinematics helper expects native CAPE/CIN fields even though
    V3 no longer uses them in its hazard equations. For models that do not
    publish those native grids in the open feed, harmless zero placeholders are
    injected only to satisfy the kinematics helper. Thermodynamic instability is
    still reconstructed from the vertical sounding.
    """
    work = ds.copy()
    if "temperature_2m" not in work:
        raise KeyError("standardized model dataset requires temperature_2m")
    like = work["temperature_2m"]
    if "native_cape" not in work:
        work["native_cape"] = xr.zeros_like(like)
    if "native_cin" not in work:
        work["native_cin"] = xr.zeros_like(like)
    result = gfs_diagnostics_v3_probabilities(work)
    result.attrs["diagnostic_model"] = str(ds.attrs.get("model", "UNKNOWN"))
    return result
