from __future__ import annotations

import numpy as np
import xarray as xr


CATEGORY_ORDER = ["NONE", "TSTM", "MRGL", "SLGT", "ENH", "MDT", "HIGH"]
CATEGORY_CODE = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}


def categorical_outlook(
    severe_any_probability: xr.DataArray,
    thunderstorm_probability: xr.DataArray,
    config: dict,
) -> xr.DataArray:
    """Create a categorical risk grid from calibrated probability fields.

    Hazard-specific overrides can be layered on later; this first implementation
    keeps categorical thresholds centralized in YAML rather than scattered as
    magic numbers in code.
    """

    severe, thunder = xr.align(severe_any_probability, thunderstorm_probability)
    result = xr.zeros_like(severe, dtype=np.int8)

    tstm_cfg = config["categorical"]["TSTM"]
    result = xr.where(
        thunder >= float(tstm_cfg["thunderstorm_probability_min"]),
        CATEGORY_CODE["TSTM"],
        result,
    )

    for category in ["MRGL", "SLGT", "ENH", "MDT", "HIGH"]:
        threshold = float(config["categorical"][category]["severe_any_probability_min"])
        result = xr.where(severe >= threshold, CATEGORY_CODE[category], result)

    result.name = "categorical_risk"
    result.attrs["category_order"] = CATEGORY_ORDER
    result.attrs["category_code"] = CATEGORY_CODE
    return result


def category_name(code: int) -> str:
    try:
        return CATEGORY_ORDER[int(code)]
    except (IndexError, ValueError):
        return "NONE"
