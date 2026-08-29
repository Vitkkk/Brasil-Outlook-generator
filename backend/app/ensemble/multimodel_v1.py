from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter


HAZARD_FIELDS = (
    "thunderstorm", "severe", "tornado", "hail", "wind",
    "convective_initiation", "supercell", "qlcs", "sig_tornado_support",
)


@dataclass(slots=True)
class ConsensusSummary:
    models: list[str]
    model_count: int
    method: str


def _regrid_like(ds: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    if (
        np.array_equal(np.asarray(ds.latitude), np.asarray(reference.latitude))
        and np.array_equal(np.asarray(ds.longitude), np.asarray(reference.longitude))
    ):
        return ds
    return ds.interp(
        latitude=reference.latitude,
        longitude=reference.longitude,
        method="linear",
    )


def _stack(models: Mapping[str, xr.Dataset], field: str, reference: xr.Dataset) -> xr.DataArray:
    arrays = []
    for name, ds in models.items():
        if field not in ds:
            continue
        aligned = _regrid_like(ds[[field]], reference)[field]
        arrays.append(aligned.expand_dims(model=[name]))
    if not arrays:
        raise KeyError(f"No model supplied field {field}")
    return xr.concat(arrays, dim="model", join="exact")


def _grid_resolution_deg(reference: xr.Dataset) -> float:
    lat = np.asarray(reference.latitude, dtype=float)
    if lat.size < 2:
        return 0.25
    return float(np.nanmedian(np.abs(np.diff(lat))))


def _neighborhood_stack(stack: xr.DataArray, radius_km: float = 75.0) -> xr.DataArray:
    """Return each model's local upper envelope within a mesoscale neighborhood.

    Deterministic global models frequently displace the same severe corridor by
    25-100 km. A strict point-to-point median therefore interprets position
    uncertainty as weak meteorological support. This operation keeps the models
    independent but allows nearby maxima to count as regional agreement.
    """
    resolution = _grid_resolution_deg(stack.to_dataset(name="field"))
    cell_km = max(20.0, resolution * 111.0)
    radius_cells = max(1, int(round(radius_km / cell_km)))
    size = 2 * radius_cells + 1

    values = np.asarray(stack, dtype=float)
    filtered = np.empty_like(values)
    for i in range(values.shape[0]):
        plane = values[i]
        finite = np.isfinite(plane)
        work = np.where(finite, plane, -np.inf)
        local = maximum_filter(work, size=size, mode="nearest")
        local[~np.isfinite(local)] = np.nan
        filtered[i] = local

    return xr.DataArray(
        filtered,
        coords=stack.coords,
        dims=stack.dims,
        attrs=stack.attrs,
    )


def _clip(da: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.clip, da, 0.0, 1.0)


def multimodel_consensus(models: Mapping[str, xr.Dataset]) -> tuple[xr.Dataset, ConsensusSummary]:
    """Create deterministic multi-model consensus with spatial tolerance.

    V2 fixes the principal conservative bias found in the first live case:
    strict point-to-point averaging punished GFS/ECMWF when both forecast the
    same severe corridor but displaced maxima by several grid cells. Each model
    still runs independently. The consensus now combines the native median,
    a 75-km neighborhood median, and a modest upper-signal term. Forecast
    confidence remains separate from risk magnitude.

    No permanent model skill weights are used yet. Historical verification will
    later learn region/season/lead-time/variable-specific weights.
    """
    if len(models) < 2:
        raise ValueError("Multimodel consensus requires at least two models")

    reference = next(iter(models.values()))
    output: dict[str, xr.DataArray] = {}
    raw_stacks: dict[str, xr.DataArray] = {}
    neighborhood_stacks: dict[str, xr.DataArray] = {}

    for field in HAZARD_FIELDS:
        stack = _stack(models, field, reference)
        neighborhood = _neighborhood_stack(stack, radius_km=75.0)
        raw_stacks[field] = stack
        neighborhood_stacks[field] = neighborhood

        median = stack.median("model", skipna=True)
        q75 = stack.quantile(0.75, dim="model", skipna=True).drop_vars("quantile", errors="ignore")
        maximum = stack.max("model", skipna=True)
        minimum = stack.min("model", skipna=True)
        spread = maximum - minimum

        neighborhood_median = neighborhood.median("model", skipna=True)
        neighborhood_q75 = neighborhood.quantile(0.75, dim="model", skipna=True).drop_vars("quantile", errors="ignore")

        # Native signal stays dominant. Neighborhood support resolves modest
        # placement errors without simply painting the maximum everywhere.
        regional = (
            0.55 * median
            + 0.30 * neighborhood_median
            + 0.10 * q75
            + 0.05 * neighborhood_q75
        )
        # Never make the new method weaker than the original native median.
        consensus = _clip(xr.apply_ufunc(np.maximum, median, regional))

        output[field] = consensus
        output[f"{field}_median"] = median
        output[f"{field}_max"] = maximum
        output[f"{field}_min"] = minimum
        output[f"{field}_spread"] = spread
        output[f"{field}_neighborhood_median"] = neighborhood_median

    severe_stack = raw_stacks["severe"]
    severe_neighborhood = neighborhood_stacks["severe"]

    # Native agreement and regional agreement are intentionally separate.
    for label, threshold in (("mrgl", 0.05), ("slgt", 0.15), ("enh", 0.30), ("mdt", 0.45)):
        output[f"agreement_{label}"] = (severe_stack >= threshold).mean("model")
        output[f"regional_agreement_{label}"] = (severe_neighborhood >= threshold).mean("model")

    # Organization boost: if the models regionally agree on severe convection
    # and simultaneously support initiation + supercell organization, preserve
    # the upper-end scenario instead of compressing it into broad SLGT.
    organization = xr.apply_ufunc(
        np.minimum,
        output["convective_initiation"],
        output["supercell"],
    )
    org_factor = _clip((organization - 0.30) / 0.45)
    regional_slgt = output["regional_agreement_slgt"]
    regional_enh = output["regional_agreement_enh"]
    agreement_factor = _clip(0.45 * regional_slgt + 0.55 * regional_enh)
    severe_boost = 0.09 * org_factor * agreement_factor
    output["severe_pre_organization_boost"] = output["severe"]
    output["severe_organization_boost"] = severe_boost
    output["severe"] = _clip(output["severe"] + severe_boost)

    # Confidence is independent of hazard magnitude. Positional disagreement
    # reduces confidence somewhat, but must not automatically erase the hazard.
    severe_spread = output["severe_spread"]
    tornado_spread = output["tornado_spread"]
    hail_spread = output["hail_spread"]
    wind_spread = output["wind_spread"]
    normalized_spread = (
        severe_spread / 0.45
        + tornado_spread / 0.20
        + hail_spread / 0.45
        + wind_spread / 0.45
    ) / 4.0
    regional_support = (
        output["regional_agreement_slgt"] + output["regional_agreement_enh"]
    ) / 2.0
    output["forecast_confidence"] = _clip(
        1.0 - 0.75 * normalized_spread + 0.15 * regional_support
    )

    ds = xr.Dataset(output)
    ds.attrs.update(
        source="MULTIMODEL_CONSENSUS_V2",
        models=",".join(models.keys()),
        model_count=len(models),
        calibrated=False,
        neighborhood_radius_km=75.0,
        method="native median + 75-km regional agreement + organization preservation; no fixed skill weights",
    )
    return ds, ConsensusSummary(
        models=list(models.keys()),
        model_count=len(models),
        method=ds.attrs["method"],
    )
