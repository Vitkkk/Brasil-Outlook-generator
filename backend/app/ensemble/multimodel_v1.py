from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import xarray as xr


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
    names = []
    for name, ds in models.items():
        if field not in ds:
            continue
        aligned = _regrid_like(ds[[field]], reference)[field]
        arrays.append(aligned.expand_dims(model=[name]))
        names.append(name)
    if not arrays:
        raise KeyError(f"No model supplied field {field}")
    return xr.concat(arrays, dim="model", join="exact")


def multimodel_consensus(models: Mapping[str, xr.Dataset]) -> tuple[xr.Dataset, ConsensusSummary]:
    """Create a robust deterministic multi-model consensus.

    V1 deliberately avoids fixed model weights. Each model first runs the same
    physical diagnostics. Consensus then uses the median as the stable center,
    retains part of the upper-quartile and maximum signal so strong minority
    solutions are not averaged away, and publishes spread/agreement separately.
    Historical verification can later replace this neutral V1 with skill-based
    regional/seasonal/lead-time weights.
    """
    if len(models) < 2:
        raise ValueError("Multimodel consensus requires at least two models")

    reference = next(iter(models.values()))
    output: dict[str, xr.DataArray] = {}

    for field in HAZARD_FIELDS:
        stack = _stack(models, field, reference)
        median = stack.median("model", skipna=True)
        q75 = stack.quantile(0.75, dim="model", skipna=True).drop_vars("quantile", errors="ignore")
        maximum = stack.max("model", skipna=True)
        minimum = stack.min("model", skipna=True)
        spread = maximum - minimum

        # Robust center + modest preservation of stronger solutions.
        consensus = median + 0.25 * (q75 - median) + 0.10 * (maximum - q75)
        consensus = xr.apply_ufunc(np.clip, consensus, 0.0, 1.0)

        output[field] = consensus
        output[f"{field}_median"] = median
        output[f"{field}_max"] = maximum
        output[f"{field}_min"] = minimum
        output[f"{field}_spread"] = spread

    # Agreement metrics relevant for the categorical product.
    severe_stack = _stack(models, "severe", reference)
    output["agreement_mrgl"] = (severe_stack >= 0.05).mean("model")
    output["agreement_slgt"] = (severe_stack >= 0.15).mean("model")
    output["agreement_enh"] = (severe_stack >= 0.30).mean("model")
    output["agreement_mdt"] = (severe_stack >= 0.45).mean("model")

    # Confidence is deliberately separate from risk. Lower inter-model spread
    # yields greater confidence, but a high-risk/low-confidence outcome remains possible.
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
    output["forecast_confidence"] = xr.apply_ufunc(
        np.clip, 1.0 - normalized_spread, 0.0, 1.0
    )

    ds = xr.Dataset(output)
    ds.attrs.update(
        source="MULTIMODEL_CONSENSUS_V1",
        models=",".join(models.keys()),
        model_count=len(models),
        calibrated=False,
        method="median + upper-quartile signal preservation; no fixed skill weights",
    )
    return ds, ConsensusSummary(
        models=list(models.keys()),
        model_count=len(models),
        method=ds.attrs["method"],
    )
