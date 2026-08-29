from __future__ import annotations

import numpy as np
from scipy import ndimage


def coherent_mask(
    mask: np.ndarray,
    *,
    closing_iterations: int = 1,
    min_component_cells: int = 3,
) -> np.ndarray:
    """Round/bridge tiny gaps and remove visual one-pixel risk islands.

    This is a presentation/polygonization filter only. It never changes the raw
    meteorological probability fields stored in NetCDF.
    """
    work = np.asarray(mask, dtype=bool)
    if not work.any():
        return work

    structure = ndimage.generate_binary_structure(2, 1)
    if closing_iterations > 0:
        work = ndimage.binary_closing(
            work,
            structure=structure,
            iterations=int(closing_iterations),
            border_value=0,
        )

    if min_component_cells > 1:
        labels, count = ndimage.label(work, structure=structure)
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= int(min_component_cells)
            keep[0] = False
            work = keep[labels]

    return work
