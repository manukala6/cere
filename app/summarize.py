"""Aggregate a fine land-cover raster onto a coarse GridSpec.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from flox.xarray import xarray_reduce

from build_grid import GridSpec


def summarize_landcover(
        base: xr.DataArray,
        grid: GridSpec,
        mask: xr.DataArray,
        #rule: ClassRule
) -> xr.DataArray:
    """Sum fine pixels into the coarse `grid` cells, then mask cells outside the boundary.

    `base` must already be prepared by prep_base: clipped to `grid.bounds`
    (`clip_base`) and carrying a `pixel_area_ha` coord (`add_pixel_area`).
    `mask` is the coarse (y, x) mask from `rasterize_mask(boundary, grid.to_dataset())`.
    """
    minx, miny, maxx, maxy = grid.bounds

    # Probability-weighted tree area per fine pixel (ha): P(tree) * area.
    ## TODO: THIS IS HARDCODED FOR TTC, REMOVE LATER
    tree_area = (base / 100.0) * base.pixel_area_ha

    # integer coarse-cell index for each fine pixel. Named so flox can group by them.
    ix = xr.DataArray(
        np.floor((base.x.values - minx) / grid.res).astype("int64"),
        dims="x", coords={"x": base.x}, name="ix",
    )
    iy = xr.DataArray(
        np.floor((maxy - base.y.values) / grid.res).astype("int64"),
        dims="y", coords={"y": base.y}, name="iy",
    )

    # Two-step reduction (one dim at a time) avoids building a full grid in memory.
    tmp = xarray_reduce(
        tree_area, ix, func="sum", fill_value=0, ## TODO: REPLACE TREE_AREA WITH BASE
        expected_groups=np.arange(grid.nx), dim="x",
    )
    analysis_grid = xarray_reduce(
        tmp, iy, func="sum", fill_value=0,      ## Allow for other aggregations other than sum
        expected_groups=np.arange(grid.ny), dim="y",
    )

    analysis_grid = (
        analysis_grid
        .rename(iy="y", ix="x")
        .assign_coords(y=grid.y, x=grid.x)
        .rename("analysis_grid")
    )

    # make_geocube's coords can differ from the computed grid centers by tiny
    # float rounding; .where would then inner-join and silently drop cells.
    # Force the mask onto ttc_ha_gridded's exact coords before masking.
    mask = mask.assign_coords(y=analysis_grid.y, x=analysis_grid.x)
    analysis_grid = analysis_grid.where(mask)

    # Integer 
    return analysis_grid