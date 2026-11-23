import numpy as np
import xarray as xr

from build_grid import Bounds, GridSpec
from checks import FINE_BASE, FINE_GRID, FineBase, overlaps, validate

EARTH_RADIUS_M = 6_371_008.8


def prepare_base(base: xr.DataArray, grid: GridSpec) -> FineBase:
    """Clip the fine source to `grid`, attach pixel area, and validate it for summarize_landcover.

    Validates at each handoff so a failure names its real cause:
    the source's layout and overlap before clipping, the clip itself (>= 2 pixels
    each way, which add_pixel_area needs), and the finished base against FINE_BASE.
    """
    validate(base, FINE_GRID + [overlaps(grid.bounds)], "source")
    base = clip_base(base, grid.bounds)
    validate(base, FINE_GRID, "clipped source")
    base = add_pixel_area(base)
    validate(base, FINE_BASE, "prepared base")
    return FineBase(base)

def clip_base(base: xr.DataArray, bbox: Bounds) -> xr.DataArray:
    """ Clip base analysis layer to study area"""
    minx, miny, maxx, maxy = bbox
    return base.sel(y=slice(maxy, miny), x=slice(minx, maxx))


def add_pixel_area(base: xr.DataArray) -> xr.DataArray:
    """Add per-row pixel area (ha) as a `pixel_area_ha` coordinate.

    Area depends only on latitude: each cell spans `dx` in longitude and `dy`
    in latitude, so its area is R^2 * dx_rad * (sin(lat+half) - sin(lat-half)).
    """
    dx = float(base.x[1] - base.x[0])
    dy = abs(float(base.y[1] - base.y[0]))
    lat = np.radians(base.y.values)
    half = np.radians(dy) / 2
    area_ha = (
        EARTH_RADIUS_M**2 * np.radians(dx) * (np.sin(lat + half) - np.sin(lat - half)) / 1e4
    )
    return base.assign_coords(pixel_area_ha=("y", area_ha.astype("float32")))