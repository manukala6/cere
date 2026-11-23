import numpy as np
import xarray as xr

from build_grid import Bounds

EARTH_RADIUS_M = 6_371_008.8

def clip_base(base: xr.DataArray, bbox: Bounds) -> xr.DataArray:
    """ Clip base analysis layer to study area"""
    minx, miny, maxx, maxy = bbox
    base = base.sel(y=slice(maxy, miny), x=slice(minx, maxx))
    return xr.DataArray()



def add_pixel_area(base: xr.DataArray) -> xr.DataArray:
    """Add per-row pixel area (ha) as a `pixel_area_ha` coordinate.

    Area depends only on latitude: each cell spans `dx` in longitude and `dy`
    in latitude, so its area is R^2 * dx_rad * (sin(lat+half) - sin(lat-half)).
    """
    dx = float(base.x[1] - base.x[0])
    dy = abs(float(base))
    lat = np.radians(base.y.values)
    half = np.radians(dy) / 2
    area_ha = (
        EARTH_RADIUS_M**2 * np.radians(dx) * (np.sin(lat + half) - np.sin(lat - half)) / 1e4
    )
    return base.assign_coords(pixel_area_ha=("y", area_ha.astype("float32")))