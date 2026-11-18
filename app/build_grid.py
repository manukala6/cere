"""Takes in base analysis layer and outputs a gridded summary zarr
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (registers the `.rio` accessor used below)
import shapely
from geocube.api.core import make_geocube

# Mean Earth radius (m), used for the pixel-area calculation.
EARTH_RADIUS_M = 6_371_008.8

def add_pixel_area(ds):
    """Add per-row pixel area (ha) as a `pixel_area_ha` coordinate.

    Area depends only on latitude: each cell spans `dx` in longitude and `dy`
    in latitude, so its area is R^2 * dx_rad * (sin(lat+half) - sin(lat-half)).
    """
    dx = float(ds.x[1] - ds.x[0])
    dy = abs(float(ds.y[1] - ds.y[0]))
    lat = np.radians(ds.y.values)
    half = np.radians(dy) / 2
    area_ha = (
        EARTH_RADIUS_M**2 * np.radians(dx) * (np.sin(lat + half) - np.sin(lat - half)) / 1e4
    )
    return ds.assign_coords(pixel_area_ha=("y", area_ha.astype("float32")))

def build_grid(boundary_gdf, res, crs):
    """Build an empty xarray grid at `res` degrees covering the boundary bbox.

    Bounds are snapped outward to a clean `res` lattice; coordinates are cell
    centers with y descending (to match typical raster orientation).
    """
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    decimals = int(np.ceil(-np.log10(res)))
    minx = np.round(np.floor(minx / res) * res, decimals)
    miny = np.round(np.floor(miny / res) * res, decimals)
    maxx = np.round(np.ceil(maxx / res) * res, decimals)
    maxy = np.round(np.ceil(maxy / res) * res, decimals)

    nx = int(round((maxx - minx) / res))
    ny = int(round((maxy - miny) / res))

    x = minx + (np.arange(nx) + 0.5) * res
    y = maxy - (np.arange(ny) + 0.5) * res

    grid = xr.Dataset(coords={"y": ("y", y), "x": ("x", x)}).rio.write_crs(crs)
    bounds = (minx, miny, maxx, maxy)
    return grid, bounds, (nx, ny), (x, y)


def rasterize_mask(boundary_gdf, grid):
    """Rasterize the boundary onto `grid` -> (y, x) array, 1 inside / 0 outside."""
    boundary_gdf = boundary_gdf.to_crs(grid.rio.crs)
    boundary_gdf = boundary_gdf.assign(in_bound=1)
    mask = make_geocube(
        vector_data=boundary_gdf[["in_bound", "geometry"]],
        measurements=["in_bound"],
        like=grid,
        fill=0,
    )
    return mask["in_bound"]

def build_grid_geojson(shape, coords, res, mask, crs):
    """Build a fishnet GeoDataFrame of grid cells (one polygon per in-bound cell).

    Each feature carries its zarr indices (iy, ix) and cell center (x, y) so a
    dashboard can map a clicked cell back to the zarr via `isel(y=iy, x=ix)`.
    Only cells inside the boundary are emitted, so there are no dead clicks.
    """
    nx, ny = shape
    x, y = coords
    xx, yy = np.meshgrid(x, y)                                # (ny, nx) cell centers
    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    half = res / 2
    polys = shapely.box(xx - half, yy - half, xx + half, yy + half)

    inside = mask.values.astype(bool)                         # keep only in-boundary cells
    gdf = gpd.GeoDataFrame(
        {
            "iy": iy_idx[inside],
            "ix": ix_idx[inside],
            "x": xx[inside],
            "y": yy[inside],
        },
        geometry=polys[inside],
        crs=crs,
    )
    return gdf

# TODO summarize base data

# TODO add prefect tasks/flow