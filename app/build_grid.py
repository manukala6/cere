"""Takes in base analysis layer and outputs a gridded summary zarr
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (registers the `.rio` accessor used below)
import shapely
from geocube.api.core import make_geocube
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_008.8
Bounds = tuple[float, float, float, float]

@dataclass(frozen=True)
class GridSpec:
    "Creates the summary grid to aggregate the base LULC data"
    bounds: Bounds
    res: float
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        # res > 0
        if self.res <= 0:
            raise ValueError(f"Resolution must be a positive float")

    @property 
    def nx(self) -> int: return round((self.bounds[2] - self.bounds[0]) / self.res)

    @property
    def ny(self) -> int: return round((self.bounds[3] - self.bounds[1]) / self.res)

    @property
    def shape(self) -> tuple[int, int]: return (self.ny, self.nx)

    @property
    def x(self) -> np.ndarray: return self.bounds[0] + (np.arange(self.nx) + 0.5) * self.res

    @property
    def y(self) -> np.ndarray: return self.bounds[3] - (np.arange(self.ny) + 0.5) * self.res

    def to_dataset(self) -> xr.Dataset:
        """Empty (y, x) coords Dataset with CRS — the `like=` template for rasterize_mask."""
        return xr.Dataset(coords={"y": ("y", self.y), "x": ("x", self.x)}).rio.write_crs(self.crs)


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


def build_grid(boundary_gdf, res, crs) -> GridSpec:
    """Build an empty xarray grid at `res` degrees covering the boundary bbox.

    Bounds are snapped outward to a clean `res` lattice; coordinates are cell
    centers with y descending (to match typical raster orientation).
    """
    minx, miny, maxx, maxy = boundary_gdf.total_bounds

    Grid = GridSpec(bounds=(minx, miny, maxx, maxy), res=res)
    
    return Grid


def build_grid_geojson(grid: GridSpec, mask: xr.DataArray) -> gpd.GeoDataFrame:
    """Build a fishnet GeoDataFrame of grid cells (one polygon per in-bound cell).

    Each feature carries its zarr indices (iy, ix) and cell center (x, y) so a
    dashboard can map a clicked cell back to the zarr via `isel(y=iy, x=ix)`.
    Only cells inside the boundary are emitted, so there are no dead clicks.

    `mask` is the (y, x) boundary mask from `rasterize_mask` (1 inside, 0
    outside) and must have shape `grid.shape`.
    """
    inside = np.asarray(mask).astype(bool)                    # keep only in-boundary cells
    if inside.shape != grid.shape:
        raise ValueError(f"mask shape {inside.shape} != grid shape {grid.shape}")

    ny, nx = grid.shape                                       # numpy order: (rows=y, cols=x)
    xx, yy = np.meshgrid(grid.x, grid.y)                      # (ny, nx) cell centers
    iy_idx, ix_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    half = grid.res / 2
    polys = shapely.box(xx - half, yy - half, xx + half, yy + half)

    return gpd.GeoDataFrame(
        {
            "iy": iy_idx[inside],
            "ix": ix_idx[inside],
            "x": xx[inside],
            "y": yy[inside],
        },
        geometry=polys[inside],
        crs=grid.crs,
    )