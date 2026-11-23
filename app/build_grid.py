"""Takes in base analysis layer and outputs a gridded summary zarr
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (registers the `.rio` accessor used below)
import shapely
from geocube.api.core import make_geocube
from dataclasses import dataclass

Bounds = tuple[float, float, float, float]

# Tolerance (in cells) for float noise when checking/snapping to the res lattice.
_LATTICE_TOL = 1e-9


@dataclass(frozen=True)
class GridSpec:
    """Regular lat/lon summary grid that the base LULC data is aggregated onto.

    `bounds` must already span a whole number of `res` cells; use
    `GridSpec.from_bounds` to snap arbitrary bounds outward first.
    """
    bounds: Bounds
    res: float
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        """Validate only; never modify the inputs."""
        if self.res <= 0:
            raise ValueError(f"res must be positive, got {self.res}")
        minx, miny, maxx, maxy = self.bounds
        if not (minx < maxx and miny < maxy):
            raise ValueError(f"bounds must satisfy minx < maxx and miny < maxy, got {self.bounds}")
        if miny < -90 or maxy > 90:
            raise ValueError(
                f"y bounds must lie within [-90, 90] (GridSpec expects degrees), got {self.bounds}"
            )
        for name, extent in (("x", maxx - minx), ("y", maxy - miny)):
            cells = extent / self.res
            if abs(cells - round(cells)) > _LATTICE_TOL * max(1.0, cells):
                raise ValueError(
                    f"{name} extent {extent} is not a whole number of {self.res} cells; "
                    f"use GridSpec.from_bounds to snap the bounds"
                )

    @classmethod
    def from_bounds(cls, bounds: Bounds, res: float, crs: str = "EPSG:4326") -> "GridSpec":
        """Snap arbitrary bounds outward onto the global `res` lattice (multiples of res).

        Snaps via integer cell indices, so values like -10.25 at res=0.25 stay exact
        (rounding to a fixed number of decimals would turn them into -10.2).
        """
        if res <= 0:
            raise ValueError(f"res must be positive, got {res}")
        minx, miny, maxx, maxy = bounds

        def snap(value: float, op) -> float:
            # round the index first so float noise (2.9999999999999996) can't add a cell
            index = op(np.round(value / res, 9))
            return float(np.round(index * res, 10))

        snapped = (snap(minx, np.floor), snap(miny, np.floor),
                   snap(maxx, np.ceil), snap(maxy, np.ceil))
        return cls(bounds=snapped, res=res, crs=crs)

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