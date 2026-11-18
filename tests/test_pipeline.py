import numpy as np
import pytest
import xarray as xr

from build_grid import (
    add_pixel_area,
    build_grid,
    rasterize_mask,
    build_grid_geojson,
)


def test_build_grid(boundary_gdf, res, crs):
    """The inset square snaps out to the unit lattice: 2x2 cells, y descending."""
    grid, bounds, shape, coords = build_grid(boundary_gdf, res, crs)
    x, y = coords

    assert bounds == (0.0, 0.0, 1.0, 1.0)
    assert shape == (2, 2)
    assert x.tolist() == [0.25, 0.75]
    assert y.tolist() == [0.75, 0.25]  # descending for raster orientation
    assert grid.rio.crs == crs


def test_add_pixel_area_structure(grid):
    """Coordinate is attached per-row, float32, and strictly positive."""
    out = add_pixel_area(grid)
    area = out.pixel_area_ha

    assert area.dims == ("y",)
    assert area.shape == grid.y.shape
    assert area.dtype == np.dtype("float32")
    assert (area.values > 0).all()


def test_add_pixel_area_decreases_toward_pole(grid):
    """Grid y = [0.75, 0.25]; the higher-latitude cell has the smaller area."""
    area = add_pixel_area(grid).pixel_area_ha.values
    assert area[0] < area[1]  # y=0.75 (nearer pole) < y=0.25 (nearer equator)


def test_add_pixel_area_equator_magnitude(crs):
    """A 1x1 degree cell at the equator is ~1.236e6 ha (12,364 km^2)."""
    ds = xr.Dataset(
        coords={"y": ("y", np.array([0.5, -0.5])), "x": ("x", np.array([0.5, 1.5]))}
    )
    area = add_pixel_area(ds).pixel_area_ha.values
    assert area[0] == pytest.approx(1.236e6, rel=1e-3)
    assert area[0] == pytest.approx(area[1])  # symmetric about the equator


def test_rasterize_mask(boundary_gdf, grid):
    """Boundary covers every snapped cell -> an all-ones (ny, nx) mask."""
    mask = rasterize_mask(boundary_gdf, grid)

    assert mask.shape == (grid.y.size, grid.x.size)
    assert set(np.unique(mask.values)).issubset({0, 1})
    assert (mask.values == 1).all()


def test_build_grid_geojson_roundtrip(boundary_gdf, grid, res, crs):
    """Every emitted cell maps back to its zarr index, one polygon per in-bound cell."""
    mask = rasterize_mask(boundary_gdf, grid)
    x = grid.x.values
    y = grid.y.values
    shape = (x.size, y.size)
    gdf = build_grid_geojson(shape, (x, y), res, mask, crs)

    assert len(gdf) == int(mask.values.sum()) == 4
    assert set(["iy", "ix", "x", "y"]).issubset(gdf.columns)
    assert gdf.crs == crs

    for row in gdf.itertuples():
        # index round-trip: clicked cell -> correct zarr coordinate
        assert grid.x.isel(x=row.ix).item() == row.x
        assert grid.y.isel(y=row.iy).item() == row.y
        # each polygon spans exactly one cell (res x res)
        minx, miny, maxx, maxy = row.geometry.bounds
        assert maxx - minx == pytest.approx(res)
        assert maxy - miny == pytest.approx(res)
