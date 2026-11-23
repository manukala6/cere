import numpy as np
import pytest
import xarray as xr

from build_grid import (
    GridSpec,
    rasterize_mask,
    build_grid_geojson,
)


def test_from_bounds_boundary_bbox(boundary_gdf, res, crs):
    """The boundary's bbox (as flow.py builds it) snaps out to the unit lattice: 2x2 cells."""
    grid = GridSpec.from_bounds(tuple(boundary_gdf.to_crs(crs).total_bounds), res, crs)

    assert grid.bounds == (0.0, 0.0, 1.0, 1.0)
    assert grid.shape == (2, 2)
    assert grid.x.tolist() == [0.25, 0.75]
    assert grid.y.tolist() == [0.75, 0.25]  # descending for raster orientation
    assert grid.crs == crs
    assert grid.to_dataset().rio.crs == crs


def test_from_bounds_snaps_outward():
    """A wider-than-tall box snaps to (ny, nx) = (2, 3), in numpy order."""
    grid = GridSpec.from_bounds((0.1, 0.1, 1.4, 0.9), 0.5)

    assert grid.bounds == (0.0, 0.0, 1.5, 1.0)
    assert grid.shape == (2, 3)
    assert grid.x.tolist() == [0.25, 0.75, 1.25]


def test_from_bounds_keeps_quarter_degree_edges():
    """Mozambique-like bbox at 0.25: the top edge must stay -10.25, not round to -10.2."""
    grid = GridSpec.from_bounds((30.2174, -26.8681, 40.839, -10.4713), 0.25)

    assert grid.bounds == (30.0, -27.0, 41.0, -10.25)
    assert grid.shape == (67, 44)
    assert grid.y[0] == pytest.approx(-10.375)
    assert grid.x[0] == pytest.approx(30.125)


def test_from_bounds_ignores_float_noise():
    """0.3 / 0.1 == 2.9999999999999996 must not snap out to an extra cell."""
    grid = GridSpec.from_bounds((0.3, 0.3, 0.7, 0.7), 0.1)

    assert grid.bounds == (0.3, 0.3, 0.7, 0.7)
    assert grid.shape == (4, 4)


@pytest.mark.parametrize(
    "bounds, res, match",
    [
        ((0.1, 0.0, 1.0, 1.0), 0.5, "whole number"),   # x extent 0.9 = 1.8 cells
        ((0.0, 0.0, 1.0, 1.0), 0.0, "res must be positive"),
        ((1.0, 0.0, 0.0, 1.0), 0.5, "minx < maxx"),
        ((0.0, 80.0, 1.0, 95.0), 0.5, r"\[-90, 90\]"),
    ],
)
def test_gridspec_rejects_invalid(bounds, res, match):
    with pytest.raises(ValueError, match=match):
        GridSpec(bounds=bounds, res=res)


def test_rasterize_mask(boundary_gdf, grid):
    """Boundary covers every snapped cell -> an all-ones (ny, nx) mask."""
    mask = rasterize_mask(boundary_gdf, grid)

    assert mask.shape == (grid.y.size, grid.x.size)
    assert set(np.unique(mask.values)).issubset({0, 1})
    assert (mask.values == 1).all()


@pytest.fixture
def wide_grid(res, crs):
    """Non-square 2 x 3 grid (ny=2, nx=3) so an nx/ny swap can't pass unnoticed."""
    return GridSpec(bounds=(0.0, 0.0, 1.5, 1.0), res=res, crs=crs)


def test_build_grid_geojson_roundtrip(wide_grid, res, crs):
    """Every emitted cell maps back to its zarr index; out-of-bound cells are dropped."""
    mask = xr.DataArray(
        np.array([[1, 1, 1],
                  [1, 1, 0]]),                     # drop the bottom-right cell (iy=1, ix=2)
        dims=("y", "x"),
        coords={"y": wide_grid.y, "x": wide_grid.x},
    )
    gdf = build_grid_geojson(wide_grid, mask)

    assert len(gdf) == int(mask.values.sum()) == 5
    assert (1, 2) not in set(zip(gdf.iy, gdf.ix))
    assert set(["iy", "ix", "x", "y"]).issubset(gdf.columns)
    assert gdf.crs == crs

    for row in gdf.itertuples():
        # index round-trip: clicked cell -> correct zarr coordinate
        assert wide_grid.x[row.ix] == row.x
        assert wide_grid.y[row.iy] == row.y
        # each polygon spans exactly one cell (res x res), centered on (x, y)
        minx, miny, maxx, maxy = row.geometry.bounds
        assert maxx - minx == pytest.approx(res)
        assert maxy - miny == pytest.approx(res)
        assert row.geometry.centroid.x == pytest.approx(row.x)
        assert row.geometry.centroid.y == pytest.approx(row.y)


def test_build_grid_geojson_rejects_mismatched_mask(wide_grid):
    """A mask in the wrong (nx, ny) order fails fast instead of mislabeling cells."""
    with pytest.raises(ValueError, match="mask shape"):
        build_grid_geojson(wide_grid, np.ones((3, 2)))
