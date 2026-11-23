import numpy as np
import pytest
import xarray as xr

from build_grid import GridSpec
from checks import FINE_BASE, SchemaError, validate
from prep_base import clip_base, add_pixel_area, prepare_base


@pytest.fixture
def source():
    """0.1 deg source over [-0.5, 2.0] x [-0.5, 1.5], wider than the grid; y descending."""
    x = -0.45 + np.arange(25) * 0.1
    y = 1.45 - np.arange(20) * 0.1
    return xr.DataArray(
        np.zeros((2, 20, 25), dtype="float32"),
        dims=("year", "y", "x"),
        coords={"year": [2020, 2021], "y": y, "x": x},
    )


@pytest.fixture
def grid_2x3():
    return GridSpec(bounds=(0.0, 0.0, 1.5, 1.0), res=0.5)


def test_prepare_base_returns_clipped_valid_base(source, grid_2x3):
    base = prepare_base(source, grid_2x3)

    validate(base, FINE_BASE, "base")
    assert dict(base.sizes) == {"year": 2, "y": 10, "x": 15}


def test_prepare_base_blames_an_ascending_source(source, grid_2x3):
    """Caught before clipping, so the error names the cause, not the empty clip it causes."""
    with pytest.raises(SchemaError, match="source failed") as err:
        prepare_base(source.sortby("y"), grid_2x3)
    assert "y must be strictly descending" in str(err.value)


def test_prepare_base_rejects_a_disjoint_source(source):
    far = GridSpec(bounds=(10.0, 10.0, 11.0, 11.0), res=0.5)
    with pytest.raises(SchemaError, match="does not overlap"):
        prepare_base(source, far)


def test_prepare_base_rejects_a_clip_too_small_for_pixel_area(source):
    """A grid overlapping only the source's last pixel column clips to x = 1."""
    sliver = GridSpec(bounds=(1.9, 0.0, 2.4, 0.5), res=0.5)
    with pytest.raises(SchemaError, match="clipped source failed") as err:
        prepare_base(source, sliver)
    assert "x has 1 entries (need >= 2)" in str(err.value)


def test_clip_base_keeps_only_pixels_inside_bounds():
    """0.1 deg source wider than the bounds -> exactly the 10 x 15 pixels inside them."""
    x = -0.45 + np.arange(25) * 0.1
    y = 1.45 - np.arange(20) * 0.1                 # descending, like the TTC zarr
    src = xr.DataArray(np.zeros((20, 25)), dims=("y", "x"), coords={"y": y, "x": x})

    out = clip_base(src, (0.0, 0.0, 1.5, 1.0))

    assert dict(out.sizes) == {"y": 10, "x": 15}
    assert out.y[0] == pytest.approx(0.95) and out.y[-1] == pytest.approx(0.05)
    assert out.x[0] == pytest.approx(0.05) and out.x[-1] == pytest.approx(1.45)


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
