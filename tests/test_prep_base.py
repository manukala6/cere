import numpy as np
import pytest
import xarray as xr

from prep_base import clip_base, add_pixel_area


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
