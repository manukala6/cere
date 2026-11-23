import dask
import numpy as np
import pytest
import xarray as xr

from checks import (
    FINE_BASE,
    SchemaError,
    has_coord,
    has_dims,
    min_size,
    monotonic,
    overlaps,
    validate,
)
from build_grid import GridSpec
from prep_base import add_pixel_area, prepare_base
from summarize import summarize_landcover


@pytest.fixture
def fine_base():
    """A prepared 0.1 deg base over [0, 1.5] x [0, 1]: y descending, pixel_area_ha attached."""
    x = 0.05 + np.arange(15) * 0.1
    y = 0.95 - np.arange(10) * 0.1
    da = xr.DataArray(
        np.zeros((2, 10, 15), dtype="float32"),
        dims=("year", "y", "x"),
        coords={"year": [2020, 2021], "y": y, "x": x},
    )
    return add_pixel_area(da)


def test_prepared_base_passes(fine_base):
    validate(fine_base, FINE_BASE + [overlaps((0.0, 0.0, 1.5, 1.0))], "base")


def test_missing_pixel_area_names_the_fix(fine_base):
    with pytest.raises(SchemaError, match="run prep_base.prepare_base first"):
        validate(fine_base.drop_vars("pixel_area_ha"), FINE_BASE, "base")


def test_reports_every_failure_at_once(fine_base):
    """Ascending y and no pixel area -> both listed in one error."""
    bad = fine_base.drop_vars("pixel_area_ha").sortby("y")
    with pytest.raises(SchemaError, match=r"failed 2 check\(s\)") as err:
        validate(bad, FINE_BASE, "base")
    assert "y must be strictly descending" in str(err.value)
    assert "missing coord 'pixel_area_ha'" in str(err.value)


def test_empty_clip_is_caught(fine_base):
    empty = fine_base.sel(y=slice(0.0, 1.0))     # ascending slice on descending y -> 0 rows
    with pytest.raises(SchemaError, match="y has 0 entries"):
        validate(empty, FINE_BASE, "base")


def test_wrong_dims_raise_schema_error_not_keyerror():
    """Rules skip what they can't check, so a malformed input still gets a clean report."""
    da = xr.DataArray(np.zeros((3, 3)), dims=("a", "b"))
    with pytest.raises(SchemaError, match=r"missing dims \['x', 'y'\]"):
        validate(da, FINE_BASE, "base")


@pytest.mark.parametrize(
    "check, expected",
    [
        (has_dims("y", "x", "year"), None),
        (has_dims("y", "band"), "missing dims ['band']"),
        (min_size("x", 16), "x has 15 entries (need >= 16); empty clip?"),
        (monotonic("y"), "y must be strictly ascending"),
        (monotonic("x", descending=True), "x must be strictly descending"),
        (has_coord("pixel_area_ha", dims=("x",)), "coord 'pixel_area_ha' has dims ('y',), expected ('x',)"),
        (overlaps((5.0, 5.0, 6.0, 6.0)), "extent x[0.05, 1.45] y[0.05, 0.95] does not overlap bounds (5.0, 5.0, 6.0, 6.0)"),
    ],
)
def test_rule_messages(fine_base, check, expected):
    assert check(fine_base) == expected


def test_has_coord_positive(fine_base):
    zeroed = fine_base.assign_coords(pixel_area_ha=fine_base.pixel_area_ha * 0)
    assert has_coord("pixel_area_ha", dims=("y",), positive=True)(zeroed) == (
        "coord 'pixel_area_ha' must be strictly positive"
    )


def test_handoff_prepare_base_to_summarize(fine_base):
    """prepare_base's output meets summarize_landcover's contract; a raw base does not."""
    grid = GridSpec(bounds=(0.0, 0.0, 1.5, 1.0), res=0.5)
    mask = xr.DataArray(np.ones(grid.shape, dtype=int), dims=("y", "x"),
                        coords={"y": grid.y, "x": grid.x})
    raw = fine_base.drop_vars("pixel_area_ha")

    out = summarize_landcover(prepare_base(raw, grid), grid, mask)
    assert out.shape == (2, *grid.shape)

    with pytest.raises(SchemaError, match="run prep_base.prepare_base first"):
        summarize_landcover(raw, grid, mask)


def test_rules_never_compute_pixel_data(fine_base):
    """Validating a lazy base (as from open_zarr + add_pixel_area) triggers no dask compute."""
    lazy = add_pixel_area(fine_base.drop_vars("pixel_area_ha").chunk({"y": 5, "x": 5}))

    def no_compute(*args, **kwargs):
        raise AssertionError("validate triggered a dask compute")

    with dask.config.set(scheduler=no_compute):
        validate(lazy, FINE_BASE, "base")
