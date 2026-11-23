"""Data contracts between pipeline stages.

A schema is a list of small rules. `validate` runs every rule and reports all
failures at once. Rules read metadata only (dims, sizes, 1-D coordinates), never
pixel values, so validating a lazy 10 m dask array triggers no compute.

Usage (see prep_base.prepare_base and summarize.summarize_landcover):
    validate(base, FINE_BASE + [overlaps(grid.bounds)], "prepared base")
    return FineBase(base)
"""

from typing import Callable, Iterable, NewType

import numpy as np
import xarray as xr

from build_grid import Bounds

XrObject = xr.DataArray | xr.Dataset

# A rule: returns None when the object passes, or a message saying what's wrong.
Check = Callable[[XrObject], str | None]


class SchemaError(ValueError):
    """Data that breaks a stage's contract."""


# ── reusable rules ───────────────────────────────────────────────────────────
# Each rule skips quietly when a dim/coord it needs is missing, because
# has_dims / has_coord report that; so every rule in a schema can always run.

def has_dims(*dims: str) -> Check:
    """All of `dims` are present (extra dims, e.g. "year", are allowed)."""
    def check(obj: XrObject) -> str | None:
        missing = sorted(set(dims) - set(obj.dims))
        return f"missing dims {missing}" if missing else None
    return check


def min_size(dim: str, n: int = 2) -> Check:
    """Dimension `dim` has at least `n` entries."""
    def check(obj: XrObject) -> str | None:
        if dim not in obj.dims:
            return None
        size = obj.sizes[dim]
        return f"{dim} has {size} entries (need >= {n}); empty clip?" if size < n else None
    return check


def monotonic(dim: str, descending: bool = False) -> Check:
    """Coordinate `dim` is strictly ascending (or strictly descending)."""
    order = "descending" if descending else "ascending"

    def check(obj: XrObject) -> str | None:
        if dim not in obj.coords or obj.sizes.get(dim, 0) < 2:
            return None
        steps = np.diff(obj[dim].values)
        ok = (steps < 0).all() if descending else (steps > 0).all()
        return None if ok else f"{dim} must be strictly {order}"
    return check


def has_coord(name: str, dims: tuple[str, ...], positive: bool = False, hint: str = "") -> Check:
    """Coordinate `name` exists on exactly `dims` (and is > 0 everywhere if `positive`)."""
    suffix = f"; {hint}" if hint else ""

    def check(obj: XrObject) -> str | None:
        if name not in obj.coords:
            return f"missing coord {name!r}{suffix}"
        coord = obj.coords[name]
        if coord.dims != tuple(dims):
            return f"coord {name!r} has dims {coord.dims}, expected {tuple(dims)}{suffix}"
        if positive and not bool((coord > 0).all()):
            return f"coord {name!r} must be strictly positive{suffix}"
        return None
    return check


def overlaps(bounds: Bounds) -> Check:
    """The x/y extent intersects `bounds` = (minx, miny, maxx, maxy). Built per call."""
    minx, miny, maxx, maxy = bounds

    def check(obj: XrObject) -> str | None:
        if any(d not in obj.coords or obj.sizes.get(d, 0) == 0 for d in ("x", "y")):
            return None
        xs, ys = obj["x"].values, obj["y"].values
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x1 < minx or x0 > maxx or y1 < miny or y0 > maxy:
            return (f"extent x[{x0:g}, {x1:g}] y[{y0:g}, {y1:g}] "
                    f"does not overlap bounds {bounds}")
        return None
    return check


# ── named schemas: the contracts between stages ──────────────────────────────

# Fine-resolution raster layout that clip_base and add_pixel_area assume
# (an ascending y would make clip_base's slice(maxy, miny) return nothing).
FINE_GRID: list[Check] = [
    has_dims("y", "x"),
    min_size("y"),
    min_size("x"),
    monotonic("y", descending=True),
    monotonic("x"),
]

# Fine source after prep_base.prepare_base (clip -> pixel area),
# as required by summarize_landcover.
FINE_BASE: list[Check] = FINE_GRID + [
    has_coord("pixel_area_ha", dims=("y",), positive=True,
              hint="run prep_base.prepare_base first"),
]


# ── runner ───────────────────────────────────────────────────────────────────

def validate(obj: XrObject, schema: Iterable[Check], name: str) -> None:
    """Run every rule in `schema`; raise one SchemaError listing all failures."""
    errors = [msg for check in schema if (msg := check(obj)) is not None]
    if errors:
        raise SchemaError(f"{name} failed {len(errors)} check(s):\n  - " + "\n  - ".join(errors))


# Marks a DataArray that passed FINE_BASE. Zero runtime cost: mypy flags passing a
# raw DataArray where a FineBase is expected (the "forgot to run prep" mistake).
FineBase = NewType("FineBase", xr.DataArray)
