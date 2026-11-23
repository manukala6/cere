"""Build the coarse land-cover analysis zarr: grid -> prep -> aggregate -> export.

Rough outline for now; later `build_analysis_grid` becomes a Prefect flow driven
by a config. Run as a script for the default Mozambique TTC build.
"""

from pathlib import Path

import geopandas as gpd
import xarray as xr
from dask.diagnostics import ProgressBar

from build_grid import GridSpec, rasterize_mask, build_grid_geojson
from prep_base import prepare_base
from summarize import summarize_landcover


def build_analysis_grid(
        boundary_path: str | Path,
        zarr_path: str | Path,
        out_path: str | Path,
        *,
        var: str = "treecover",
        out_var: str = "ttc_ha_gridded",
        res: float = 0.25,
        crs: str = "EPSG:4326",
        geojson_path: str | Path | None = None,
) -> Path:
    
    ###### CONSTRUCT COARSE GRID `build_grid.py` ######
    # read study area
    boundary = gpd.read_file(boundary_path)

    # build coarse grid for study area
    minx, miny, maxx, maxy = boundary.to_crs(crs).total_bounds
    grid = GridSpec.from_bounds((minx, miny, maxx, maxy), res, crs)

    # build the mask
    mask = rasterize_mask(boundary, grid.to_dataset())

    ###### PREPARE SOURCE DATA (FINE RES) FOR AGGREGATION `prep_base.py` ######
    # open the source zarr (lazy)
    base = xr.open_zarr(zarr_path)[var]

    # clip to bounds, add pixel area, and validate (-> FineBase)
    base = prepare_base(base, grid)

    # label pixels
    # TODO: label_pixels / class rule; the TTC weighting is still hard-coded in summarize_landcover

    ###### AGGREGATION FROM FINE PIXELS TO COARSE GRID CELLS `summarize.py` ######
    # sum pixels into grid cells, then mask cells outside the boundary (still lazy)
    analysis_grid = summarize_landcover(base, grid, mask)

    # calculate shares per grid cells
    # TODO

    ###### ADD CLIMATE LAYERS
    # add ERA5-Land KDD and GDD

    # add CHIRPS precipitation shock

    ###### EXPORT ######
    # write to zarr -- dask computes here. flox leaves uneven chunks, which zarr
    # can't store; the coarse result is small, so collapse to one chunk.
    out = (
        analysis_grid
        .rename(out_var)
        .chunk({dim: -1 for dim in analysis_grid.dims})
        .to_dataset()
    )
    print(f"writing {out_var} {dict(out.sizes)} -> {out_path}")
    with ProgressBar():
        out.to_zarr(out_path, mode="w", consolidated=False)

    # write to geojson
    if geojson_path is not None:
        build_grid_geojson(grid, mask).to_file(geojson_path, driver="GeoJSON")

    return Path(out_path)


if __name__ == "__main__":
    # data is currently in another repo TODO: fix this later
    data_dir = Path(__file__).resolve().parents[2] / "droughts_deforestation" / "data"
    build_analysis_grid(
        boundary_path=data_dir / "geometries" / "gadm41_MOZ_0.shp",
        zarr_path=data_dir / "processed" / "tropical_tree_cover.zarr",
        out_path=data_dir / "processed" / "ttc_ha_gridded_0p25_cere.zarr",
        geojson_path=data_dir / "processed" / "ttc_grid_0p25_cere.geojson",
    )
