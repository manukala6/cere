import xarray as xr
import geopandas as gpd

from build_grid import GridSpec, rasterize_mask
from prep_base import clip_base, add_pixel_area, label_pixels
from summarize import summarize_landcover

###### CONSTRUCT COARSE GRID `build_grid.py` ######
# read study area
boundary_path = ""
boundary = gpd.read_file(boundary_path)

# build coarse grid for study area
res = 0.25
crs = "EPSG:4326"
minx, miny, maxx, maxy = boundary.to_crs(crs).total_bounds
grid = GridSpec.from_bounds((minx, miny, maxx, maxy), res, crs)

# build the mask
mask = rasterize_mask(boundary, grid.to_dataset())

###### PREPARE SOURCE DATA (FINE RES) FOR AGGREGATION `prep_base.py` ######
# open the source zarr
zarr_path = ""
base = xr.open_zarr(zarr_path)["variable_of_interest"]

# clip to bounds
base = clip_base(base, grid.bounds)

# add pixel area
base = add_pixel_area(base)

# label pixels

###### AGGREGATION FROM FINE PIXELS TO COARSE GRID CELLS `summarize.py` ######
# sum pixels into grid cells

# mask pixels

# calculate shares per grid cells

###### ADD CLIMATE LAYERS
# add ERA5-Land KDD and GDD

# add CHIRPS precipitation shock

###### EXPORT ######
# write to zarr, ``

# write to gejson
