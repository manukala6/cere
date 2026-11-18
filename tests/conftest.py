import pytest
import shapely
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray

@pytest.fixture
def res():
    return 0.5

@pytest.fixture
def crs():
    return "EPSG:4326"

@pytest.fixture
def boundary_gdf(crs):
    return gpd.GeoDataFrame(
        {"id": [1]}, geometry=[shapely.box(0.1, 0.1, 0.9, 0.9)], crs=crs
    )

@pytest.fixture
def grid(crs):
    x = np.array([0.25, 0.75]) # centers
    y = np.array([0.75, 0.25]) # centers
    return xr.Dataset(coords={"y": ("y", y), "x": ("x", x)}).rio.write_crs(crs)
