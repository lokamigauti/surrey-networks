import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
import struct
import rasterio as rio
from rasterio.enums import MergeAlg
import geocube
from geocube.api.core import make_geocube

# The LRI is a tensor with two spatial dimensions and one dimension relative to the road function (e.g. access road,
# local road, and A-road). The LRI is calculated using a series of raster maps of the roads filtered by their functions,
# spatially displaced by the mean wind in the previous hour. In the “per point” model, each point receives a cut of this
# tensor centred in the displaced point location with a rectangular size r, which we fixed as 9.

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
ROADS = 'Roads/'
RESOLUTION = (100, 100)

def advect_roads(wind, roads):
    if any(np.isnan(wind)):
        return np.nan
    roads = roads.copy()
    roads['geometry'] = roads.translate(xoff=-wind[0], yoff=-wind[1])
    roads = roads.set_crs(epsg=27700, allow_override=True)
    out_grid = make_geocube(
        vector_data=roads,
        resolution=RESOLUTION,
        fill=0,
        group_by='function',
        rasterize_function=lambda **kwargs: geocube.rasterize.rasterize_image(**kwargs, merge_alg=MergeAlg.add),
    )
    out_grid = out_grid.rio.reproject('EPSG:4326')

    ## Debug #
    # file_name = 'A_test' + '_' + str(RESOLUTION[0]) + 'x' + str(RESOLUTION[1]) + 'trans.tif'
    # out_grid.sel(function='A Road').rio.to_raster(DATA_DIR + ROADS + 'Functions/' + file_name.replace(' ', '_'))
    return out_grid


if __name__ == '__main__':
    roads = gpd.read_file(DATA_DIR + ROADS + 'roads.gpkg')
    era5 = xr.open_dataset(DATA_DIR + 'ERA5/formatted/era5.nc').isel(longitude=1, latitude=1)
    era5 = era5.sel(time=slice('2021-08-04', '2021-12-13'))
    wind = [[era5.sel(time=time).u10.values, era5.sel(time=time).v10.values] for time in era5.time.values]
    wind = wind * 60  # m/h
    lri = advect_roads(wind[0], roads)
    for wind_ in wind.pop(0):
        lri_time = advect_roads(wind_, roads)
        lri = xr.concat((lri, lri_time), dim='time')
        lri.to_netcdf(DATA_DIR + ROADS + 'lri.nc')
