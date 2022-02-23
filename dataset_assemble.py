import numpy as np
import pandas as pd
import xarray as xr
from tools import calc_angle_concordance
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.enums import MergeAlg
import geocube
from geocube.api.core import make_geocube

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'
ERA5 = 'G:/My Drive/IC/Doutorado/Sandwich/Data/ERA5/'
PFI = 'G:/My Drive/IC/Doutorado/Sandwich/Data/Features/'
ROADS = 'Roads/'

OUTSKIRTS_SIZE = 50 / 111139
LON_RESOLUTION = 20
LAT_RESOLUTION = 20



def ord_dist(lon_lat, lcm, n, stationless=False):
    dist = np.sqrt((lon_lat.longitude - lcm.longitude) ** 2 + (lon_lat.latitude - lcm.latitude) ** 2)
    dist = dist.sortby(dist)
    if stationless:
        dist = dist.drop('station')
    return dist[n]


def ord_stat_pm10(lon_lat, lcm, n, stationless=False):
    stat = ord_dist(lon_lat, lcm, n)
    station_ = stat.station
    pm10 = lcm.sel(station=station_).pm10
    if not stationless:
        pm10['station'] = lon_lat.station
        return pm10
    pm10 = pm10.drop('station')
    return pm10


def ord_stat_pm25(lon_lat, lcm, n, stationless=False):
    stat = ord_dist(lon_lat, lcm, n)
    station_ = stat.station
    pm25 = lcm.sel(station=station_).pm25
    if not stationless:
        pm25['station'] = lon_lat.station
        return pm25
    pm25 = pm25.drop('station')
    return pm25


def ord_stat_pm1(lon_lat, lcm, n, stationless=False):
    stat = ord_dist(lon_lat, lcm, n)
    station_ = stat.station
    pm1 = lcm.sel(station=station_).pm1
    if not stationless:
        pm1['station'] = lon_lat.station
        return pm1
    pm1 = pm1.drop('station')
    return pm1


def ord_stat_angle(lon_lat, lcm, n, stationless=False):
    stat = ord_dist(lon_lat, lcm, n)
    station_ = stat.station
    u = lon_lat.longitude - lcm.sel(station=station_).longitude
    v = lon_lat.latitude - lcm.sel(station=station_).latitude
    angle = np.degrees(np.arctan2(u, v))
    if stationless:
        angle = angle.drop('station')
    return angle


def calc_lri(lon_lat):
    roads = gpd.read_file(DATA_DIR + ROADS + 'roads.gpkg')
    roads = roads.set_crs(epsg=27700, allow_override=True)
    roads_raster = make_geocube(
        vector_data=roads,
        resolution=(5, 5),
        fill=0,
        group_by='function',
        rasterize_function=lambda **kwargs: geocube.rasterize.rasterize_image(**kwargs, merge_alg=MergeAlg.add),
    )
    roads_raster = roads_raster.rio.reproject('EPSG:4326')
    roads_raster = roads_raster.n.to_dataset(dim='function') \
        .reset_coords(drop=True)

    lon_min = lon_lat.longitude - (OUTSKIRTS_SIZE / 2)
    lon_max = lon_lat.longitude + (OUTSKIRTS_SIZE / 2)
    lat_min = lon_lat.latitude - (OUTSKIRTS_SIZE / 2)
    lat_max = lon_lat.latitude + (OUTSKIRTS_SIZE / 2)
    lri = roads_raster.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max)).sum()
    return lri


def training_dataset_assemble(data, savepath):
    data['nearest_monitor'] = data.groupby('station').map(ord_dist, args=(data, 1))
    data['2nearest_monitor'] = data.groupby('station').map(ord_dist, args=(data, 2))

    data['nearest_monitor_pm10'] = data.groupby('station').map(ord_stat_pm10,
                                                               args=(data, 1))
    data['2nearest_monitor_pm10'] = data.groupby('station').map(ord_stat_pm10,
                                                                args=(data, 2))

    data['nearest_monitor_pm25'] = data.groupby('station').map(ord_stat_pm25,
                                                               args=(data, 1))
    data['2nearest_monitor_pm25'] = data.groupby('station').map(ord_stat_pm25,
                                                                args=(data, 2))

    data['nearest_monitor_pm1'] = data.groupby('station').map(ord_stat_pm1, args=(data, 1))
    data['2nearest_monitor_pm1'] = data.groupby('station').map(ord_stat_pm1, args=(data, 2))

    data['day_of_week'] = data.time.dt.dayofweek
    data['month'] = data.time.dt.month

    data['nearest_monitor_angle'] = data.groupby('station').map(ord_stat_angle,
                                                                args=(data, 1))
    data['2nearest_monitor_angle'] = data.groupby('station').map(ord_stat_angle,
                                                                 args=(data, 2))

    era5 = xr.open_dataset(ERA5 + 'formatted/era5.nc')
    era5['wind_angle'] = np.degrees(np.arctan2(era5.u10, era5.v10))

    era5 = era5.sel(time=slice(data.time.min(),
                               data.time.max()))
    era5 = era5.sel(time=~era5.get_index('time').duplicated(keep='last'))
    wind_angle_central = era5['wind_angle'][:, 1, 1]
    wind_angle_central = wind_angle_central.drop_duplicates(dim='time', keep='last').squeeze().reset_coords(
        drop=True)
    data = xr.merge([data, wind_angle_central])
    data['nearest_monitor_angle_wind'] = calc_angle_concordance(data.wind_angle,
                                                                data.nearest_monitor_angle)
    data['2nearest_monitor_angle_wind'] = calc_angle_concordance(data.wind_angle,
                                                                 data['2nearest_monitor_angle'])

    lri = data.groupby('station').map(calc_lri)
    data = xr.merge([data, lri])

    era5_varslist = list(era5.keys())
    sw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=0).rename({var: 'sw_' + var for var in era5_varslist}).reset_coords(drop=True)
    s_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=0).rename({var: 's_' + var for var in era5_varslist}).reset_coords(drop=True)
    se_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=0).rename({var: 'se_' + var for var in era5_varslist}).reset_coords(drop=True)
    w_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=1).rename({var: 'w_' + var for var in era5_varslist}).reset_coords(drop=True)
    c_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=1).rename({var: 'c_' + var for var in era5_varslist}).reset_coords(drop=True)
    e_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=1).rename({var: 'e_' + var for var in era5_varslist}).reset_coords(drop=True)
    nw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=2).rename({var: 'nw_' + var for var in era5_varslist}).reset_coords(drop=True)
    n_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=2).rename({var: 'n_' + var for var in era5_varslist}).reset_coords(drop=True)
    ne_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=2).rename({var: 'ne_' + var for var in era5_varslist}).reset_coords(drop=True)

    data = xr.merge([data, sw_era5, s_era5, se_era5, w_era5,
                     c_era5, e_era5, e_era5, nw_era5, n_era5, ne_era5])

    data.to_netcdf(savepath)


if __name__ == '__main__':
    lcm_path = OUTPUT_DIR + LCS + 'data_calibrated.nc'
    lcm = xr.open_dataarray(lcm_path)
    lcm = lcm.sel(time=~lcm.get_index('time').duplicated(keep='last'))
    lcm_meta_path = DATA_DIR + LCS + 'Woking Green Party deatils.csv'
    lcm_meta = pd.read_csv(lcm_meta_path)
    lcm_coords = lcm_meta[['Device/Sensor Name assigned', 'lat', 'lon']]. \
        rename(columns={'Device/Sensor Name assigned': 'station'}).set_index('station')
    lcm = lcm.to_dataset(dim='variable')
    lcm = lcm.resample(time='1H').mean()
    lcm = lcm[['pm10_cal', 'pm25_cal', 'pm1_cal', 'rh', 'temp', 'dew', 'wetbulb']]
    lcm = lcm.rename_vars({'pm10_cal': 'pm10',
                           'pm25_cal': 'pm25',
                           'pm1_cal': 'pm1'})
    lcm['longitude'] = lcm_coords['lon']
    lcm['latitude'] = lcm_coords['lat']

    lcm_validation = lcm.sel(station=['WokingGreens#2', 'WokingGreens#7', 'WokingGreens#8'])
    lcm_training = lcm.sel(station=['WokingGreens#1', 'WokingGreens#3', 'WokingGreens#4',
                                    'WokingGreens#5', 'WokingGreens#6'])

    training_dataset_assemble(lcm_validation, DATA_DIR + 'Features/validation.nc')
    training_dataset_assemble(lcm_training, DATA_DIR + 'Features/training.nc')

    data = lcm.copy()

    lon_list = np.linspace(data.longitude.min(), data.longitude.max(), LON_RESOLUTION)
    lat_list = np.linspace(data.latitude.min(), data.latitude.max(), LAT_RESOLUTION)

    map_data = data.time.rename('map_data')
    map_data = map_data.expand_dims({'map_lon': lon_list, 'map_lat': lat_list})
    map_data = map_data.to_dataset().drop_vars('map_data')
    map_data['longitude'] = map_data.groupby('map_lon').map(lambda x: x.map_lon)
    map_data['latitude'] = map_data.groupby('map_lat').map(lambda x: x.map_lat)
    map_data = map_data.stack(point=('map_lon', 'map_lat'))

    map_data['nearest_monitor'] = map_data.groupby('point').map(ord_dist, args=(data, 0, True))
    map_data['2nearest_monitor'] = map_data.groupby('point').map(ord_dist, args=(data, 1, True))

    a = ord_stat_pm10(map_data.isel(point=0), data, 0, stationless=True)

    map_data['nearest_monitor_pm10'] = map_data.groupby('point').map(ord_stat_pm10, args=(data, 0, True))
    map_data['2nearest_monitor_pm10'] = map_data.groupby('point').map(ord_stat_pm10, args=(data, 1, True))

    map_data['nearest_monitor_pm25'] = map_data.groupby('point').map(ord_stat_pm25, args=(data, 0, True))
    map_data['2nearest_monitor_pm25'] = map_data.groupby('point').map(ord_stat_pm25, args=(data, 1, True))

    map_data['nearest_monitor_pm1'] = map_data.groupby('point').map(ord_stat_pm1, args=(data, 0, True))
    map_data['2nearest_monitor_pm1'] = map_data.groupby('point').map(ord_stat_pm1, args=(data, 1, True))

    map_data['day_of_week'] = map_data.time.dt.dayofweek
    map_data['month'] = map_data.time.dt.month

    map_data['nearest_monitor_angle'] = map_data.groupby('point').map(ord_stat_angle, args=(data, 0, True))
    map_data['2nearest_monitor_angle'] = map_data.groupby('point').map(ord_stat_angle, args=(data, 1, True))

    era5 = xr.open_dataset(ERA5 + 'formatted/era5.nc')
    era5['wind_angle'] = np.degrees(np.arctan2(era5.u10, era5.v10))

    era5 = era5.sel(time=slice(map_data.time.min(),
                               map_data.time.max()))
    era5 = era5.sel(time=~era5.get_index('time').duplicated(keep='last'))
    wind_angle_central = era5['wind_angle'][:, 1, 1]
    wind_angle_central = wind_angle_central.drop_duplicates(dim='time', keep='last').squeeze().reset_coords(drop=True)
    map_data = xr.merge([map_data, wind_angle_central])
    map_data['nearest_monitor_angle_wind'] = calc_angle_concordance(map_data.wind_angle,
                                                                    map_data.nearest_monitor_angle)
    map_data['2nearest_monitor_angle_wind'] = calc_angle_concordance(map_data.wind_angle,
                                                                     map_data['2nearest_monitor_angle'])

    lri = map_data.groupby('point').map(calc_lri)
    map_data = xr.merge([map_data, lri])

    era5_varslist = list(era5.keys())
    sw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=0).rename({var: 'sw_' + var for var in era5_varslist}).reset_coords(drop=True)
    s_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=0).rename({var: 's_' + var for var in era5_varslist}).reset_coords(drop=True)
    se_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=0).rename({var: 'se_' + var for var in era5_varslist}).reset_coords(drop=True)
    w_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=1).rename({var: 'w_' + var for var in era5_varslist}).reset_coords(drop=True)
    c_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=1).rename({var: 'c_' + var for var in era5_varslist}).reset_coords(drop=True)
    e_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=1).rename({var: 'e_' + var for var in era5_varslist}).reset_coords(drop=True)
    nw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=0, y=2).rename({var: 'nw_' + var for var in era5_varslist}).reset_coords(drop=True)
    n_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=1, y=2).rename({var: 'n_' + var for var in era5_varslist}).reset_coords(drop=True)
    ne_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
        .isel(x=2, y=2).rename({var: 'ne_' + var for var in era5_varslist}).reset_coords(drop=True)

    map_data = xr.merge([map_data, sw_era5, s_era5, se_era5, w_era5,
                     c_era5, e_era5, e_era5, nw_era5, n_era5, ne_era5])

    map_data.unstack().to_netcdf(DATA_DIR + 'Features/map.nc')

    # # change to 0 for general application
    # lcm_training['nearest_monitor'] = lcm_training.groupby('station').map(ord_dist, args=(lcm_training, 1))
    # lcm_training['2nearest_monitor'] = lcm_training.groupby('station').map(ord_dist, args=(lcm_training, 2))
    #
    # lcm_validation['nearest_monitor'] = lcm_validation.groupby('station').map(ord_dist, args=(lcm_validation, 1))
    # lcm_validation['2nearest_monitor'] = lcm_validation.groupby('station').map(ord_dist, args=(lcm_validation, 2))
    #
    # lcm_training['nearest_monitor_pm10'] = lcm_training.groupby('station').map(ord_stat_pm10, args=(lcm_training, 1))
    # lcm_training['2nearest_monitor_pm10'] = lcm_training.groupby('station').map(ord_stat_pm10, args=(lcm_training, 2))
    #
    # lcm_validation['nearest_monitor_pm10'] = lcm_validation.groupby('station').map(ord_stat_pm10,
    #                                                                                args=(lcm_validation, 1))
    # lcm_validation['2nearest_monitor_pm10'] = lcm_validation.groupby('station').map(ord_stat_pm10,
    #                                                                                 args=(lcm_validation, 2))
    #
    # lcm_training['nearest_monitor_pm25'] = lcm_training.groupby('station').map(ord_stat_pm25, args=(lcm_training, 1))
    # lcm_training['2nearest_monitor_pm25'] = lcm_training.groupby('station').map(ord_stat_pm25, args=(lcm_training, 2))
    #
    # lcm_validation['nearest_monitor_pm25'] = lcm_validation.groupby('station').map(ord_stat_pm25,
    #                                                                                args=(lcm_validation, 1))
    # lcm_validation['2nearest_monitor_pm25'] = lcm_validation.groupby('station').map(ord_stat_pm25,
    #                                                                                 args=(lcm_validation, 2))
    #
    #
    #
    #
    #
    # lcm_training['nearest_monitor_pm1'] = lcm_training.groupby('station').map(ord_stat_pm1, args=(lcm_training, 1))
    # lcm_training['2nearest_monitor_pm1'] = lcm_training.groupby('station').map(ord_stat_pm1, args=(lcm_training, 2))
    #
    # lcm_training['day_of_week'] = lcm_training.time.dt.dayofweek
    # lcm_training['month'] = lcm_training.time.dt.month
    #
    # lcm_validation['nearest_monitor_pm1'] = lcm_validation.groupby('station').map(ord_stat_pm1,
    #                                                                               args=(lcm_validation, 1))
    # lcm_validation['2nearest_monitor_pm1'] = lcm_validation.groupby('station').map(ord_stat_pm1,
    #                                                                                args=(lcm_validation, 2))
    #
    # lcm_validation['day_of_week'] = lcm_validation.time.dt.dayofweek
    # lcm_validation['month'] = lcm_validation.time.dt.month
    #
    #
    #
    #
    #
    # lcm_training['nearest_monitor_angle'] = lcm_training.groupby('station').map(ord_stat_angle, args=(lcm_training, 1))
    # lcm_training['2nearest_monitor_angle'] = lcm_training.groupby('station').map(ord_stat_angle, args=(lcm_training, 2))
    #
    # lcm_validation['nearest_monitor_angle'] = lcm_validation.groupby('station').map(ord_stat_angle,
    #                                                                                 args=(lcm_validation, 1))
    # lcm_validation['2nearest_monitor_angle'] = lcm_validation.groupby('station').map(ord_stat_angle,
    #                                                                                  args=(lcm_validation, 2))
    #
    # era5 = xr.open_dataset(ERA5 + 'formatted/era5.nc')
    # era5['wind_angle'] = np.degrees(np.arctan2(era5.u10, era5.v10))
    # era5_validation = era5.copy()
    #
    # era5 = era5.sel(time=slice(lcm_training.time.min(),
    #                            lcm_training.time.max()))
    # era5 = era5.sel(time=~era5.get_index('time').duplicated(keep='last'))
    # wind_angle_central = era5['wind_angle'][:, 1, 1]
    # wind_angle_central = wind_angle_central.drop_duplicates(dim='time', keep='last').squeeze().reset_coords(drop=True)
    # lcm_training = xr.merge([lcm_training, wind_angle_central])
    # lcm_training['nearest_monitor_angle_wind'] = calc_angle_concordance(lcm_training.wind_angle,
    #                                                                     lcm_training.nearest_monitor_angle)
    # lcm_training['2nearest_monitor_angle_wind'] = calc_angle_concordance(lcm_training.wind_angle,
    #                                                                      lcm_training['2nearest_monitor_angle'])
    #
    # era5_validation = era5_validation.sel(time=slice(lcm_validation.time.min(),
    #                                                  lcm_validation.time.max()))
    # era5_validation = era5_validation.sel(time=~era5_validation.get_index('time').duplicated(keep='last'))
    # wind_angle_central = era5_validation['wind_angle'][:, 1, 1]
    # wind_angle_central = wind_angle_central.drop_duplicates(dim='time', keep='last').squeeze().reset_coords(drop=True)
    # lcm_validation = xr.merge([lcm_validation, wind_angle_central])
    # lcm_validation['nearest_monitor_angle_wind'] = calc_angle_concordance(lcm_validation.wind_angle,
    #                                                                       lcm_validation.nearest_monitor_angle)
    # lcm_validation['2nearest_monitor_angle_wind'] = calc_angle_concordance(lcm_validation.wind_angle,
    #                                                                        lcm_validation['2nearest_monitor_angle'])
    #
    #
    #
    #
    #
    # lri = lcm_training.groupby('station').map(calc_lri)
    # lcm_training = xr.merge([lcm_training, lri])
    #
    # lri = lcm_validation.groupby('station').map(calc_lri)
    # lcm_validation = xr.merge([lcm_validation, lri])
    #
    # era5_varslist = list(era5.keys())
    # sw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=0).rename({var: 'sw_' + var for var in era5_varslist}).reset_coords(drop=True)
    # s_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=0).rename({var: 's_' + var for var in era5_varslist}).reset_coords(drop=True)
    # se_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=0).rename({var: 'se_' + var for var in era5_varslist}).reset_coords(drop=True)
    # w_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=1).rename({var: 'w_' + var for var in era5_varslist}).reset_coords(drop=True)
    # c_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=1).rename({var: 'c_' + var for var in era5_varslist}).reset_coords(drop=True)
    # e_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=1).rename({var: 'e_' + var for var in era5_varslist}).reset_coords(drop=True)
    # nw_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=2).rename({var: 'nw_' + var for var in era5_varslist}).reset_coords(drop=True)
    # n_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=2).rename({var: 'n_' + var for var in era5_varslist}).reset_coords(drop=True)
    # ne_era5 = era5.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=2).rename({var: 'ne_' + var for var in era5_varslist}).reset_coords(drop=True)
    #
    # lcm_training = xr.merge([lcm_training, sw_era5, s_era5, se_era5, w_era5,
    #                          c_era5, e_era5, e_era5, nw_era5, n_era5, ne_era5])
    #
    # era5_validation_varslist = list(era5_validation.keys())
    # sw_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=0).rename({var: 'sw_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # s_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=0).rename({var: 's_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # se_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=0).rename({var: 'se_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # w_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=1).rename({var: 'w_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # c_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=1).rename({var: 'c_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # e_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=1).rename({var: 'e_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # nw_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=0, y=2).rename({var: 'nw_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # n_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=1, y=2).rename({var: 'n_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    # ne_era5 = era5_validation.rename({'longitude': 'x', 'latitude': 'y'}) \
    #     .isel(x=2, y=2).rename({var: 'ne_' + var for var in era5_validation_varslist}).reset_coords(drop=True)
    #
    # lcm_validation = xr.merge([lcm_validation, sw_era5, s_era5, se_era5, w_era5,
    #                            c_era5, e_era5, e_era5, nw_era5, n_era5, ne_era5])
    #
    # lcm_training.to_netcdf(DATA_DIR + 'Features/training.nc')
    # lcm_validation.to_netcdf(DATA_DIR + 'Features/validation.nc')
