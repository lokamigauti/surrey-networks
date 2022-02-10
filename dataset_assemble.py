import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'

if __name__ == '__main__':
    lcm_path = OUTPUT_DIR + LCS + 'data_calibrated.nc'
    lcm = xr.open_dataarray(lcm_path)
    lcm_meta_path = DATA_DIR + LCS + 'Woking Green Party deatils.csv'
    lcm_meta = pd.read_csv(lcm_meta_path)
    lcm_coords = lcm_meta[['Device/Sensor Name assigned', 'lat', 'lon']]. \
        rename(columns={'Device/Sensor Name assigned': 'station'}).set_index('station')
    lcm = lcm.to_dataset(dim='variable')
    lcm = lcm[['pm10_cal', 'pm25_cal', 'pm1_cal', 'rh', 'temp', 'dew', 'wetbulb']]
    lcm = lcm.rename_vars({'pm10_cal': 'pm10',
                           'pm25_cal': 'pm25',
                           'pm1_cal': 'pm1'})
    lcm['longitude'] = lcm_coords['lon']
    lcm['latitude'] = lcm_coords['lat']

    lcm_validation = lcm.sel(station=['WokingGreens#2', 'WokingGreens#7', 'WokingGreens#8'])
    lcm_training = lcm.sel(station=['WokingGreens#1', 'WokingGreens#3', 'WokingGreens#4',
                                    'WokingGreens#5', 'WokingGreens#6'])

    def ord_dist(lon_lat, lcm, n):
        dist = np.sqrt((lon_lat.longitude - lcm.longitude) ** 2 + (lon_lat.latitude - lcm.latitude) ** 2)
        dist = dist.sortby(dist)
        return dist[n]

    # change to 0 for general application
    lcm_training['nearest_monitor'] = lcm_training.groupby('station').map(ord_dist, args=(lcm_training, 1))
    lcm_training['2nearest_monitor'] = lcm_training.groupby('station').map(ord_dist, args=(lcm_training, 2))

    def ord_stat_pm10(lon_lat, lcm, n):
        stat = ord_dist(lon_lat, lcm, n)
        station_ = stat.station
        pm10 = lcm.sel(station=station_).pm10
        pm10['station'] = lon_lat.station
        return pm10

    lcm_training['nearest_monitor_pm10'] = lcm_training.groupby('station').map(ord_stat_pm10, args=(lcm_training, 1))
    lcm_training['2nearest_monitor_pm10'] = lcm_training.groupby('station').map(ord_stat_pm10, args=(lcm_training, 2))

    def ord_stat_pm25(lon_lat, lcm, n):
        stat = ord_dist(lon_lat, lcm, n)
        station_ = stat.station
        pm25 = lcm.sel(station=station_).pm25
        pm25['station'] = lon_lat.station
        return pm25

    lcm_training['nearest_monitor_pm25'] = lcm_training.groupby('station').map(ord_stat_pm25, args=(lcm_training, 1))
    lcm_training['2nearest_monitor_pm25'] = lcm_training.groupby('station').map(ord_stat_pm25, args=(lcm_training, 2))

    def ord_stat_pm1(lon_lat, lcm, n):
        stat = ord_dist(lon_lat, lcm, n)
        station_ = stat.station
        pm1 = lcm.sel(station=station_).pm1
        pm1['station'] = lon_lat.station
        return pm1

    lcm_training['nearest_monitor_pm1'] = lcm_training.groupby('station').map(ord_stat_pm1, args=(lcm_training, 1))
    lcm_training['2nearest_monitor_pm1'] = lcm_training.groupby('station').map(ord_stat_pm1, args=(lcm_training, 2))

    lcm_training['day_of_week'] = lcm_training.time.dt.dayofweek
    lcm_training['month'] = lcm_training.time.dt.month