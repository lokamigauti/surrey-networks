# Resample data scripts.
import numpy as np
import pandas as pd
import xarray as xr

import tools

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'

if __name__ == '__main__':
    # resample pm hourly
    pm_path = OUTPUT_DIR + LCS + 'data_calibrated.nc'
    pm = xr.open_dataarray(pm_path)
    pm_hourly = tools.resample(pm, '1H')
    pm_hourly.to_netcdf(OUTPUT_DIR + LCS + 'data_hourly.nc')

    # grid pm
    pm_lat_lon = tools.stations_to_lat_lon(OUTPUT_DIR + LCS + 'data_hourly.nc',
                                           DATA_DIR + LCS + 'Woking Green Party deatils.csv',
                                           save=True,
                                           save_path=OUTPUT_DIR + LCS + 'data_lat_lon.nc')
    pm_lat_lon