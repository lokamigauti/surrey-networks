import numpy as np
import xarray as xr
import pandas as pd
import tools
import glob
import re
from io import StringIO

from datetime import datetime

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'

ERA5 = 'ERA5/wind_2021.nc'

LCS = 'WokingNetwork/'


def make_da(filepaths):
    da_list = []
    dateparse = lambda x: datetime.strptime(x, '%d/%m/%y %H:%M')
    for filepath in filepaths:
        file_raw = open(filepath, 'r').read()
        pattern = re.compile('[^"]+')
        station = pattern.search(file_raw[:file_raw.index("\n")]).group()
        df = pd.read_csv(StringIO(file_raw),
                         skiprows=6,
                         date_parser=dateparse,
                         names=['time', 'aqi', 'h_aqi', 'pm1', 'h_pm1', 'pm25', 'h_pm25', 'pm10', 'h_pm10', 'temp',
                                'h_temp', 'l_temp', 'rh', 'h-rh', 'l-rh', 'dew', 'h_dew', 'l_dew', 'wetbulb',
                                'h_wetbulb', 'l_wetbulb', 'heatindex', 'h_heatindex'],
                         parse_dates=['time'],
                         index_col='time')
        da = df.to_xarray().to_array()
        da['station'] = station
        da = da.expand_dims(dim='station')
        da_list.append(da)
    da = xr.concat(da_list, dim='station')
    da_clear = tools.convert_to_float_and_replace_nan(da, deep_copy=True)
    da_clear.to_netcdf(DATA_DIR + 'Imported/lcs.nc')
    return da_clear


if __name__ == '__main__':
    filepaths = glob.glob(DATA_DIR + LCS + 'WokingGreens*')

    da = make_da(filepaths)
