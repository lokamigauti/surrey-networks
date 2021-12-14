import xarray as xr
import pandas as pd
import tools
import glob
from datetime import datetime

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
METEOROLOGY = 'Heathrow/'
LCS = 'WokingNetwork/'


def make_da(filepaths):
    da_list = []
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    for filepath in filepaths:
        df = pd.read_csv(filepath,
                         date_parser=dateparse,
                         parse_dates=['time'],
                         index_col='time')
        da = df.to_xarray().to_array()
        da_list.append(da)
    da = xr.concat(da_list, dim='time')
    da_clear = tools.convert_to_float_and_replace_nan(da, deep_copy=True)
    da_clear.to_netcdf(DATA_DIR + 'Imported/meteorology.nc')
    da_clear.transpose().to_pandas().to_csv(OUTPUT_DIR + LCS + 'meteorology.csv')
    return da_clear


if __name__ == '__main__':
    filepaths = glob.glob(DATA_DIR + METEOROLOGY + 'export*')
    make_da(filepaths)
    print('End of execution')