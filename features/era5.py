import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/ERA5/'
WOKING_OUTSKIRTS_LON_MIN = -0.6344602
WOKING_OUTSKIRTS_LON_MAX = -0.4544372
WOKING_OUTSKIRTS_LAT_MIN = 51.2844718
WOKING_OUTSKIRTS_LAT_MAX = 51.3494766
CENTRAL_LON = -0.540
CENTRAL_LON_ROUND = -0.5
CENTRAL_LAT = 51.327
CENTRAL_LAT_ROUND = 51.25

if __name__ == '__main__':
    # get era5T dataset
    filepaths = glob.glob(DATA_DIR + 'era5*')
    era5_list = []
    for filepath in filepaths:
        era5 = xr.open_dataset(filepath)
        era5_list.append(era5)
    era5 = xr.concat(era5_list, dim='time')
    era5t = era5.sel(expver=5)

    # set local time
    era5t_local_time_index = era5t.time.to_index()\
        .tz_localize(tz='UTC')\
        .tz_convert(tz='Europe/London')\
        .tz_localize(None)  # np.datetime do not handle timezones
    era5t_local = era5t.copy()
    era5t_local['time'] = era5t_local_time_index
    # daylight savings test
    # utc_test = era5t.sel(time=slice('2021-03-27', '2021-03-29'))
    # local_test = era5t_local.sel(time=slice('2021-03-27', '2021-03-29'))
    # utc_test.isel(longitude=1, latitude=1).u10.plot()
    # local_test.isel(longitude=1, latitude=1).u10.plot()
    # plt.show()

    # frame data to points inside the local and its neighbors
    era5t_local_frame = era5t_local.sortby('latitude', 'longitude')
    era5t_local_frame = era5t_local_frame.sel(longitude=slice(CENTRAL_LON_ROUND-0.25, CENTRAL_LON_ROUND+0.25),
                                              latitude=slice(CENTRAL_LAT_ROUND-0.25, CENTRAL_LAT_ROUND+0.25))

    # save netcdf
    era5t_local_frame.to_netcdf(DATA_DIR + 'formatted/era5.nc')