import xarray as xr

if __name__ == '__main__':
    DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
    OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
    LCS = 'WokingNetwork/'
    WIND = 'Heathrow/'
    ERA5 = 'ERA5/'

    lcs = xr.open_dataset(OUTPUT_DIR + LCS + 'data_lat_lon.nc')
    wind = xr.open_dataarray(DATA_DIR + WIND + 'wind.nc')
    era5 = xr.open_dataset(DATA_DIR + ERA5 + 'era5_2021.nc')

    wind = wind.rename('local_wind').to_dataset(dim='variable')\
        .rename({name: 'heathrow_' + name for name in wind.indexes['variable']})

    era5 = era5.sel(expver=5).rename({'longitude': 'lon', 'latitude':'lat'}).reset_coords(names='expver', drop=True)

    merge = xr.merge([era5, wind, lcs], compat='equals')

    merge.to_netcdf(DATA_DIR + '/Timeseries/time_series.nc')