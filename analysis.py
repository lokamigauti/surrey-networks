import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import calibration
import numpy as np

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'

# da = xr.open_dataarray(DATA_DIR+'ERA5/download.nc')
# da = da.assign_coords(longitude = (da.coords['longitude'].values+180)%360-180)
# da = da.sortby('longitude')
# fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
# da.plot(ax=ax, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()
# da.sel(latitude=-23, longitude=-46, method='nearest').plot()
#
# da = xr.open_dataset(DATA_DIR+'ERA5/wind_2021.nc')
# da_toplot = da.sel(time='2021-08-03T23:00:00').isel(expver=0)\
#     .coarsen(latitude=3, longitude=3, boundary='trim').mean()\
#     .sortby('latitude')
# fig, ax = plt.subplots(1, 1)
# ax.streamplot(da_toplot.longitude.values,
#               da_toplot.latitude.values,
#               da_toplot['u10'].values,
#               da_toplot['v10'].values)
# # da_toplot.plot.streamplot(x='longitude', y='latitude', u='u10', v='v10', ax=ax)
# ax.coastlines()
# plt.show()
#
# fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
#
# wind_mag = (da_toplot['u10'] ** 2 + da_toplot['v10'] ** 2) ** .5
# fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
# wind_mag.plot.contourf(levels=6, transform=ccrs.PlateCarree(), cmap='viridis')
# da_toplot.plot.quiver(x='longitude', y='latitude', u='u10', v='v10', ax=ax, transform=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()

def plot_era5_wind(data, sel_time, sel_title):
    da_toplot = data.sel(time=sel_time)\
        .sortby('longitude').sel(longitude=slice(-2.4, 2)).sortby('latitude').sel(latitude=slice(50, 52))

    ax = plt.subplot(projection=ccrs.Orthographic(-0.554106, 51.318604))
    p = da_toplot.plot.quiver(
        x='longitude', y='latitude',
        u='u10', v='v10',
        subplot_kws=dict(projection=ccrs.Orthographic(-0.554106, 51.318604),
                         facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.coastlines()
    plt.title(sel_title)
    plt.show()


if __name__ == '__main__':
    other_vars = xr.open_dataarray(DATA_DIR + 'Imported/lcs.nc')
    calibration_params = calibration.import_json_as_dict(OUTPUT_DIR + LCS + 'calibration_parameters.json')
    pm = xr.open_dataarray(OUTPUT_DIR + LCS + 'pm_calibrated.nc')

    mean_pm_per_station = pm.stack(station_pm=('station', 'variable')).groupby('station_pm').mean('time')
    mean_pm = mean_pm_per_station.unstack().mean('station')
    mean_pm.sel(variable='pm10_cal')
    mean_pm.sel(variable='pm25_cal')
    mean_pm.sel(variable='pm1_cal')

    std_pm_per_station = pm.stack(station_pm=('station', 'variable')).groupby('station_pm').std('time')
    std_pm = std_pm_per_station.unstack().mean('station')
    std_pm.sel(variable='pm10_cal')
    std_pm.sel(variable='pm25_cal')
    std_pm.sel(variable='pm1_cal')

    pm_daily = pm.resample(time='1d').mean().mean(dim='station')
    max_pm = pm_daily.idxmax(dim='time')

    max_da = pm_daily.copy()
    n_maxes = 3
    higher_days = {}
    for n_higher in range(1, n_maxes + 1):
        max_pm = max_da.idxmax(dim='time')
        higher_days[n_higher] = dict(zip(max_pm.coords['variable'].values, max_pm.values))
        for pm_size, timestamp in zip(max_pm.coords['variable'].values, max_pm.values):
            max_da.loc[pm_size, timestamp] = np.nan

    higher_days

    sec_max_pm = pm_daily.copy()
    sec_max_pm.loc['pm10_cal', '2021-09-10T00:00:00.000000000'] = np.nan

    era5 = xr.open_dataset(DATA_DIR + 'ERA5/era5_2021.nc').isel(expver=0)
    era5 = era5.resample(time='1d').mean(skipna=True)
    era5 = era5.assign_coords(longitude=(era5.coords['longitude'].values + 180) % 360 - 180)
    era5 = era5.sortby('longitude')

    for n_high, pm_times in higher_days.items():
        for pm_size in pm_times:
            title = f'{n_high} max of {pm_size}, day {pm_times[pm_size]}'
            plot_era5_wind(era5, pm_times[pm_size], title)

    # plot test

    da_toplot = era5.sel(time='2021-08-03T00:00:00.000000000')\
        .sortby('longitude').sel(longitude=slice(-2.4, 2)).sortby('latitude').sel(latitude=slice(50, 52))

    ax = plt.subplot(projection=ccrs.Orthographic(-0.554106, 51.318604))
    p = da_toplot.plot.quiver(
        x='longitude', y='latitude',
        u='u10', v='v10',
        subplot_kws=dict(projection=ccrs.Orthographic(-0.554106, 51.318604),
                         facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.coastlines()
    plt.title('abc')
    plt.show()

    # mean wind

    era5_mean = era5.sortby('time').sel(time=slice(pm.time.min(), pm.time.max())).mean(dim='time')
    da_toplot = era5_mean\
        .sortby('longitude').sel(longitude=slice(-2.4, 2)).sortby('latitude').sel(latitude=slice(50, 52))
    da_toplot = da_toplot.assign(wind_mag=(da_toplot['u10'] ** 2 + da_toplot['v10'] ** 2) ** .5)
    wind_mag = (da_toplot['u10'] ** 2 + da_toplot['v10'] ** 2) ** .5

    ax = plt.subplot(projection=ccrs.Orthographic(-0.554106, 51.318604))
    #ax.set_style('dark_background')
    wind_mag.plot.contourf(levels=30, transform=ccrs.PlateCarree(), cmap='viridis', ax=ax,
                           cbar_kwargs={'label': 'Wind speed (m/s)'})
    p = da_toplot.plot.quiver(
        x='longitude', y='latitude',
        u='u10', v='v10',
        color='w',
        subplot_kws=dict(projection=ccrs.Orthographic(-0.554106, 51.318604),
                         facecolor="gray"),
        transform=ccrs.PlateCarree(),
    )
    p.axes.coastlines(color='w')
    plt.title('Mean wind')
    plt.savefig(OUTPUT_DIR + LCS + 'mean_wind.png')
    plt.show()
