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


if __name__ == '__main__':

    da = xr.open_dataarray(DATA_DIR + 'Imported/lcs.nc')
    calibration_params = calibration.import_json_as_dict(OUTPUT_DIR + LCS + 'calibration_parameters.json')

    #
    # da_calibrated = da.copy().rename('calibrated')
    # for pm in ['pm10', 'pm25', 'pm1']:
    #     cal = da_calibrated.groupby('station').map(calibrator, args=(pm, calibration_params)).copy().rename('case')
    #     da_calibrated = xr.concat([da_calibrated, cal], dim='variable')
    # da_calibrated = make_calibration(da, calibration_params, save=True)
    da_calibrated = xr.open_dataarray(OUTPUT_DIR + LCS + 'pm_calibrated.nc')
    mean_pm_per_station = da_calibrated.stack(station_pm=('station', 'variable')).groupby('station_pm').mean('time')
    mean_pm = mean_pm_per_station.unstack().mean('station')
    mean_pm.sel(variable='pm10_cal')
    mean_pm.sel(variable='pm25_cal')
    mean_pm.sel(variable='pm1_cal')

    da_pm = da.sel(variable=['pm10', 'pm25', 'pm1'])
    mean_pm_per_station_uncal = da_pm.stack(station_pm=('station', 'variable')).groupby('station_pm').mean('time')
    mean_pm_uncal = mean_pm_per_station_uncal.unstack().mean('station')
    mean_pm_uncal.sel(variable='pm10')
    mean_pm_uncal.sel(variable='pm25')
    mean_pm_uncal.sel(variable='pm1')

    median_pm_per_station = da_calibrated.stack(station_pm=('station', 'variable')).groupby('station_pm').median('time')
    median_pm = median_pm_per_station.unstack().median('station')
    median_pm.sel(variable='pm10_cal')
    median_pm.sel(variable='pm25_cal')
    median_pm.sel(variable='pm1_cal')
