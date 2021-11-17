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


def calibrator(data, target, calibration_params):
    X = data.sel(variable=[target, 'rh', 'temp']).values.copy()
    if data.station.values.shape == ():
        station_ = data.station.values.tolist()
        y = calibration.calibrate(X.transpose(), calibration_params[target][station_]).copy()
    else:
        station_ = data.station.values[0]
        y = calibration.calibrate(X[0].transpose(), calibration_params[target][station_]).copy()
    da = xr.DataArray(
        y.reshape(-1, 1),
        coords=[('time', data.time.values.copy()), ('variable', [target+'_cal'])])
    da = da.astype('float32')
    return da


if __name__ == '__main__':
    # calibration plots

    da = xr.open_dataarray(DATA_DIR + 'Imported/lcs.nc')
    calibration_params = calibration.open_calibration_data(OUTPUT_DIR + LCS + 'calibration_parameters.json')
    # target = 'pm10'
    # station = 'WokingGreens#5'
    # X = da.sel(station=station, variable=[target, 'rh', 'temp']).values.copy()
    # X_cal = calibration.calibrate(X.transpose(), calibration_params[target][station]).copy()
    # calibration.calibrate(X.transpose()[0], calibration_params[target][station])
    da_calibrated = da.copy().rename('calibrated')
    for pm in ['pm10', 'pm25', 'pm1']:
        cal = da_calibrated.groupby('station').map(calibrator, args=(pm, calibration_params)).copy().rename('case')
        da_calibrated = xr.concat([da_calibrated, cal], dim='variable')

    da_calibrated.
