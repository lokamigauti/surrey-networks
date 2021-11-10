import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'

da = xr.open_dataarray(DATA_DIR+'ERA5/download.nc')
da = da.assign_coords(longitude = (da.coords['longitude'].values+180)%360-180)
da = da.sortby('longitude')
fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
da.plot(ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()
da.sel(latitude=-23, longitude=-46, method='nearest').plot()

da = xr.open_dataset(DATA_DIR+'ERA5/wind_2021.nc')
da_toplot = da.sel(time='2021-08-03T23:00:00').isel(expver=0)\
    .coarsen(latitude=3, longitude=3, boundary='trim').mean()\
    .sortby('latitude')
fig, ax = plt.subplots(1, 1)
ax.streamplot(da_toplot.longitude.values,
              da_toplot.latitude.values,
              da_toplot['u10'].values,
              da_toplot['v10'].values)
# da_toplot.plot.streamplot(x='longitude', y='latitude', u='u10', v='v10', ax=ax)
ax.coastlines()
plt.show()

fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})

wind_mag = (da_toplot['u10'] ** 2 + da_toplot['v10'] ** 2) ** .5
fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
wind_mag.plot.contourf(levels=6, transform=ccrs.PlateCarree(), cmap='viridis')
da_toplot.plot.quiver(x='longitude', y='latitude', u='u10', v='v10', ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()