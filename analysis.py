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
da.isel(time=0).coarsen(latitude=3, longitude=3, boundary='trim').mean().sortby('latitude').plot.quiver(x='longitude', y='latitude', u='u10', v='v10')
plt.show()