import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import winsound

beep_duration = 1000  # milliseconds
beep_freq = 440  # Hz

plt.style.use('bmh')

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'

central_longitude = -0.540
central_latitude = 51.327

if __name__ == '__main__':
    pm_path = OUTPUT_DIR + LCS + 'data_calibrated.nc'
    pm = xr.open_dataarray(pm_path)
    pm_meta_path = DATA_DIR + LCS + 'Woking Green Party deatils.csv'
    pm_meta = pd.read_csv(pm_meta_path)

    pm_coords = pm_meta[['Device/Sensor Name assigned', 'lat', 'lon']]. \
        rename(columns={'Device/Sensor Name assigned': 'station'}).set_index('station')

    pm_pd = pm.isel(time=30000).to_pandas()
    pm_pd = pd.concat([pm_pd, pm_coords], axis=1).reset_index().set_index(['lat', 'lon'])

    lat_res = 15
    lon_res = 30

    # map_proj = ccrs.LambertConformal(central_longitude=central_longitude, central_latitude=central_latitude)
    request = cimgt.Stamen(style='terrain')
    map_proj = request.crs

    p = pm_pd.to_xarray().pm1_cal

    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)


    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean()\
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean()\
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    p = p.plot(
        transform=ccrs.PlateCarree(),
        subplot_kws={'projection': map_proj},
    )

    p.axes.set_global()
    p.axes.coastlines()
    p.axes.set_extent([-0.636, -0.456, 51.285, 51.380])
    p.axes.add_image(request, 13)
    plt.show()