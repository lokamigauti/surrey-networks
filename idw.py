import cartopy.crs as ccrs
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

if __name__ == '__main__':
    pm_path = OUTPUT_DIR + LCS + 'pm_calibrated.nc'
    pm = xr.open_dataarray(pm_path)
    pm_meta_path = DATA_DIR + LCS + 'Woking Green Party deatils.csv'
    pm_meta = pd.read_csv(pm_meta_path)

    pm_coords = pm_meta[['Device/Sensor Name assigned', 'lat', 'lon']].\
        rename(columns={'Device/Sensor Name assigned': 'station'}).set_index('station')

    pm = pm.assign_coords({'lat': pm_coords.loc[pm.station.values].lat}).swap_dims({'station': 'lat'})\
        .expand_dims({'lon': pm_coords.loc[pm.station.values].lon}).sortby('lon').sortby('lat')

    # This is the map projection we want to plot *onto*
    map_proj = ccrs.LambertConformal(central_longitude=-0.540, central_latitude=51.327)
    p = pm.sel(variable='pm1_cal').isel(time=10000)
    p = p.plot(
        transform=ccrs.PlateCarree(),
        subplot_kws={"projection": map_proj},
    )

    p.axes.set_global()
    p.axes.coastlines()
    plt.show()