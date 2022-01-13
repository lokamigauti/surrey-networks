import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pyinterp.backends.xarray
import pyinterp.fill
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

    pm_pd = pm.isel(time=31000).to_pandas()
    pm_pd = pd.concat([pm_pd, pm_coords], axis=1).reset_index().set_index(['lat', 'lon'])

    lat_res = 150
    lon_res = 300

    request = cimgt.Stamen(style='terrain')
    map_proj = request.crs

    # Plot 1 - Solo points

    p = pm_pd.to_xarray().pm1_cal

    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    p = p.plot(
        transform=ccrs.PlateCarree(),
        subplot_kws={'projection': map_proj},
        alpha=0.75
    )

    p.axes.set_global()
    p.axes.coastlines()
    p.axes.set_extent([-0.636, -0.456, 51.285, 51.350])
    p.axes.add_image(request, 13)
    p.axes.set_title("Interpol: None")
    plt.show()

    # Plot 2 - Loess

    p = pm_pd.to_xarray().pm1_cal
    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    grid = pyinterp.backends.xarray.Grid2D(p, geodetic=False)
    filled = pyinterp.fill.loess(grid, nx=3, ny=3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=map_proj)
    lons, lats = np.meshgrid(grid.x, grid.y, indexing='ij')
    pcm = ax.pcolormesh(lons,
                        lats,
                        filled,
                        cmap='viridis',
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        alpha=0.75
                        )
    ax.set_title("Interpol: LOESS")
    ax.set_extent([-0.636, -0.456, 51.285, 51.350])
    ax.add_image(request, 13)
    fig.colorbar(pcm, ax=[ax]).set_label('pm1')
    fig.show()

    # Plot 3 - Gauss-Seidel

    p = pm_pd.to_xarray().pm1_cal
    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    grid = pyinterp.backends.xarray.Grid2D(p, geodetic=False)
    has_converged, filled = pyinterp.fill.gauss_seidel(grid)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=map_proj)
    lons, lats = np.meshgrid(grid.x, grid.y, indexing='ij')
    pcm = ax.pcolormesh(lons,
                        lats,
                        filled,
                        cmap='viridis',
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        alpha=0.75
                        )
    ax.set_title("Interpol: Gauss-Seidel")
    ax.set_extent([-0.636, -0.456, 51.285, 51.350])
    ax.add_image(request, 13)
    fig.colorbar(pcm, ax=[ax]).set_label('pm1')
    fig.show()