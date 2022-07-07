import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pyinterp.backends.xarray
import pyinterp.fill
import winsound

import invdisttree as idt

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

    pm_pd = pm.isel(time=3000).to_pandas()
    pm_pd = pd.concat([pm_pd, pm_coords], axis=1).reset_index().set_index(['lat', 'lon'])

    lat_res = 150
    lon_res = 300

    # lat_res = 5
    # lon_res = 10

    request = cimgt.Stamen(style='terrain')
    map_proj = request.crs

    district_bounds = ShapelyFeature(Reader(DATA_DIR + 'DistrictBoundaries/Local_Authority_Districts_(May_2021)_UK_BGC').geometries(), ccrs.epsg(27700),
                                   linewidth=2, facecolor='none',
                                   edgecolor=(0.5, 0.5, 0.5, 1))

    roads_SU = ShapelyFeature(
        Reader(DATA_DIR + 'DistrictBoundaries/road_data/SU_RoadLink').geometries(),
        ccrs.epsg(27700),
        linewidth=1, facecolor='none',
        edgecolor=(0.5, 0.5, 0.5, 1))

    roads_TQ = ShapelyFeature(
        Reader(DATA_DIR + 'DistrictBoundaries/road_data/TQ_RoadLink').geometries(),
        ccrs.epsg(27700),
        linewidth=1, facecolor='none',
        edgecolor=(0.5, 0.5, 0.5, 1))

    # Plot 1 - Solo points

    lat_res_point = 15
    lon_res_point = 30

    p = pm_pd.to_xarray().pm25_cal

    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res_point, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res_point - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res_point, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res_point - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    fig, axes = plt.subplots(ncols=1, nrows=1, subplot_kw={'projection': map_proj})
    p_plot = p.plot(
        transform=ccrs.PlateCarree(),
        zorder=10,
        ax=axes,
        cbar_kwargs={'shrink': 0.75,
                     'anchor': (0, 0.49),
                     }
    )
    p_plot.axes.set_global()
    p_plot.axes.coastlines()
    p_plot.axes.set_extent([-0.636, -0.456, 51.285, 51.350])
    p_plot.axes.add_feature(district_bounds, alpha=0.2)
    p_plot.axes.add_feature(roads_SU, alpha=0.1)
    p_plot.axes.add_feature(roads_TQ, alpha=0.1)
    fig.suptitle('Interpolation Method: None',
                 x=0.022,
                 y=0.8,
                 ha='left',
                 )
    size = fig.get_size_inches() * fig.dpi
    axes.set_title('Contains OS data © Crown copyright and database right 2022',
                   fontdict={'fontsize': 7,
                             'verticalalignment': 'bottom'},
                   loc='right',
                   y=-0.1,
                   )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'BasicModels/none.png', dpi=300)
    plt.show()

    # Plot 2 - Loess

    p = pm_pd.to_xarray().pm25_cal
    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)
    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    grid = pyinterp.backends.xarray.Grid2D(p, geodetic=False)
    filled = pyinterp.fill.loess(grid, nx=50, ny=50)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=map_proj)
    lons, lats = np.meshgrid(grid.x, grid.y, indexing='ij')
    pcm = ax.pcolormesh(lons,
                        lats,
                        filled,
                        cmap='viridis',
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        )
    fig.suptitle('Interpolation Method: LOESS',
                 x=0.022,
                 y=0.8,
                 ha='left',
                 )
    ax.set_title('Contains OS data © Crown copyright and database right 2022',
                   fontdict={'fontsize': 7,
                             'verticalalignment': 'bottom'},
                   loc='right',
                   y=-0.1,
                   )
    ax.set_extent([-0.636, -0.456, 51.285, 51.350])
    ax.add_feature(district_bounds, alpha=0.2)
    ax.add_feature(roads_SU, alpha=0.1)
    ax.add_feature(roads_TQ, alpha=0.1)
    #ax.add_image(request, 13)
    plt.tight_layout()
    fig.colorbar(pcm,
                 ax=[ax],
                 shrink=0.75,
                 anchor=(0, 0.49),
                 ).set_label('pm25')
    plt.savefig(OUTPUT_DIR + 'BasicModels/loess.png', dpi=300)
    plt.show()

    # Plot 3 - Gauss-Seidel

    p = pm_pd.to_xarray().pm25_cal
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
                        )
    fig.suptitle('Interpolation Method: Gauss-Seidel',
                 x=0.022,
                 y=0.8,
                 ha='left',
                 )
    ax.set_title('Contains OS data © Crown copyright and database right 2022',
                 fontdict={'fontsize': 7,
                           'verticalalignment': 'bottom'},
                 loc='right',
                 y=-0.1,
                 )
    ax.set_extent([-0.636, -0.456, 51.285, 51.350])
    # ax.add_image(request, 13)
    ax.add_feature(district_bounds, alpha=0.2)
    ax.add_feature(roads_SU, alpha=0.1)
    ax.add_feature(roads_TQ, alpha=0.1)
    plt.tight_layout()
    fig.colorbar(pcm,
                 ax=[ax],
                 shrink=0.75,
                 anchor=(0, 0.49),
                 ).set_label('pm25')
    plt.savefig(OUTPUT_DIR + 'BasicModels/gs.png', dpi=300)
    plt.show()

    # Plot 4 - IDT

    p = pm_pd.to_xarray().pm25_cal

    lat_bins, lat_step = np.linspace(p.lat.min(), p.lat.max(), lat_res, retstep=True)
    lat_centres = np.linspace(p.lat.min() + (lat_step / 2), p.lat.max() - (lat_step / 2), lat_res - 1)

    lon_bins, lon_step = np.linspace(p.lon.min(), p.lon.max(), lon_res, retstep=True)
    lon_centres = np.linspace(p.lon.min() + (lon_step / 2), p.lon.max() + (lon_step / 2), lon_res - 1)

    p = p.groupby_bins('lat', lat_bins, labels=lat_centres, include_lowest=True).mean() \
        .groupby_bins('lon', lon_bins, labels=lon_centres, include_lowest=True).mean() \
        .rename({'lat_bins': 'lat',
                 'lon_bins': 'lon'})

    N = 10000
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 4  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    pot = 1  # weights ~ 1 / distance**p
    cycle = .25
    seed = 1

    np.random.seed(seed)

    stations_lon_lat = [[value[1], value[0]] for value in pm_pd.index.values]

    lat_list = [value for value in p.indexes['lat']]
    lon_list = [value for value in p.indexes['lon']]

    lon_lat = [[lon, lat] for lat in lat_list for lon in lon_list]

    invdisttree = idt.Invdisttree(stations_lon_lat, pm_pd.pm25_cal.values, leafsize=leafsize, stat=1)
    interpol = invdisttree(np.array(lon_lat), nnear=Nnear, eps=eps, p=pot)
    interpol = interpol.reshape([lat_res-1, lon_res-1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=map_proj)
    pcm = ax.pcolormesh(lon_list,
                        lat_list,
                        interpol,
                        cmap='viridis',
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        #alpha=0.75
                        )
    fig.suptitle('Interpolation Method: IDW',
                 x=0.022,
                 y=0.8,
                 ha='left',
                 )
    ax.set_title('Contains OS data © Crown copyright and database right 2022',
                 fontdict={'fontsize': 7,
                           'verticalalignment': 'bottom'},
                 loc='right',
                 y=-0.1,
                 )
    ax.set_extent([-0.636, -0.456, 51.285, 51.350])
    ax.add_feature(district_bounds, alpha=0.2)
    ax.add_feature(roads_SU, alpha=0.1)
    ax.add_feature(roads_TQ, alpha=0.1)
    plt.tight_layout()
    fig.colorbar(pcm,
                 ax=[ax],
                 shrink=0.75,
                 anchor=(0, 0.49),
                 ).set_label('pm25')
    plt.savefig(OUTPUT_DIR + 'BasicModels/idw.png', dpi=300)
    plt.show()

    a=1