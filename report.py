import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/WokingNetwork/report/'

if __name__ == '__main__':
    data = xr.open_dataset(DATA_DIR + 'Features/training.nc')
    validation = xr.open_dataset(DATA_DIR + 'Features/validation.nc')
    data = data.resample(time="1D", base=0).mean(skipna=True)
    validation = validation.resample(time="1D", base=0).mean(skipna=True)
    all_data = xr.concat([data, validation], dim='station').sortby('station')
    # station_vars = ['pm10', 'pm25', 'pm1', 'rh', 'temp', 'dew', 'wetbulb', 'longitude', 'latitude']
    station_vars = ['pm10', 'pm25', 'pm1']
    station_vars_ds = xr.concat([data[station_vars],
                                 validation[station_vars]],
                                dim='station').sortby('station')
    average_pm = station_vars_ds.mean(dim='station')
    dpm = station_vars_ds - average_pm
    average_dpm = dpm.mean(dim='time')
    list(average_dpm.pm10)

    ax = average_dpm.to_pandas().plot.barh()
    ax.set_xlabel('Local pollution (ug/m³)')
    plt.show()

    ax = dpm.pm10.rolling(time=5).median().transpose().to_pandas().plot()
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Local PM10 (ug/m³)')
    plt.show()

    dpmplot = dpm.pm10.rolling(time=5).median()
    dpmplot.to_pandas().plot.hist(logy=True, subplots=True, layout=(4, 2), figsize=(9, 7))
    plt.xlabel('Local PM10 (ug/m³)', position=(-0.1, 0))
    plt.show()

    ws = (all_data.c_u10 ** 2 + all_data.c_v10 ** 2) ** 0.5
    ws = ws.isel(station=0).reset_coords('station', drop=True)
    ws.plot.hist()
    plt.show()

    weak_wind = ws.where(ws < ws.quantile(0.25), drop=True).time
    strong_wind = ws.where(ws > ws.quantile(0.25), drop=True).time
    ww_data = all_data.sel(time=weak_wind)
    sw_data = all_data.sel(time=strong_wind)
    london_data = sw_data.where(sw_data.c_wind_angle < 90).where(sw_data.c_wind_angle > 0)
    other_data = sw_data.where(sw_data.c_wind_angle > 90)

    # pl = ww_data.pm1.transpose().to_pandas().plot.hist(logy=True, subplots=True, layout=(4, 2), figsize=(9, 7),
    #                                                    bins=25, alpha=0.5, legend = False)
    #
    # london_data.pm1.transpose().to_pandas().plot.hist(logy=True, subplots=True, layout=(4, 2), figsize=(9, 7),
    #                                                   bins=25, title='Local and London PM10 (ug/m³)', alpha=1,
    #                                                   ax=pl, legend=False)
    # station = 1
    # for i in [0, 1, 2, 3]:
    #     for j in [0, 1]:
    #         pl[i][j].text(0.5, 1, 'WokingGreens#' + str(station), horizontalalignment='center', verticalalignment='bottom', transform=pl[i][j].transAxes)
    #         pl[i][j].legend(['Local', 'London'])
    #         station = station + 1
    # plt.show()

    ww_df = ww_data.pm1.transpose().to_pandas()
    london_df = london_data.pm1.transpose().to_pandas()
    other_df = other_data.pm1.transpose().to_pandas()
    list_df = [ww_df, london_df, other_df]

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(9, 7))
    station = 1
    for i in [0, 1, 2, 3]:
        for j in [0, 1]:
            ww_df_station = ww_df['WokingGreens#' + str(station)]
            london_df_station = london_df['WokingGreens#' + str(station)]
            other_df_station = other_df['WokingGreens#' + str(station)]
            list_df = [ww_df_station, london_df_station, other_df_station]
            ax[i][j].hist(list_df, stacked=True, label=['Local', 'London', 'Other'],
                          bins=np.linspace(0, 50, 50))
            ax[i][j].legend()
            ax[i][j].text(0.5, 1, 'WokingGreens#' + str(station), horizontalalignment='center',
                          verticalalignment='bottom', transform=ax[i][j].transAxes)
            station = station + 1
    fig.suptitle('PM1 concentration (ug/m³)', y=0.02)
    fig.tight_layout()
    plt.show()

    station_vars_ds.groupby('station').mean('time')
    station_vars_ds.groupby('station').median('time')
    station_vars_ds.groupby('station').std('time')

    station_vars_ds_quarts = station_vars_ds.quantile([0, 0.25, 0.5, 0.75, 1], dim='time')
    monitors_n = [1, 2, 3, 4, 5, 6, 7, 8]
    for pm in ['pm1', 'pm25', 'pm10']:
        station_vars_ds_nquart = []
        for quart in [0, 0.25, 0.5, 0.75]:
            station_vars_ds_nquart.append(
                station_vars_ds[pm].where((station_vars_ds_quarts[pm].sel(quantile=quart) <=
                                           station_vars_ds[pm]) &
                                          (station_vars_ds[pm] <=
                                           station_vars_ds_quarts[pm].sel(quantile=quart + 0.25)),
                                          drop=True))

        fig, axes = plt.subplots(2, 2)
        for n, ax in enumerate(axes.flat):
            delta_matrix = np.abs(station_vars_ds_nquart[n] - station_vars_ds_nquart[n].rename(station='station_'))
            cmap = cm.get_cmap("viridis", 5)
            mat = ax.matshow(delta_matrix.mean('time'), cmap=cmap)
            plt.colorbar(mat, ax=ax, label='ug/m³')
            ax.title.set_text('Interquartile ' + str(n+1))
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(monitors_n)
            ax.set_yticks(np.arange(8))
            ax.set_yticklabels(labels=monitors_n)
        plt.suptitle(pm)
        plt.subplots_adjust(wspace=0.4,
                            hspace=0.4)
        plt.show()
