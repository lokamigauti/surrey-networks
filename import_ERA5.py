#!/usr/bin/env python
import numpy as np
import cdsapi
DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/ERA5/'

def retrieve_era5(c, year, months, days, hours, params, area = [0, 0, 0, 0]):
    if area == [0, 0, 0, 0]:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'param': params,
                'year': year,
                'month': months,
                'day': days,
                'time': hours,
                'format': 'netcdf',
            },
            DATA_DIR + f'era5_{year}.nc')
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': params,
            'year': year,
            'month': months,
            'day': days,
            'time': hours,
            'format': 'netcdf',
            'area': area
        },
        DATA_DIR + f'era5_{year}.nc')



if __name__ == '__main__':
    c = cdsapi.Client()

    params = [
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
        'boundary_layer_dissipation', 'boundary_layer_height', 'forecast_surface_roughness', 'low_cloud_cover',
        'soil_type', 'surface_pressure', 'total_cloud_cover', 'total_precipitation',
    ]

    UK_data_window = [
        61, -21, 46, 12
    ]

    hours = [f'{x:02d}' + ':00' for x in np.arange(0, 24)]
    months = [f'{x:02d}' for x in np.arange(1, 13)]
    days = [f'{x:02d}' for x in np.arange(1, 32)]
    years = [2021]
    for year in years:
        retrieve_era5(c, year, months, days, hours, params, UK_data_window)

