import numpy as np
import xarray as xr

ERA5 = 'G:/My Drive/IC/Doutorado/Sandwich/Data/ERA5/'
PFI = 'G:/My Drive/IC/Doutorado/Sandwich/Data/Features/'

def calc_pfi(era5, pfar_angle, output_path):

    era5['wind_angle'] = np.degrees(np.arctan2(era5.u10, era5.v10))
    phi = (era5.wind_angle - pfar_angle)
    phi = abs(phi)
    pfi = xr.where(phi > 180, phi-2*(phi-180), phi)

    pfi.to_netcdf(output_path)
    return pfi

if __name__ == '__main__':
    era5 = xr.open_dataset(ERA5 + 'formatted/era5.nc')

    calc_pfi(era5, 10, PFI + 'pfi.nc')