import xarray as xr
from tools import calc_angle_concordance

ERA5 = 'G:/My Drive/IC/Doutorado/Sandwich/Data/ERA5/'
PFI = 'G:/My Drive/IC/Doutorado/Sandwich/Data/Features/'

if __name__ == '__main__':
    era5 = xr.open_dataset(ERA5 + 'formatted/era5.nc')
    era5['wind_angle'] = np.degrees(np.arctan2(era5.u10, era5.v10))
    calc_angle_concordance(era5.wind_angle, 10, PFI + 'pfi.nc')
