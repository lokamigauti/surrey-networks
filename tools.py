import numpy as np
from copy import deepcopy
import xarray as xr
import pandas as pd

def size_in_memory(da):
    """
    Check xarray (dask or not) size in memory without loading
    Parameters.
    From: gabrielmpp - github.
    ----------
    da
    -------
    """
    if da.dtype == 'float64':
        size = 64
    elif da.dtype == 'float32':
        size = 32
    else:
        raise TypeError('array dtype not recognized.')

    n_positions = np.prod(da.shape)
    total_size = size * n_positions / (8 * 1e9)
    print(f'Total array size is: {total_size} GB')
    return None

def convert_to_float_and_replace_nan(da, deep_copy=False, precision=32):
    """
    Intended to use with imports
    :param da:
    :param deep_copy:
    :param precision:
    :return:
    """
    if deep_copy:
        da = da.copy()

    data_temp = da.values.copy()

    original_shape = deepcopy(data_temp.shape)

    data_temp = data_temp.flatten()

    for idx, value in enumerate(data_temp):
        try:
            data_temp[idx] = float(value)
        except ValueError:
            data_temp[idx] = np.nan
    data_temp = data_temp.reshape(original_shape)
    da = da.copy(data=data_temp)
    da = da.astype(f'float{precision}')
    return da

def flatten_data(da, sample_dim='time', feature_dim='variable', output_path='none'):
    """
    Transform DataArray into a flat DataFrame. Saves a file if output_dir is provided.
    :param da: 3D+DataArray
    :param sample_dim: String with name of sample dimension
    :param feature_dim: String with name of feature dimension
    :param output_path: String with output path
    :return: Simple-index DataFrame
    """
    dims_to_features = set(da.dims) - {sample_dim, feature_dim}
    dims_to_features = np.array(tuple(dims_to_features))
    df = da.stack(station_variable=np.append(dims_to_features, feature_dim)).to_pandas()
    df = df.reset_index().melt(id_vars=sample_dim)
    df = df.pivot_table(index=np.append(sample_dim, dims_to_features).tolist()
                        , columns=[feature_dim]
                        , values=['value'])
    df = df.droplevel(0, axis=1).reset_index(level=dims_to_features)
    if not output_path == 'none':
        df.to_csv(output_path)
    return df

def resample(da, delta, upsampling=False, time_dim_name='time'):
    """
    Perform resample in a DataArray given the name of the time dim
    :param da: xr.DataArray
    :param delta: resampling interval
    :param upsampling: Is the operation an Upsampling?
    :param time_dim_name: Name of the time dimension. Labels need to be in datetime format.
    :return: resampled DataArray
    """
    da.rename({time_dim_name: 'time'})
    if upsampling:
        return da.resample(time=delta, skipna=True, closed={"left", "right"}).interpolate('linear')
    return da.resample(time=delta, skipna=True).mean()

def stations_to_lat_lon(pm_path, meta_path, meta_station_col_name='Device/Sensor Name assigned', save=False, save_path=''):
    pm = xr.open_dataarray(pm_path)
    pm_meta = pd.read_csv(meta_path)
    pm_coords = pm_meta[[meta_station_col_name, 'lat', 'lon']]. \
        rename(columns={meta_station_col_name: 'station'}).set_index('station')
    da_list = []
    for time in range(pm.time.__len__()):
        pm_time = pm.isel(time=time)
        pm_pd = pm_time.to_pandas()
        pm_pd = pd.concat([pm_pd, pm_coords], axis=1)
        pm_pd['time'] = pm_time.time.values
        pm_pd = pm_pd.reset_index().set_index(['lat', 'lon', 'time'])
        da = pm_pd.to_xarray()
        da_list.append(da)
    p = xr.concat(da_list, dim='time')
    if save:
        p.to_netcdf(save_path)
    return p

def calc_angle_concordance(wind_angle, target_angle, output_path=None):
    phi = (wind_angle - target_angle)
    phi = abs(phi)
    pfi = xr.where(phi > 180, phi - 2 * (phi - 180), phi)
    pfi = pfi / 180

    if output_path is not None:
        pfi.to_netcdf(output_path)
    return pfi