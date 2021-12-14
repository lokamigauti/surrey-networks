import numpy as np
from copy import deepcopy

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