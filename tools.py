import numpy as np

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