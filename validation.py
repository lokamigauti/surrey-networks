import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

plt.style.use('bmh')

central_longitude = -0.540
central_latitude = 51.327
plot_extent = [-0.636, -0.456, 51.285, 51.350]

def separate_training_validation_test(dataset, method='cross'):

    return training, validation, test, meta