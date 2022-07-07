import xarray as xr
import numpy as np

from sklearn_xarray import wrap, RegressorWrapper, Target
from sklearn_xarray.preprocessing import Splitter, Sanitizer, Featurizer
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn_xarray.datasets import load_dummy_dataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


from sklearn import linear_model

import matplotlib.pyplot as plt

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'

X = xr.open_dataset(DATA_DIR + 'Features/training.nc').drop_vars(['pm10', 'pm25']).copy()


sani = Sanitizer(dim='time', groupby='station', group_dim='station')

sani.fit(X)

sani.transform(X)

y = Target(coord='pm1')(X)

reg = wrap(linear_model.LinearRegression, reshapes='station', sample_dim='time')

reg.fit(X, y)

if __name__ == '__main__':
    pl.fit(X, y)

