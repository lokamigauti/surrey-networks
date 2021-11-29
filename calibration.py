import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import json

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
OUTPUT_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Output/'
LCS = 'WokingNetwork/'


class LeoMetrics:
    metrics_dict = {
        'r2': r2_score,
        'mse': mean_squared_error,
        'mape': mean_absolute_percentage_error
    }

    def __init__(self, metric, sampledim='time'):
        assert metric in self.metrics_dict.keys(), "Metric not founded"
        self.metric = metric
        self.sampledim = sampledim

    def apply(self, X, y, copy=True):
        """

        :param copy:
        :param X: ND DataArray
        :param y: ND DataArray
        :return: ND-1 array with the required metric (sample dimension will be su´´ressed)
        """
        if copy:
            X = X.copy()
            y = y.copy()

        from copy import deepcopy
        assert len(X.shape) == len(y.shape) + 1, "Adadds"
        x_feature_dims = list(deepcopy(X.dims))
        x_extra_dim = [d for d in x_feature_dims if d not in y.dims][0]
        y_expanded = []
        for extra_coord in X[x_extra_dim]:
            y_expanded.append(
                y.copy()
            )
        y_expanded = xr.concat(y_expanded, dim=X[x_extra_dim])

        X_p = self.preprocess(X)
        y_p = self.preprocess(y_expanded)

        f = self.metrics_dict[self.metric]
        da_metric = X_p.isel({self.sampledim: 0}).drop(self.sampledim) \
            .copy(data=f(y_p.values, X_p.values, multioutput='raw_values'))
        return da_metric.unstack()

    def preprocess(self, da):
        stacked_features_list = list(da.dims)
        stacked_features_list.remove(self.sampledim)
        da = da.stack(stacked_features=stacked_features_list).transpose(self.sampledim, ...)
        return da


def test_ridge(X_train, y_train, X_test, y_test, alpha, plot=False):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = reg.predict(X_test_scaled)

    coef = reg.coef_
    intercept = reg.intercept_

    if plot:
        fig, ax = plt.subplots()
        ax.plot(y_train, X_train[:, 0], '.', label='sensor')
        ax.plot(y_train, reg.predict(X_train_scaled), '.', label='calibration')
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, '--')
        ax.set_title(f'train{alpha}')
        ax.legend(loc='upper left')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(y_test, X_test[:, 0], '.', label='sensor')
        ax.plot(y_test, y_pred, '.', label='calibration')
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, '--')
        ax.set_title(f'{alpha}')
        ax.legend(loc='upper left')
        plt.show()

    return coef, intercept


def test_ridge_poly(X_train, y_train, X_test, y_test, alpha, degree, plot=False):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    coef, intercept = test_ridge(X_train_poly, y_train, X_test_poly, y_test, alpha, plot=True)

    return coef, intercept


def calibration_data_import(path):
    station_names = []
    for station_number in range(1, 9):
        station_names.append(f'S{station_number}')

    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df.columns = df.columns.str.split("_", expand=True)
    df = df.rename(columns={'T': 'temp',
                            'Temp': 'temp',
                            'RH': 'rh',
                            'PM10': 'pm10',
                            'PM2.5': 'pm25',
                            'PM1': 'pm1'})
    df = df.stack(level=0).reset_index(level=1).rename(columns={"level_1": "station"})
    df = df.set_index([df.index, 'station'])
    ds = df.to_xarray()
    ds.to_netcdf(DATA_DIR + 'Imported/calibration.nc')
    da = ds.to_array()
    da = da.transpose('time', 'station', 'variable')
    return da


def apply_poly_ridge(X, y, degree, alpha):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    scaler = StandardScaler().fit(X_poly)
    X_poly_scaled = scaler.transform(X_poly)

    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_poly_scaled, y)

    return_dict = {'coef': reg.coef_,
                   'intercept': reg.intercept_,
                   'mean': scaler.mean_,
                   'std': np.sqrt(scaler.var_),
                   'poly_degree': degree}

    return return_dict


def apply_model_on_dimensions(X, y, model_dims=['station'], **regression_kwargs):
    """

    :param X:
    :param y:
    :param model_dims:
    :param regression_kwargs:
    :return:
    """
    X = X.copy()
    y = y.copy()

    if len(model_dims) > 1:
        X = X.stack(model_dim=model_dims)
    else:
        X = X.rename({model_dims[0]: 'model_dim'})

    params_dict = {}
    for dict_key in X.model_dim.values:
        params_dict[dict_key] = apply_poly_ridge(X.sel(model_dim=dict_key), y, **regression_kwargs)

    return params_dict


def get_ridge_parameters(alpha, degree, targets, save=False):
    regression_params = {}
    for target in targets:
        y = da.sel(station='Ref', variable=target)
        X = da.drop_sel(station='Ref').sel(variable=[target, 'rh', 'temp'])
        regression_params[target] = apply_model_on_dimensions(X, y, degree=degree[target], alpha=alpha[target])

    if save:
        with open(OUTPUT_DIR + LCS + 'calibration_parameters.json', 'w') as fp:
            json.dump(regression_params, fp, default=dict2json)

    # Import example:
    # with open(OUTPUT_DIR + LCS + 'calibration_parameters.json', 'r') as fp:
    #     calibration_params = json.load(fp, object_hook=deconvert)

    return regression_params


def calibrate(station_outputs, params):
    if station_outputs.ndim == 1:
        y = np.zeros(1)
        station_outputs_length = 1
    else:
        y = np.zeros(len(station_outputs[:, 0]))
        station_outputs_length = len(station_outputs)
    poly_features = PolynomialFeatures(degree=params['poly_degree'], include_bias=False)
    for n in range(0, station_outputs_length):
        if np.any(np.isnan(station_outputs[n])):
            y[n] = np.nan
        else:
            if station_outputs_length == 1:
                station_output = np.array([station_outputs, np.zeros(station_outputs.shape)])
            else:
                station_output = np.array([station_outputs[n], np.zeros(station_outputs[n].shape)])
            station_output_poly = poly_features.fit_transform(station_output)[0]
            station_output_poly_normalised = (station_output_poly - params['mean']) / params['std']
            polys = station_output_poly_normalised * params['coef']
            y[n] = params['intercept'] + polys.sum()
    return y


def dict2json(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def json2dict(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return x


def calibrator(data, target, calibration_params):
    X = data.sel(variable=[target, 'rh', 'temp']).values.copy()
    station_ = data.station.values[0]
    y = calibrate(X.reshape(X.shape[1:3]).transpose(), calibration_params[target][station_]).copy()
    da = xr.DataArray(
        y.reshape(-1, 1),
        coords=[('time', data.time.values.copy()), ('variable', [target + '_cal'])])
    da = da.astype('float32')
    return da


def make_calibration(data, calibration_params, save=False):
    da_calibrated = data.copy().rename('calibrated')
    pms = ['pm10', 'pm25', 'pm1']
    cal_list = []
    for pm in pms:
        cal = da_calibrated.groupby('station').map(calibrator, args=(pm, calibration_params)).copy().rename('case')
        cal_list.append(cal)
    da_calibrated = xr.concat(cal_list, dim='variable')
    if save:
        da_calibrated.to_netcdf(OUTPUT_DIR + LCS + 'pm_calibrated.nc')
    return da_calibrated


def import_json_as_dict(path):
    with open(path, 'r') as fp:
        dt = json.load(fp, object_hook=json2dict)
    return dt


if __name__ == '__main__':
    characterise_pre_cal = True
    find_hyperparameters = False
    calibrate_stations = False
    plotting = False

    # load data
    calibration_data_path = DATA_DIR + LCS + 'calibration_std.csv'
    da = calibration_data_import(calibration_data_path)
    station_data_path = DATA_DIR + 'Imported/lcs.nc'
    data = xr.open_dataarray(station_data_path)

    if not calibrate_stations:
        ridge_parameters_path = OUTPUT_DIR + LCS + 'calibration_parameters.json'
        calibration_params = import_json_as_dict(ridge_parameters_path)
        pm_calibrated_path = OUTPUT_DIR + LCS + 'pm_calibrated.nc'
        pm_calibrated = xr.open_dataarray(pm_calibrated_path)

    if characterise_pre_cal:
        stations = np.roll(da.indexes['station'].values, -1)
        variables = ['pm1', 'rh', 'pm25', 'temp', 'pm10']
        titles = ['PM\u2081', 'Relative Humidity',
                  'PM\u2082\u2085', 'Temperature',
                  'PM\u2081\u2080', '']
        labels = ['μg/m³', '%',
                  'μg/m³', '°C',
                  'μg/m³', '']
        date_form = DateFormatter("%H:%M")
        g = da.reindex(variable=['pm1', 'rh', 'pm25', 'temp', 'pm10']
                       , station=stations) \
            .plot.line(x='time'
                       , row='variable'
                       , col_wrap=2
                       , sharey=False
                       , sharex=False
                       , add_legend=False)
        for i, ax in enumerate(g.axes.flat):
            ax.set_title(titles[i])
            ax.set_ylabel(labels[i])
            ax.set_xlabel('')
            ax.xaxis.set_major_formatter(date_form)
        g.fig.legend(labels=stations, loc='lower right', bbox_to_anchor=(0.39, 0.1, 0.5, 0.5))
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_timeseries.png')
        plt.show()

        r_pearson = xr.corr(da.sel(station='Ref'), da, dim='time')
        sns.heatmap(r_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r')
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_pearsonr.png')
        plt.show()

        r2_pearson = r_pearson ** 2
        sns.heatmap(r2_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r²')
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_pearsonr2.png')
        plt.show()

        r2_precal = LeoMetrics('r2').apply(da, da.sel(station='Ref').drop('station'))
        sns.heatmap(r2_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('R²')
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_r2.png')
        plt.show()

        mse_precal = LeoMetrics('mse').apply(da, da.sel(station='Ref').drop('station')) ** 0.5
        sns.heatmap(mse_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Root-mean-square error')
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_rmse.png')
        plt.show()

        mape_precal = LeoMetrics('mape').apply(da, da.sel(station='Ref').drop('station'))
        sns.heatmap(mape_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Mean absolute percentage error')
        plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_mape.png')
        plt.show()

    if find_hyperparameters:
        # The hyperparameters determination was performed by eye because of the low number of samples
        y = da.sel(station='Ref', variable='pm1').values.copy()
        X = da.sel(station='WokingGreens#3', variable=['pm1', 'rh', 'temp']).values.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=13462)
        alpha = 5
        degree = 3
        coef, intercept = test_ridge_poly(X_train, y_train, X_test, y_test, alpha, degree, plot=True)
        coef, intercept = test_ridge_poly(X, y, X, y, alpha, degree, plot=True)

    if calibrate_stations:
        # make parameters json
        alpha = {'pm10': 30,
                 'pm25': 10,
                 'pm1': 5}
        degree = {'pm10': 3,
                  'pm25': 3,
                  'pm1': 3}
        targets = ['pm10', 'pm25', 'pm1']
        calibration_params = get_ridge_parameters(alpha, degree, targets, save=True)
        pm_calibrated = make_calibration(data, calibration_params, save=True)

    if plotting:
        # Plots
        X = da.sel(station='WokingGreens#5', variable=['pm1', 'rh', 'temp']).values.copy()
        X_cal = calibrate(X, regression_params['pm1']['WokingGreens#5']).copy()

        fig, ax = plt.subplots()
        ax.plot(y, X[:, 0], '.', label='sensor')
        ax.plot(y, X_cal, '.', label='calibration')
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, '--')
        ax.set_title(f'train{alpha}')
        ax.legend(loc='upper left')
        plt.show()

        da.sel(station='WokingGreens#1')['pm10_cal'] = calibrate(X, regression_params['pm10']['WokingGreens#1']).copy()
