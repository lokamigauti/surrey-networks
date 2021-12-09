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
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import json
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')
from bokeh.plotting import show

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


def weight_array(X, X_real_min, X_real_max):
    """
    Make weights array, doubles the weight inside the real limits of the data
    :param X: data np.array
    :param X_real_min: real minimum
    :param X_real_max: real maximum
    :return: weights
    """
    weights = np.ones(len(X))
    for n, x, in enumerate(X):
        if x > X_real_min and x < X_real_max:
            weights[n] = weights[n] * 2
    return weights


def make_weights_array(X, weights_pol_ranges):
    weights = []
    var_names = ['target', 'temp', 'rh']
    for n, k in enumerate(var_names):
        weights.append(weight_array(X[:, n],
                                    weights_pol_ranges[k + '_real_min'],
                                    weights_pol_ranges[k + '_real_max']))
    weights = np.array(weights).reshape(X.shape)
    weights_1d = np.ones(len(weights))
    for n in range(0, len(weights)):
        weights_1d[n] = weights[n].mean()
    return weights_1d


def apply_poly_ridge(X, y, degree, alpha, weights_params='none'):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    scaler = StandardScaler().fit(X_poly)
    X_poly_scaled = scaler.transform(X_poly)

    reg = linear_model.Ridge(alpha=alpha)

    if weights_params == 'none':
        reg.fit(X_poly_scaled, y)
    else:
        weights = make_weights_array(X, weights_params)
        reg.fit(X_poly_scaled, y, weights)

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


def get_ridge_parameters(da, alpha, degree, targets, weights_params, save=False):
    regression_params = {}
    for target in targets:
        y = da.sel(station='Ref', variable=target)
        X = da.drop_sel(station='Ref').sel(variable=[target, 'rh', 'temp'])
        regression_params[target] = apply_model_on_dimensions(X, y, degree=degree[target], alpha=alpha[target], weights_params=weights_params)

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
    station_ = data.station.values.item()
    if len(X.shape) == 3:
        X = X[0]
    if X.shape[0] == 3:
        X = X.transpose()
    y = calibrate(X, calibration_params[target][station_]).copy()
    da = xr.DataArray(y.reshape(-1, 1),
        coords=[('time', data.time.values.copy()), ('variable', [target + '_cal'])])
    da = da.astype('float32')
    return da


def make_calibration(data, calibration_params, output_path='none'):
    da_calibrated = data.copy().rename('calibrated')
    pms = ['pm10', 'pm25', 'pm1']
    cal_list = []
    for pm in pms:
        cal = da_calibrated.groupby('station').map(calibrator, args=(pm, calibration_params)).copy().rename('case')
        cal_list.append(cal)
    da_calibrated = xr.concat(cal_list, dim='variable')
    if not output_path == 'none':
        da_calibrated.to_netcdf(output_path)
    return da_calibrated


def import_json_as_dict(path):
    with open(path, 'r') as fp:
        dt = json.load(fp, object_hook=json2dict)
    return dt


def cross_corr(da, station, reference, variable, dt, mode='full', method='auto', plot=False):
    ref_var = da.sel(station=reference, variable=variable).copy()
    station_var = da.sel(station=station, variable=variable).copy()

    corr = signal.correlate(ref_var, station_var, mode=mode, method=method)
    lags = signal.correlation_lags(len(ref_var), len(station_var))
    lag = lags[corr.argmax()] * dt

    if plot:
        fig, (ax_ref, ax_station, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
        ax_ref.plot(ref_var)
        ax_ref.set_title('Reference')
        ax_ref.set_xlabel('Sample Number')
        ax_station.plot(station_var)
        ax_station.set_title('Station')
        ax_station.set_xlabel('Sample Number')
        ax_corr.plot(lags, corr)
        ax_corr.set_title('Cross-correlated signal')
        ax_corr.set_xlabel('Lag')
        ax_ref.margins(0, 0.1)
        ax_station.margins(0, 0.1)
        ax_corr.margins(0, 0.1)
        fig.tight_layout()
        plt.show()

    return lag


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


if __name__ == '__main__':
    characterise_cal = True
    find_hyperparameters = False
    calibrate_stations = False

    # load data
    calibration_data_path = DATA_DIR + LCS + 'calibration_std.csv'
    calibration_chamber_data = calibration_data_import(calibration_data_path)
    station_data_path = DATA_DIR + 'Imported/lcs.nc'
    data = xr.open_dataarray(station_data_path)

    calibration_chamber_pm1_25 = calibration_chamber_data.sel(variable='pm25') \
                                 - calibration_chamber_data.sel(variable='pm1')
    calibration_chamber_pm1_25 = calibration_chamber_pm1_25.assign_coords(variable='pm1_25').expand_dims(dim='variable')
    calibration_chamber_data = xr.concat([calibration_chamber_data, calibration_chamber_pm1_25], dim='variable')

    calibration_chamber_pm25_10 = calibration_chamber_data.sel(variable='pm10') \
                                 - calibration_chamber_data.sel(variable='pm25')
    calibration_chamber_pm25_10 = calibration_chamber_pm25_10.assign_coords(variable='pm25_10').expand_dims(dim='variable')
    calibration_chamber_data = xr.concat([calibration_chamber_data, calibration_chamber_pm25_10], dim='variable')



    if not calibrate_stations:
        ridge_parameters_path = OUTPUT_DIR + LCS + 'calibration_parameters.json'
        calibration_params = import_json_as_dict(ridge_parameters_path)
        pm_calibrated_path = OUTPUT_DIR + LCS + 'pm_calibrated.nc'
        pm_calibrated = xr.open_dataarray(pm_calibrated_path)
        data = data.combine_first(pm_calibrated)

    if characterise_cal:
        stations = np.roll(calibration_chamber_data.indexes['station'].values, -1)
        variables = ['pm1', 'rh', 'pm25', 'temp', 'pm10']
        titles = ['PM\u2081', 'Relative Humidity',
                  'PM\u2082\u2085', 'Temperature',
                  'PM\u2081\u2080', '']
        labels = ['μg/m³', '%',
                  'μg/m³', '°C',
                  'μg/m³', '']
        date_form = DateFormatter("%H:%M")
        g = calibration_chamber_data.reindex(variable=['pm1', 'rh', 'pm25', 'temp', 'pm10']
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
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_timeseries.png')
        plt.show()

        # from holoviews.operation import gridmatrix
        #
        # hv_ds = hv.Dataset(da.to_dataset(dim='station'))
        # hv_ds.to(hv.Image, kdims=["time", "station"], dynamic=False)
        # show(hv.render(scatter))
        # group = hv_ds.groupby('variable', container_type=hv.NdOverlay)
        # grid = gridmatrix(group, diagonal_type=hv.Scatter)
        #
        # img = hv.Image((range(3), range(5), np.random.rand(5, 3)), datatype=['grid'])
        # img
        # show(hv.render(img))

        # da.plot(x='station',
        #         y='station',
        #         row='variable',
        #         col_wrap=2,
        #         sharey=False,
        #         sharex=False)
        #
        # plt.show()

        r_pearson = xr.corr(calibration_chamber_data.sel(station='Ref'), calibration_chamber_data, dim='time')
        sns.heatmap(r_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_pearsonr.png')
        plt.show()
        r_pearson.groupby('variable').mean(dim='station')
        r_pearson.groupby('variable').std(dim='station')

        r2_pearson = r_pearson ** 2
        sns.heatmap(r2_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r²')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_pearsonr2.png')
        plt.show()
        r2_pearson.groupby('variable').mean(dim='station')
        r2_pearson.groupby('variable').std(dim='station')

        r2_precal = LeoMetrics('r2').apply(calibration_chamber_data, calibration_chamber_data.sel(station='Ref').drop('station'))
        sns.heatmap(r2_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('R²')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_r2.png')
        plt.show()
        r2_precal.groupby('variable').mean(dim='station')
        r2_precal.groupby('variable').std(dim='station')

        rmse_precal = LeoMetrics('mse').apply(calibration_chamber_data, calibration_chamber_data.sel(station='Ref').drop('station')) ** 0.5
        sns.heatmap(rmse_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Root-mean-square error')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_rmse.png')
        plt.show()
        rmse_precal.groupby('variable').mean(dim='station')
        rmse_precal.groupby('variable').std(dim='station')

        mape_precal = LeoMetrics('mape').apply(calibration_chamber_data, calibration_chamber_data.sel(station='Ref').drop('station'))
        sns.heatmap(mape_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Mean absolute percentage error')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_mape.png')
        plt.show()
        mape_precal.groupby('variable').mean(dim='station')
        mape_precal.groupby('variable').std(dim='station')

        # Calibration

        alpha = {'pm10': 50,
                 'pm25': 20,
                 'pm1': 10}
        degree = {'pm10': 3,
                  'pm25': 3,
                  'pm1': 3}
        weights_ranges = {'target_real_min': 0,
                          'target_real_max': 15,
                          'temp_real_min': 0,
                          'temp_real_max': 20,
                          'rh_real_min': 0,
                          'rh_real_max': 100}
        targets = ['pm10', 'pm25', 'pm1']
        calibration_params = get_ridge_parameters(calibration_chamber_data, alpha, degree, targets, weights_ranges, save=True)
        chamber_calibration = make_calibration(calibration_chamber_data.drop_sel(station='Ref'), calibration_params)
        chamber_calibration = chamber_calibration.combine_first(calibration_chamber_data)

        chamber_calibration.loc[dict(station='Ref', variable=['pm1_cal', 'pm25_cal', 'pm10_cal'])] \
            = chamber_calibration.loc[dict(station='Ref', variable=['pm1', 'pm25', 'pm10'])].values
        chamber_calibration = chamber_calibration.reindex(variable=['pm1',
                                                                    'pm1_cal',
                                                                    'pm25',
                                                                    'pm25_cal',
                                                                    'pm10',
                                                                    'pm10_cal',
                                                                    'rh',
                                                                    'temp'])

        stations = np.roll(chamber_calibration.indexes['station'].values, -1)
        variables = ['pm1_cal', 'pm25_cal', 'pm10_cal']
        titles = ['PM\u2081 Calibrated',
                  'PM\u2082\u2085 Calibrated',
                  'PM\u2081\u2080 Calibrated',
                  '']
        labels = ['μg/m³',
                  'μg/m³',
                  'μg/m³',
                  '']
        date_form = DateFormatter("%H:%M")
        g = chamber_calibration.reindex(variable=['pm1_cal', 'pm25_cal', 'pm10_cal']
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
        plt.savefig(OUTPUT_DIR + LCS + 'cal_timeseries.png')
        plt.show()

        stations = np.roll(chamber_calibration.indexes['station'].values, -1)
        variables = ['pm1', 'pm25', 'pm10']
        titles = ['PM\u2081 Uncalibrated',
                  'PM\u2082\u2085 Uncalibrated',
                  'PM\u2081\u2080 Uncalibrated',
                  '']
        labels = ['μg/m³',
                  'μg/m³',
                  'μg/m³',
                  '']
        date_form = DateFormatter("%H:%M")
        g = chamber_calibration.reindex(variable=variables
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
        plt.savefig(OUTPUT_DIR + LCS + 'precal_timeseries.png')
        plt.show()

        # TODO: plot cal and uncal vs ref
        plt.style.use('bmh')
        da_ref = chamber_calibration.sel(station='Ref')
        da_others = chamber_calibration.isel(station=slice(1, None))
        variables_to_plot = ['pm1_cal', 'pm25_cal', 'pm10_cal']
        for variable_to_plot in variables_to_plot:
            fig, axs = plt.subplots(1, 2, figsize=[12, 5], sharey=True)
            da_others_cal = da_others.copy().sel(variable=variable_to_plot)
            da_others_uncal = da_others.copy().sel(variable=variable_to_plot.replace('_cal', ''))
            da_ref_ = da_ref.copy().sel(variable=variable_to_plot.replace('_cal', ''))
            for station in da_others.station.values:
                axs[0].scatter(da_ref_.values, da_others_cal.sel(station=station).values)
                axs[1].scatter(da_ref_.values, da_others_uncal.sel(station=station).values)
            axs[0].plot([0, 175], [0, 175], color='k', linestyle='dashed')
            axs[1].plot([0, 175], [0, 175], color='k', linestyle='dashed')
            axs[1].legend(['1:1']+da_others.station.values.tolist())
            axs[0].set_ylabel('Observ. concentration')
            axs[0].set_xlabel('Ref. concentration')
            axs[1].set_xlabel('Ref. concentration')
            plt.suptitle(variable_to_plot)
            plt.savefig(OUTPUT_DIR + LCS + variable_to_plot + '.png')
            plt.show()

        chamber_calibration_pm = chamber_calibration.drop_sel(variable=['rh', 'temp'])

        r_pearson = xr.corr(chamber_calibration_pm.sel(station='Ref'), chamber_calibration_pm, dim='time')
        sns.heatmap(r_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r')
        # plt.savefig(OUTPUT_DIR + LCS + 'cal_pearsonr.png')
        plt.show()
        r_pearson.groupby('variable').mean(dim='station')
        r_pearson.groupby('variable').std(dim='station')

        r2_pearson = r_pearson ** 2
        sns.heatmap(r2_pearson.transpose().to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Pearson\'s r²')
        # plt.savefig(OUTPUT_DIR + LCS + 'pre_cal_pearsonr2.png')
        plt.show()
        r2_pearson.groupby('variable').mean(dim='station')
        r2_pearson.groupby('variable').std(dim='station')

        r2_precal = LeoMetrics('r2').apply(chamber_calibration_pm,
                                           chamber_calibration_pm.sel(station='Ref').drop('station'))
        sns.heatmap(r2_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('R²')
        # plt.savefig(OUTPUT_DIR + LCS + 'cal_r2.png')
        plt.show()
        r2_precal.groupby('variable').mean(dim='station')
        r2_precal.groupby('variable').std(dim='station')

        rmse_precal = LeoMetrics('mse').apply(chamber_calibration_pm,
                                              chamber_calibration_pm.sel(station='Ref').drop('station')) ** 0.5
        sns.heatmap(rmse_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Root-mean-square error')
        # plt.savefig(OUTPUT_DIR + LCS + 'cal_rmse.png')
        plt.show()
        rmse_precal.groupby('variable').mean(dim='station')
        rmse_precal.groupby('variable').std(dim='station')

        mape_precal = LeoMetrics('mape').apply(chamber_calibration_pm,
                                               chamber_calibration_pm.sel(station='Ref').drop('station'))
        sns.heatmap(mape_precal.to_pandas().drop('Ref')
                    , annot=True
                    , cmap='viridis'
                    , linewidths=2
                    , linecolor='white'
                    ).set_title('Mean absolute percentage error')
        # plt.savefig(OUTPUT_DIR + LCS + 'cal_mape.png')
        plt.show()
        mape_precal.groupby('variable').mean(dim='station')
        mape_precal.groupby('variable').std(dim='station')

    if find_hyperparameters:
        # The hyperparameters determination was performed by eye because of the low number of samples
        y = calibration_chamber_data.sel(station='Ref', variable='pm1').values.copy()
        X = calibration_chamber_data.sel(station='WokingGreens#3', variable=['pm1', 'rh', 'temp']).values.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=13462)
        alpha = 5
        degree = 3
        coef, intercept = test_ridge_poly(X_train, y_train, X_test, y_test, alpha, degree, plot=True)
        coef, intercept = test_ridge_poly(X, y, X, y, alpha, degree, plot=True)

    if calibrate_stations:
        # make parameters json
        alpha = {'pm10': 50,
                 'pm25': 20,
                 'pm1': 10}
        degree = {'pm10': 3,
                  'pm25': 3,
                  'pm1': 3}
        targets = ['pm10', 'pm25', 'pm1']
        weights_ranges = {'target_real_min': 0,
                          'target_real_max': 15,
                          'temp_real_min': 0,
                          'temp_real_max': 20,
                          'rh_real_min': 0,
                          'rh_real_max': 100}

        calibration_params = get_ridge_parameters(calibration_chamber_data, alpha, degree, targets, weights_ranges, save=True)
        pm_calibrated = make_calibration(data, calibration_params, output_path=OUTPUT_DIR + LCS + 'pm_calibrated.nc')
        data = data.combine_first(pm_calibrated)
        csv_path = OUTPUT_DIR + LCS + 'data_calibrated.csv'
        flatten_data(data, sample_dim='time', feature_dim='variable', output_path=csv_path)

