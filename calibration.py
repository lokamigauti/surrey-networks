import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
LCS = 'WokingNetwork/'


def test_ridge(X_train, y_train, X_test, y_test, alpha, plot=False):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = reg.predict(X_test_scaled)

    coef = reg.coef_
    intercept = reg.intercept_

    if explore:
        fig, ax = plt.subplots()
        ax.plot(y_train, X_train[:,1], '.', label='sensor')
        ax.plot(y_train, reg.predict(X_train_scaled), '.', label='calibration')
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, '--')
        ax.set_title(f'train{alpha}')
        ax.legend(loc='upper left')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(y_test, X_test[:,1], '.', label='sensor')
        ax.plot(y_test, y_pred, '.', label='calibration')
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals
        ax.plot(x_vals, y_vals, '--')
        ax.set_title(f'{alpha}')
        ax.legend(loc='upper left')
        plt.show()

    return coef, intercept


def test_ridge_poly(X_train, y_train, X_test, y_test, alpha, degree, plot=False):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    coef, intercept = test_ridge(X_train_poly, y_train, X_test_poly, y_test, alpha, plot=True)

    return coef, intercept


if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR+LCS+'calibration.csv', parse_dates=['time'], index_col='time')
    y = df['Ref_PM10'].to_numpy()
    X = df[['S1_PM10', 'S1_Temp', 'S1_RH']].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=18462)

    coef, intercept = test_ridge_poly(X_train, y_train, X_test, y_test, 30, 3, plot=True) # manual test of hyperparameters
    coef
    intercept



