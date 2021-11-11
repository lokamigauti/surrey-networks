import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
LCS = 'WokingNetwork/'


def apply_ridge(X_train, y_train, X_test, y_test, alpha):
    scaler_train = StandardScaler().fit(X_train)
    X_train_scaled = scaler_train.transform(X_train)
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X_train_scaled, y_train)

    scaler_test = StandardScaler().fit(X_test)
    X_test_scaled = scaler_test.transform(X_test)
    pred = reg.predict(X_test_scaled)

    fig, ax = plt.subplots()
    ax.plot(x=y_test, y=x_test, '.')
    ax.plot(x=y_test, y=pred, '.')
    x_vals = np.array(axes.get_xlim())
    y_vals = x_vals
    plt.plot(x_vals, y_vals, '--')
    plt.title(f'{alpha}')
    plt.show()

    # df[['S1_ridge', 'S1_PM10', 'Ref_PM10']].plot(x='Ref_PM10', style='.')
    # axes = plt.gca()
    # x_vals = np.array(axes.get_xlim())
    # y_vals = x_vals
    # plt.plot(x_vals, y_vals, '--')
    # plt.title(f'{alpha}')
    # plt.show()


def apply_ridge_poly(X_train, y_train, X_test, y_test, alpha, degree):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    apply_ridge(X_train_poly, y_train, X_test_poly, y_test, alpha)


if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR+LCS+'calibration.csv', parse_dates=['time'], index_col='time')
    y = df['Ref_PM10']
    X = df[['S1_Temp', 'S1_RH', 'S1_PM10']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

    apply_ridge_poly(X_train, y_train, X_test, y_test, .001, 10)
