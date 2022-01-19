import featuretools as ft
import woodwork as ww
import evalml
from evalml import AutoMLSearch
from evalml.utils import infer_feature_types
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
import pandas as pd


def test_stationarity(timeseries):
    # Adapted from https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def split_for_evalml(df, target_col):
    X = df.ww.drop([target_col])
    y = df.ww[target_col]

    return evalml.preprocessing.split_data(X, y, problem_type='regression', test_size=.2)


def feature_importances(X, reg, feats=5):
    feature_imps = [(imp, X.columns[i])
                    for i, imp in enumerate(reg.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:feats]]
