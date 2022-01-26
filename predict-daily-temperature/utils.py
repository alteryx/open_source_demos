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


def remove_nans(time_target_fs, target_col):
    # remove nans
    max_nans = 0
    for col in time_target_fs.columns:
        max_nans = max(time_target_fs[col].isna().sum(), max_nans)

    if max_nans:
        time_target_fs = time_target_fs.iloc[max_nans:]

# --> just pull this out
    X = time_target_fs

    y = X.pop(target_col)
    return X, y


def split_with_gap(df, gap, test_size=.2):
    split_point = int(df.shape[0]*(1 - test_size))

    # leave gap observations between training and test datasets
    training_data = df[:split_point]
    test_data = df[(split_point + gap):]

    return training_data, test_data
