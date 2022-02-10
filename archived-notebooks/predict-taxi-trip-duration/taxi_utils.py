from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd


def read_data(TRAIN_DIR, TEST_DIR, nrows=None):
    data_train = pd.read_csv(TRAIN_DIR,
                             parse_dates=["pickup_datetime",
                                          "dropoff_datetime"],
                             nrows=nrows)
    data_test = pd.read_csv(TEST_DIR,
                            parse_dates=["pickup_datetime"],
                            nrows=nrows)
    data_train = data_train.drop(['dropoff_datetime'], axis=1)
    data_train.loc[:, 'store_and_fwd_flag'] = data_train['store_and_fwd_flag'].map({'Y': True,
                                                                                    'N': False})
    data_test.loc[:, 'store_and_fwd_flag'] = data_test['store_and_fwd_flag'].map({'Y': True,
                                                                                  'N': False})
    data_train = data_train[data_train.trip_duration < data_train.trip_duration.quantile(0.99)]

    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]
    data_train = data_train[(data_train.pickup_longitude> xlim[0]) & (data_train.pickup_longitude < xlim[1])]
    data_train = data_train[(data_train.dropoff_longitude> xlim[0]) & (data_train.dropoff_longitude < xlim[1])]
    data_train = data_train[(data_train.pickup_latitude> ylim[0]) & (data_train.pickup_latitude < ylim[1])]
    data_train = data_train[(data_train.dropoff_latitude> ylim[0]) & (data_train.dropoff_latitude < ylim[1])]

    return (data_train, data_test)


def train_xgb(X_train, labels):
    Xtr, Xv, ytr, yv = train_test_split(X_train.values,
                                        labels,
                                        test_size=0.2,
                                        random_state=0)

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    params = {
        'min_child_weight': 1, 'eta': 0.166,
        'colsample_bytree': 0.4, 'max_depth': 9,
        'subsample': 1.0, 'lambda': 57.93,
        'booster': 'gbtree', 'gamma': 0.5,
        'silent': 1, 'eval_metric': 'rmse',
        'objective': 'reg:linear',
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=60, #227,
                      evals=evals, early_stopping_rounds=60, maximize=False,
                      verbose_eval=10)

    print('Modeling RMSE %.5f' % model.best_score)
    return model


def predict_xgb(model, X_test):
    dtest = xgb.DMatrix(X_test.values)
    ytest = model.predict(dtest)
    X_test['trip_duration'] = np.exp(ytest) - 1
    return X_test[['trip_duration']]


def feature_importances(model, feature_names):
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(feature_names))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    return feature_importance[['feature_name', 'importance']].sort_values(by='importance',
                                                                          ascending=False)


def get_train_test_fm(feature_matrix):
    X_train = feature_matrix[feature_matrix['test_data'] == False]
    X_train = X_train.drop(['test_data'], axis=1)
    labels = X_train['trip_duration']
    X_train = X_train.drop(['trip_duration'], axis=1)
    X_test = feature_matrix[feature_matrix['test_data'] == True]
    X_test = X_test.drop(['test_data', 'trip_duration'], axis=1)
    return (X_train, labels, X_test)


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = frame[v].to_dict(orient="list")

        vs = dcols.values()
        ks = dcols.keys()
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1,lvs):
                if vs[i] == vs[j]:
                    dups.append(ks[i])
                    break
    return dups
