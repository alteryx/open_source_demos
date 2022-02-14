import numpy as np
import matplotlib.pyplot as plt
import featuretools as ft
import pandas as pd
import sklearn

plt.rcdefaults()


def plot_model_performances(baseline_evalml_score, random_forest_evalml_score, baseline_score, featuretools_score):
    # data to plot
    n_groups = 2
    evalml_runs = (baseline_evalml_score, random_forest_evalml_score)
    manual_runs = (baseline_score, featuretools_score)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, evalml_runs, bar_width,
            alpha=opacity,
            color='#9300ff',
            label='EvalML Runs')

    plt.bar(index + bar_width, manual_runs, bar_width,
            alpha=opacity,
            color='gray',
            label='Manual Runs')

    plt.xlabel('Models')
    plt.ylabel('Median Absolute Error')
    plt.title('Model Performance')
    plt.xticks(index + bar_width/2,
               ("Baseline Runs", "Feature Engineering Runs"))
    plt.legend()

    plt.tight_layout()
    plt.show()


def feature_importances(X, reg, feats=5):
    feature_imps = [(imp, X.columns[i])
                    for i, imp in enumerate(reg.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:feats]]


def set_up_entityset(df, id_='es', time_index=None):
    es = ft.EntitySet(id=id_)
    es.add_dataframe(df,
                     dataframe_name='temperatures',
                     index='id',
                     make_index=True,
                     time_index=time_index)
    return es

def read_data(filepath, time_index, target_col):
    df = pd.read_csv(filepath)
    return df[[time_index, target_col]]


def get_train_test(df, gap):
    split_point = int(df.shape[0]*.7)

    # leave gap observations between training and test datasets
    training_data = df[:split_point]
    test_data = df[(split_point + gap):]

    return training_data, test_data


def add_delayed_feature(df, col_to_delay, delay_length):
    target_delay_training = df[col_to_delay].shift(delay_length)
    target_delay_training.name = 'target_delay'
    
    return pd.concat([df, target_delay_training], axis=1)

def train_and_fit_random_forest_regressor(X_train, y_train, X_test, y_test):
    reg = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)

    # Check the accuracy of our model
    preds = reg.predict(X_test)
    score = sklearn.metrics.median_absolute_error(preds, y_test)
    print('Median Abs Error: {:.2f}'.format(score))
    return reg, score