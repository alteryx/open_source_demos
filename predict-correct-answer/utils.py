import pandas as pd
import featuretools as ft
from featuretools.primitives import Sum, Mean, Hour
from featuretools.selection import remove_low_information_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show


def datashop_to_entityset(filename):
    # Make an EntitySet called Dataset with the following structure
    #
    # schools       students     problems
    #        \        |         /
    #   classes   sessions   problem steps
    #          \     |       /
    #           transactions  -- attempts
    #

    # Convert the csv into a dataframe using pandas
    data = pd.read_csv(filename, '\t', parse_dates=True)

    # Make the Transaction Id the index column of the dataframe and clean other columns
    data.index = data['Transaction Id']
    data = data.drop(['Row'], axis=1)
    data['Outcome'] = data['Outcome'].map({'INCORRECT': 0, 'CORRECT': 1})

    # Make a new 'End Time' column which is start_time + duration
    # This is /super useful/ because you shouldn't be using outcome data at
    # any point before the student has attempted the problem.
    data['End Time'] = pd.to_datetime(
        data['Time']) + pd.to_timedelta(pd.to_numeric(data['Duration (sec)']), 's')

    # Make a list of all the KC and CF columns present
    kc_and_cf_cols = [x for x in data.columns if (
        x.startswith('KC ') or x.startswith('CF '))]

    # Now we start making an entityset. We make 'End Time' a time index for 'Outcome'
    # even though our primary time index for a row is 'Time' preventing label leakage.
    es = ft.EntitySet('Dataset')
    es.add_dataframe(dataframe_name='transactions',
                     index='Transaction Id',
                     dataframe=data,
                     logical_types={'Outcome': "Boolean", 'Attempt At Step': "Categorical"},
                     time_index='Time',
                     secondary_time_index={'End Time': [
                        'Outcome', 'Is Last Attempt', 'Duration (sec)']}
                    )

    # Every transaction has a `problem_step` which is associated to a problem
    es.normalize_dataframe(base_dataframe_name='transactions',
                           new_dataframe_name='problem_steps',
                           index='Step Name',
                           additional_columns=['Problem Name'] + kc_and_cf_cols,
                           make_time_index=True)

    es.normalize_dataframe(base_dataframe_name='problem_steps',
                           new_dataframe_name='problems',
                           index='Problem Name',
                           make_time_index=True)

    # Every transaction has a `session` associated to a student
    es.normalize_dataframe(base_dataframe_name='transactions',
                           new_dataframe_name='sessions',
                           index='Session Id',
                           additional_columns=['Anon Student Id'],
                           make_time_index=True)

    es.normalize_dataframe(base_dataframe_name='sessions',
                           new_dataframe_name='students',
                           index='Anon Student Id',
                           make_time_index=True)

    # Every transaction has a `class` associated to a school
    es.normalize_dataframe(base_dataframe_name='transactions',
                           new_dataframe_name='classes',
                           index='Class',
                           additional_columns=['School'],
                           make_time_index=False)

    es.normalize_dataframe(base_dataframe_name='classes',
                           new_dataframe_name='schools',
                           index='School',
                           make_time_index=False)

    # And because we might be interested in creating features grouped
    # by attempts we normalize by those as well.
    es.normalize_dataframe(base_dataframe_name='transactions',
                           new_dataframe_name='attempts',
                           index='Attempt At Step',
                           additional_columns=[],
                           make_time_index=False)
    return es


def create_features(es, label='Outcome', custom_agg=[]):
    cutoff_times = es['transactions'].df[['Transaction Id', 'End Time', label]]
    fm, features = ft.dfs(entityset=es,
                          target_entity='transactions',
                          agg_primitives=[Sum, Mean] + custom_agg,
                          trans_primitives=[Hour],
                          max_depth=3,
                          approximate='2m',
                          cutoff_time=cutoff_times,
                          verbose=True)
    fm_enc, _ = ft.encode_features(fm, features)
    fm_enc = fm_enc.fillna(0)
    fm_enc = remove_low_information_features(fm_enc)
    labels = fm.pop(label)
    return (fm_enc, labels)


def estimate_score(fm_enc, label, splitter):
    k = 0
    for train_index, test_index in splitter.split(fm_enc):
        clf = RandomForestClassifier()
        X_train, X_test = fm_enc.iloc[train_index], fm_enc.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        score = round(roc_auc_score(preds, y_test), 2)
        print("AUC score on time split {} is {}".format(k, score))


def feature_importances(fm_enc, clf, feats=5):
    feature_imps = [(imp, fm_enc.columns[i])
                    for i, imp in enumerate(clf.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    print('Feature Importances: ')
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {}'.format(i + 1, f[1]))
    print("-----\n")
    return ([f[1] for f in feature_imps[0:feats]])


def datashop_plot(fm, col1='', col2='', label=None, names=['', '', '']):
    colorlist = ['#3A3A3A',  '#1072B9', '#B22222']
    colormap = {name: colorlist[name] for name in label}
    colors = [colormap[x] for x in label]
    labelmap = {0: 'INCORRECT', 1: 'CORRECT'}
    desc = [labelmap[x] for x in label]
    source = ColumnDataSource(dict(
        x=fm[col1],
        y=fm[col2],
        desc=desc,
        color=colors,
        index=fm.index,
        problem_step=fm['Step Name'],
        problem=fm['problem_steps.Problem Name'],
        attempt=fm['Attempt At Step']
    ))
    hover = HoverTool(tooltips=[
    ("(x,y)", "(@x, @y)"),
    ("problem", "@problem"),
    ("problem step", "@problem_step"),
    ])

    p = figure(title=names[0],
               tools=['box_zoom', hover, 'reset'], width=800)
    p.scatter(x='x',
              y='y',
              color='color',
              legend_group='desc',
              source=source,
              alpha=.6)

    p.xaxis.axis_label = names[1]
    p.yaxis.axis_label = names[2]
    return p

from sklearn.preprocessing import LabelEncoder
def inplace_encoder(X):
    for col in X:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[[col]].astype(str))
    return X
