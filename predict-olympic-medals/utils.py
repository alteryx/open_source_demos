import featuretools.primitives as ftypes
import featuretools as ft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
import itertools
import os

from sklearn.metrics import (r2_score,
                             roc_auc_score,
                             mean_squared_error,
                             f1_score)
import math
from collections import defaultdict

def remove_low_information_features(feature_matrix, features=None):
    '''
    Select features that have at least 2 unique values and that are not all null
    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are
            feature names and rows are instances
        features (list[:class:`featuretools.PrimitiveBase`] or list[str], optional):
            List of features to select
    '''
    keep = [c for c in feature_matrix
            if (feature_matrix[c].nunique(dropna=False) > 1 and
                feature_matrix[c].dropna().shape[0] > 0)]
    feature_matrix = feature_matrix[keep]
    if features is not None:
        features = [f for f in features
                    if f.get_name() in feature_matrix.columns]
        return feature_matrix, features
    return feature_matrix


def feature_importances_as_df(fitted_est, columns):
    return (pd.DataFrame({
        'Importance': fitted_est.steps[-1][1].feature_importances_,
        'Feature': columns
    }).sort_values(['Importance'], ascending=False))


def build_seed_features(es):
    # Baseline 1
    total_num_medals = ftypes.Count(es['medals_won']['medal_id'], es['countries'])
    count_num_olympics = ftypes.NUnique(
        es['countries_at_olympic_games']['Olympic Games ID'], es['countries'])
    mean_num_medals = (
        total_num_medals / count_num_olympics).rename("mean_num_medals")

    # Number of medals in each Olympics
    olympic_id = ft.Feature(es['countries_at_olympic_games']['Olympic Games ID'],
                            es['medals_won'])
    num_medals_each_olympics = [
        ftypes.Count(
            es['medals_won']['medal_id'], es['countries'],
            where=olympic_id == i).rename("num_medals_olympics_{}".format(i))
        for i in es.get_all_instances('olympic_games')
    ]
    return num_medals_each_olympics, mean_num_medals


def get_feature_importances(estimator, feature_matrix, labels, splitter,
                            n=100):
    feature_imps_by_time = {}
    for i, train_test_i in enumerate(splitter.split(None, labels.values)):
        train_i, test_i = train_test_i
        train_dates, test_dates = splitter.get_dates(i, y=labels.values)
        X = feature_matrix.values[train_i, :]
        icols_used = [i for i, c in enumerate(X.T) if not pd.isnull(c).all()]
        cols_used = feature_matrix.columns[icols_used].tolist()

        X = X[:, icols_used]
        y = labels.values[train_i]
        clf = clone(estimator)
        clf.fit(X, y)
        feature_importances = feature_importances_as_df(clf, cols_used)
        feature_imps_by_time[test_dates[-1]] = feature_importances.head(n)

    return feature_imps_by_time


def plot_confusion_matrix(cm,
                          classes=[0, 1],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Source:

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def load_entityset(data_dir='~/olympic_games_data',
                   countries_known_for_subsequent_games=False,
                   since_date=None):
    '''
    1. Load data on each medal won at every summer Olympic Games
    2. Load data about each country that won at least one medal through Olympic history
    3. Do some formatting
    4. Sort on Year
    5. Add in a column representing a unique row for each Olympic Games
    6. Separate team medals from individual medals
    7. Create Featuretools EntitySet

    olympic_games
      |
      |   countries
      |       |                  sports
    countries_at_olympic_games     |
              |               disciplines
         medals_won____________/
              |                            athletes
     medaling_athletes ____________________/

    # do a bit more probing on analysis in simple version
    # Clean up
    '''
    # Step 1
    summer = pd.read_csv(
        os.path.join(data_dir, 'summer.csv'), encoding='utf-8')
    # winter = pd.read_csv(os.path.join(data_dir, 'winter.csv'), encoding='utf-8')

    # Step 2
    countries = pd.read_csv(
        os.path.join(data_dir, 'dictionary.csv'), encoding='utf-8')
    countries.drop(['GDP per Capita', 'Population'], axis=1, inplace=True)
    # Some countries had a '*" at their end, which we need to remove
    # in order to match with economic data
    countries['Country'] = countries['Country'].str.replace('*', '')
    countries = countries.append(
        pd.DataFrame({
            'Country': ['Unknown', 'Mixed Team'],
            'Code': ['UNK', 'ZZX']
        }),
        ignore_index=True, sort=True)

    # Step 3
    # Make names First Last instead of Last, First?
    # These two lines were taken from https://www.kaggle.com/ash316/great-olympians-eda/notebook
    summer['Athlete'] = summer['Athlete'].str.split(', ').str[::-1].str.join(' ')
    summer['Athlete'] = summer['Athlete'].str.title()

    # winter['Athlete']=winter['Athlete'].str.split(', ').str[::-1].str.join(' ')
    # winter['Athlete']=winter['Athlete'].str.title()
    summer['Year'] = (pd.to_datetime(summer['Year'], format="%Y") +
                      pd.offsets.MonthEnd(6)).dt.date
    # winter['Year'] = (pd.to_datetime(winter['Year'], format="%Y")).dt.date

    # Step 4
    # summer['Games Type'] = 'Summer'
    # winter['Games Type'] = 'Winter'
    # medals_won = pd.concat([summer, winter]).sort_values(['Year'])
    medals_won = summer.sort_values(['Year'])
    if since_date is not None:
        medals_won = medals_won[medals_won['Year'] >= since_date]

    # Step 5
    medals_won['Olympic Games Name'] = medals_won['City'].str.cat(
        medals_won['Year'].astype(str), sep=' ')
    medals_won['Country'].fillna("UNK", inplace=True)

    medals_won['Olympic Games ID'] = medals_won[
        'Olympic Games Name'].factorize()[0]
    medals_won['Country'].fillna("UNK", inplace=True)
    medals_won['Country Olympic ID'] = medals_won['Country'].str.cat(
        medals_won['Olympic Games ID'].astype(str)).factorize()[0]

    # Step 6
    unique_cols = ['Year', 'Discipline', 'Event', 'Medal']
    new_medals_won = medals_won.drop_duplicates(unique_cols, keep='first').reset_index(drop=True)
    new_medals_won['medal_id'] = new_medals_won.index
    athletes_at_olympic_games = medals_won.merge(new_medals_won[unique_cols + ['medal_id']], on=unique_cols, how='left')
    athletes_at_olympic_games = athletes_at_olympic_games[['Year', 'Athlete', 'Gender', 'medal_id']]
    medals_won = new_medals_won[[c for c in new_medals_won if c not in ['Athlete', 'Gender']]]
    athletes_at_olympic_games['Athlete Medal ID'] = athletes_at_olympic_games['Athlete'].str.cat(
        athletes_at_olympic_games['medal_id'].astype(str)).factorize()[0]

    # There were 2 duplicate athlete entries in the data, get rid of them
    athletes_at_olympic_games.drop_duplicates(['Athlete Medal ID'], inplace=True)

    # Step 7
    es = ft.EntitySet("Olympic Games")
    es.entity_from_dataframe(
        "medaling_athletes",
        athletes_at_olympic_games,
        index="Athlete Medal ID",
        time_index='Year')

    es.entity_from_dataframe(
        "medals_won",
        medals_won,
        index="medal_id",
        time_index='Year')

    es.normalize_entity(
        base_entity_id="medaling_athletes",
        new_entity_id="athletes",
        index="Athlete",
        make_time_index=True,
        new_entity_time_index='Year of First Medal',
        additional_variables=['Gender'])
    es.normalize_entity(
        base_entity_id="medals_won",
        new_entity_id="countries_at_olympic_games",
        index="Country Olympic ID",
        make_time_index=True,
        new_entity_time_index='Year',
        additional_variables=[
            'City', 'Olympic Games Name', 'Olympic Games ID', 'Country'
        ])
    es.normalize_entity(
        base_entity_id="countries_at_olympic_games",
        new_entity_id="olympic_games",
        index="Olympic Games ID",
        make_time_index=False,
        copy_variables=['Year'],
        additional_variables=['City'])
    es.normalize_entity(
        base_entity_id="medals_won",
        new_entity_id="disciplines",
        index="Discipline",
        new_entity_time_index='Debut Year',
        additional_variables=['Sport'])
    es.normalize_entity(
        base_entity_id="disciplines",
        new_entity_id="sports",
        new_entity_time_index='Debut Year',
        index="Sport")

    # map countries in medals_won to those in countries
    mapping = pd.DataFrame.from_records(
        [
            ('BOH', 'AUT', 'Bohemia'),
            ('ANZ', 'AUS', 'Australasia'),
            ('TTO', 'TRI', 'Trinidad and Tobago'),
            ('RU1', 'RUS', 'Russian Empire'),
            ('TCH', 'CZE', 'Czechoslovakia'),
            ('ROU', 'ROM', 'Romania'),
            ('YUG', 'SCG', 'Yugoslavia'),
            ('URS', 'RUS', 'Soviet Union'),
            ('EUA', 'GER', 'United Team of Germany'),
            ('BWI', 'ANT', 'British West Indies'),
            ('GDR', 'GER', 'East Germany'),
            ('FRG', 'GER', 'West Germany'),
            ('EUN', 'RUS', 'Unified Team'),
            ('IOP', 'SCG', 'Yugoslavia'),
            ('SRB', 'SCG', 'Serbia'),
            ('MNE', 'SCG', 'Montenegro'),
            ('SGP', 'SIN', 'Singapore'),
        ],
        columns=['NewCode', 'Code', 'Country'])
    columns_to_pull_from_similar = [
        u'Code', u'Subregion ID', u'Land Locked Developing Countries (LLDC)',
        u'Least Developed Countries (LDC)',
        u'Small Island Developing States (SIDS)',
        u'Developed / Developing Countries', u'IncomeGroup'
    ]
    similar_countries = mapping['Code']
    similar = countries.loc[countries['Code'].isin(similar_countries)] \
                       .reindex(columns=columns_to_pull_from_similar)
    similar = similar.merge(
        mapping, on='Code', how='outer').drop(
            ['Code'], axis=1).rename(columns={'NewCode': 'Code'})
    countries = countries.append(
        similar, ignore_index=True, sort=True).reset_index(drop=True)

    es.entity_from_dataframe("countries", countries, index="Code")

    relationships = [
        ft.Relationship(es['countries']['Code'],
                        es['countries_at_olympic_games']['Country']),
        ft.Relationship(es['medals_won']['medal_id'],
                        es['medaling_athletes']['medal_id']),
    ]

    es.add_relationships(relationships)

    if countries_known_for_subsequent_games:
        es['countries_at_olympic_games'].df['Year'] -= pd.Timedelta('7 days')
    es.add_interesting_values()
    return es

def f1_micro(a, p):
    return f1_score(a, p, average='micro')


def roc_auc_score_threshold(a, p):
    score = roc_auc_score(a, p)
    return max(score, 1 - score)


def f1_score_threshold(a, p):
    score = f1_score(a, p)
    return max(score, 1 - score)


regression_scoring_funcs = {
    'r2': (r2_score, False),
    'mse': (mean_squared_error, False),
}

binary_scoring_funcs = {
    'roc_auc': (roc_auc_score_threshold, True),
    'f1': (f1_score_threshold, False),
}

binned_scoring_funcs = {
    'f1_micro': (f1_micro, False),
}


def score_predictions(splitter, prediction_series, y, y_binary, y_binned):
    X = prediction_series.values
    binary_X = (prediction_series.values >= 10).astype(int)
    binned_X = bin_labels(prediction_series, [2, 6, 10, 50])

    scores_over_time = defaultdict(list)
    for _, test_i in splitter.split(y=y_binary):
        predicted = X[test_i]
        predicted_binary = binary_X[test_i]
        predicted_binned = binned_X[test_i]
        actual = y[test_i]
        actual_binary = y_binary[test_i]
        actual_binned = y_binned[test_i]
        for scoring_name, scoring_func in regression_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual, predicted))
        for scoring_name, scoring_func in binary_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual_binary, predicted_binary))
        for scoring_name, scoring_func in binned_scoring_funcs.items():
            sfunc = scoring_func[0]
            scores_over_time[scoring_name].append(sfunc(actual_binned, predicted_binned))
    return scores_over_time


def fit_and_score(X, y, splitter, estimator, _type='regression'):
    scores = []
    dates = []
    if _type == 'regression':
        scoring_funcs = regression_scoring_funcs
    elif _type == 'classification' and np.unique(y).shape[0] > 2:
        scoring_funcs = binned_scoring_funcs
    else:
        scoring_funcs = binary_scoring_funcs

    scores = defaultdict(list)
    for i, train_test in enumerate(splitter.split(X, y)):
        train, test = train_test
        train_dates, test_dates = splitter.get_dates(i, y=y)
        dates.append(test_dates[-1])
        cloned = clone(estimator)
        cloned.fit(X[train], y[train])
        actual = y[test]
        predictions = cloned.predict(X[test])
        try:
            probs = cloned.predict_proba(X[test])
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                probs = probs[:, 1]
        except:
            probs = None
        for name, scoring_func in scoring_funcs.items():
            sfunc, proba = scoring_func
            if proba:
                using = probs
            else:
                using = predictions
            scores[name].append(sfunc(actual, using))
    df = pd.DataFrame(scores)
    df['Olympics Year'] = dates
    return df


class TimeSeriesSplitByDate(object):
    def __init__(self,
                 dates,
                 n_splits=None,
                 combine_single_class_splits=True,
                 ignore_splits=None,
                 earliest_date=None):
        self.date_name = dates.name
        self.dates = dates.to_frame()
        self.combine_single_class_splits = combine_single_class_splits
        if n_splits is None:
            if earliest_date:
                dates = dates[dates >= earliest_date]
            n_splits = dates.nunique() - 1
        self.nominal_n_splits = n_splits
        self.earliest_date = earliest_date
        self.gen_splits()
        self.splits = None
        self.ignore_splits = ignore_splits

    def split(self, X=None, y=None, groups=None):
        if self.ignore_splits:
            if self.splits is None or (y != self.y).any():
                self.y = y
                self.splits = [
                    x for i, x in enumerate(self.nominal_splits)
                    if i not in self.ignore_splits
                ]
            return self.splits
        elif self.combine_single_class_splits:
            if self.splits is None or self.y is None or (y != self.y).any():
                self.y = y
                self.splits = []
                for i, train_test in enumerate(self.nominal_splits):
                    self.splits.append(train_test)
                    while len(self.splits) > 1 and self.single_class(
                            self.splits[-1], y):
                        last = self.splits.pop(-1)
                        penultimate = self.splits.pop(-1)
                        combined = []
                        for _last, _pen in zip(last, penultimate):
                            combined.append(
                                pd.concat([pd.Series(_last),
                                           pd.Series(_pen)], sort=True).drop_duplicates()
                                .sort_values())
                        self.splits.append(combined)
            return self.splits
        else:
            return self.nominal_splits

    def single_class(self, split, y):
        return pd.Series(y[split[1]]).nunique() == 1

    def get_dates(self, split, X=None, y=None, groups=None):
        if self.splits is None or (y != self.y).any():
            self.split(None, y)
        train_i, test_i = self.splits[split]
        return [
            self.split_index.iloc[ti][self.date_name].drop_duplicates()
            .tolist() for ti in (train_i, test_i)
        ]

    def get_split_by_date(self, date):
        date = pd.Timestamp(date)
        dates = self.split_index.drop_duplicates([self.date_name])
        split_index = dates[dates[self.date_name] == date]['split'].iloc[-1]
        return self.splits[split_index]

    def gen_splits(self):
        date_index = self.dates.drop_duplicates()
        if self.earliest_date:
            early_date_index = date_index[date_index[self.date_name] <
                                          self.earliest_date]
            early_date_index['split'] = 0
            date_index = date_index[date_index[self.date_name] >=
                                    self.earliest_date]
        date_index = date_index.reset_index(drop=True)
        date_index.index.name = 'split'
        date_index = date_index.reset_index(drop=False)
        div = math.ceil(len(date_index) / (self.nominal_n_splits + 1))
        date_index['split'] = (date_index['split'] / (div)).astype(int)
        if self.earliest_date:
            date_index = pd.concat([early_date_index, date_index], sort=True)
        self.split_index = self.dates.merge(
            date_index, on=self.date_name, how='left')
        self.split_index.index = range(self.split_index.shape[0])
        splits = self.split_index['split']
        train_splits = [
            splits[splits < (i + 1)].index.values
            for i in range(self.nominal_n_splits)
        ]
        test_splits = [
            splits[splits == (i + 1)].index.values
            for i in range(self.nominal_n_splits)
        ]
        self.nominal_splits = list(zip(train_splits, test_splits))

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.split(None, y))


def bin_labels(labels, bin_edges):
    num_bins = len(bin_edges) + 1
    new_labels = [
        int(class_) for class_ in np.digitize(labels.values, bin_edges)
    ]
    bins_used = set()
    bins = []
    for i in range(num_bins):
        if i == 0:
            bins.append("<%.1f" % (bin_edges[0]))
        elif i + 1 == num_bins:
            bins.append(">=%.1f" % (bin_edges[-1]))
        else:
            bins.append("[%.1f,%.1f)" % (bin_edges[i - 1], bin_edges[i]))

    for i, lt in enumerate(new_labels):
        new_labels[i] = bins[int(lt)]
        bins_used.add(bins[int(lt)])
    bins = [b for b in bins if b in bins_used]
    return pd.Series(new_labels)


