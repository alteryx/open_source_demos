import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=50)

# Average CV score on the training set was:0.11411730905984563
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.024),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=0.35000000000000003, min_samples_leaf=5, min_samples_split=17, n_estimators=100, subsample=0.15000000000000002)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
