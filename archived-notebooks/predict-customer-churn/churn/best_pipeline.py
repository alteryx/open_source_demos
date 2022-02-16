import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=50)

exported_pipeline = ExtraTreesClassifier(bootstrap=False, 
                                         criterion="entropy", 
                                         max_features=0.1, 
                                         min_samples_leaf=5, 
                                         min_samples_split=8, 
                                         n_estimators=100,
                                         random_state = 50,
                                         n_jobs = -1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
