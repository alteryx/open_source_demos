# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Predicting a customer's next purchase using automated feature engineering
#
# <p style="margin:30px">
#     <img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
# </p>
#
# **As customers use your product, they leave behind a trail of behaviors that indicate how they will act in the future. Through automated feature engineering we can identify the predictive patterns in granular customer behavioral data that can be used to improve the customer's experience and generate additional revenue for your business.**
#
# In this tutorial, we show how [Featuretools](http://www.featuretools.com) can be used to perform feature engineering on a multi-table dataset of 3 million online grocery orders provided by Instacart to train an accurate machine learning model to predict what product a customer buys next.
#
# *Note: If you are running this notebook yourself, refer to the [read me on Github](https://github.com/featuretools/predict_next_purchase#running-the-tutorial) for instructions to download the Instacart dataset*
#
# ## Highlights
#
# * We automatically generate 150+ features using Deep Feature Synthesis and select the 20 most important features for predictive modeling
# * We build a pipeline that it can be reused for numerous prediction problems (you can try this yourself!)
# * We quickly develop a model on a subset of the data and validate on the entire dataset in a scalable manner using [Dask](http://dask.pydata.org/en/latest/).

import featuretools as ft
from dask import bag
from dask.diagnostics import ProgressBar
import pandas as pd
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
ft.__version__

# ## Step 1. Load data
#
# We start by loading in just one partion of our datast. In this case, a partition of our dataset contains the complete purchase history for each user within it. At the end of the tutorial, we will run the pipeline on every partiton to generate a final model. To learn more about loading data into Featuretools, read the guide [here](https://docs.featuretools.com/loading_data/using_entitysets.html).

data = r'C:\Users\Owner\code\feature-labs\predict-next-purchase\data'
# df = load_order_products(data, nrows=int(1e+3))

es = utils.load_entityset(data, nrows=int(1e+4))
# es

# ## Visualize EntitySet

# es.plot()

# ## Step 2. Make Labels
#
# For supervised machine learning, we need labels. These labels define what our predictive model will be used for. In this tutorial, we will predict if a customer will buy Bananas in the next 4 weeks.
#
# We generate training examples by selecting a `cutoff_time` in the past to make our labels. Using users who had acivity during `training_window` days before the `cutoff_time`, we look to see if they purchase the product in the `prediction_window`.
#
# If you are running this code yourself, feel free to experiment with any of these parameters! For example, try to predict if a customer will buy "Limes" instead of "Bananas" or increase the size of your `prediction_window`.

label_times = utils.make_labels(
    es=es,
    product_name="Banana",
    cutoff_time=pd.Timestamp('March 15, 2015'),
    prediction_window=ft.Timedelta("4 weeks"),
    training_window=ft.Timedelta("60 days"),
)
label_times.head(5)

# We can see above the our training examples contain three pieces of information: a user id, the last time we can use data before feature engineering (called the "cutoff time"), and the label to predict. These are called our "label times".

# The distribution of the labels

label_times["label"].value_counts()

# ## 3. Automated Feature Engineering
# With our label times in hand, we can use Deep Feature Synthesis to automatically generate features.
#
# When we use DFS, we specify
#
# * `target_entity` - the table to build feature for
# * `cutoff_time` the point in time to calculate the features
# * `training_window` - the amount of historical data we want to use when calculating features
#
# A good way to think of the `cutoff_time` is that it let's us "pretend" we are at an earlier point in time when generating our features so we can simulate making predictions. We get this time for each customer from the label times we generated above.

# +
feature_matrix, features = ft.dfs(
    target_entity="users",
    cutoff_time=label_times,
    training_window=ft.Timedelta("60 days"),  # same as above
    entityset=es,
    verbose=True,
)
# encode categorical values
fm_encoded, features_encoded = ft.encode_features(feature_matrix, features)

print("Number of features %s" % len(features_encoded))
fm_encoded.head(10)
# -

# ## Step 4. Machine Learning
#
# Using the default parameters, we generated 160 potential features for our prediction problem. With a few simple commands, this feature matrix can be used for machine learning

X = utils.merge_features_labels(fm_encoded, label_times)
X.drop(["user_id", "time"], axis=1, inplace=True)
X = X.fillna(0)
y = X.pop("label")

# Let's train a Random Forest and validate using 3-fold cross validation

# +
clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
scores = cross_val_score(estimator=clf, X=X, y=y, cv=3, scoring="roc_auc", verbose=True)

"AUC %.2f +/- %.2f" % (scores.mean(), scores.std())
# -

# We can see we perform noticably better than guessing! However, we have a pretty high difference in performance accross folds.
#
# To improve this, let's identify the top 20 features using a Random Forest and then perform machine learning on the whole dataset (all of the partions).

clf.fit(X, y)
top_features = utils.feature_importances(clf, features_encoded, n=20)

# To persist this features, we can save them to disk.

ft.save_features(top_features, "top_features")

# ### Understanding feature engineering in Featuretools

# Before moving forward, take a look at the feature we created. You will see that they are more than just simple transformations of columns in our raw data. Instead, they aggregations (and sometimes stacking of aggregations) across the relationships in our dataset. If you're curious how this works, learn about the Deep Feature Synthesis algorithm in our documentation [here](https://docs.featuretools.com/automated_feature_engineering/afe.html).
#
# DFS is so powerful because with no manual work, the library figured out that historical purchases of bananas are important for predicting future purchases. Additionally, it surfaces that purchasing dairy or eggs and reordering behavior are important features.
#
# Even though these features are intuitive, Deep Feature Synthesis will automatically adapt as we change the prediction problem, saving us the time of manually brainstorming and implementing these data transformation.

# ## Scaling to full dataset
#
# Once we have written the pipeline for one partition, we can easily scale it out to the full dataset using [Dask](dask.pydata.org). A similar pipeline could also be built using [Spark](http://spark.apache.org/docs/2.2.0/api/python/).

pbar = ProgressBar()
pbar.register()

# First, we assemble our partitions and map them to entity sets using the function from before. A single partition contains all the data for each user within it, so this computation is easily parallelized.

path = "partitioned_data/"
#_, dirnames, _ = os.walk(path).next()
dirnames = [os.path.join(path, d) for d in os.listdir(path)]
b = bag.from_sequence(dirnames)
entity_sets = b.map(utils.load_entityset)

# Next, we create label times for each entity set

label_times = entity_sets.map(
    utils.dask_make_labels,
    product_name="Banana",
    cutoff_time=pd.Timestamp('March 1, 2015'),
    prediction_window=ft.Timedelta("4 weeks"),
    training_window=ft.Timedelta("60 days"),
)
label_times

# load in the features from before
top_features = ft.load_features("top_features")
feature_matrices = label_times.map(utils.calculate_feature_matrix, features=top_features)

# Now, we compute with Dask. Running on a Macbook Pro with a 2.2 GHz Intel Core i7 and 16gb of ram, this takes about 20 minutes to run. The compute method can take an argument `num_workers` which defaults to using all of the cores on your computer. If you don't want it to do that, you can explicitly specify a number of workers with `feature_matrices.compute(num_workers=2)` where you can replace 2 with the number of cores you want to use.

fms_out = feature_matrices.compute()
X = pd.concat(fms_out)

# Now, we repeat the same machine learning steps from the sample dataset

X.drop(["user_id", "time"], axis=1, inplace=True)
X = X.fillna(0)
y = X.pop("label")

# +
clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
scores = cross_val_score(estimator=clf, X=X, y=y, cv=3, scoring="roc_auc", verbose=True)

"AUC %.2f +/- %.2f" % (scores.mean(), scores.std())
# -

# We can now we that our accuracy has stabalized across folds, giving us much more confidence in our model.
#
# Now, let's look at the top features

clf.fit(X, y)
top_features = utils.feature_importances(clf, top_features, n=20)

# We can see the top features shifted around a bit, but mostly stayed the same.

# ## Next Steps
#
# While this is an end-to-end example of going from raw data to a trained machine learning model, it is necessary to do further exploration before claiming we've built something impact full.
#
# Fortunately, Featuretools makes it easy to build structured data science pipeline. As a next steps, you could
# * Further validate these results by creating feature vectors at different cutoff times
# * Define other prediction problems for this dataset (you can even change the entity you are making predictions on!)
# * Save feature matrices to disk as CSVs so they can be reused with different problems without recalculating
# * Experiment with parameters to Deep Feature Synthesis
# * Create custom primitives for DFS. More info [here](https://docs.featuretools.com/automated_feature_engineering/primitives.html).

# <p>
#     <img src="https://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
# </p>
#
#
# Featuretools was created by the developers at [Feature Labs](https://www.featurelabs.com/). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact).
