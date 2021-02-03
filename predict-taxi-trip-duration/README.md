# Predict the total ride duration of taxi trips in New York City

<a style="margin:30px" href="https://www.featuretools.com">
    <img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</a>

[Featuretools](https://www.featuretools.com/) is a framework to perform automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning. This demo uses Featuretools to develop a prediction model for the New York City Taxi Trip Duration on [Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview).

Normally, solving Kaggle problems is a very iterative process. Competitors look at the dataset, determine what features they can extract, and score it with their model. They use that accuracy to make more changes to their feature extraction, and again score their model. <b>Featuretools simplifies to process to let you extract numerous features in one iteration. </b>

You can find this tutorial on the Featuretools [site](https://www.featuretools.com/project/predict-taxi-trip-duration/).

## Highlights

We can see that using Featuretools allows us to acheive better results. Featuretools is used in notebook 3 and notebook 4, both of which score in a higher percentile than the baseline score.

<center>
<img padding='0px' border="0" src="img/rank.png" width="90%" height="70%" />
</center>

## Running the tutorial

Install the requirements


    #### Mac OS
    ```
    brew install libomp
    pip install -r requirements.txt
    ```
    
    #### Linux
    ```
    sudo apt-get install build-essential
    pip install -r requirements.txt
    ```
    
You can download the data from [Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data). After downloading, save the CSV files to a directory called `data` in the root of this repository.

## Results

1. Comparing the Kaggle Scores for the notebooks also shows the better results acheived. The leaderboard score for the most advanced notebook is very close to the best score.

<center>
<img padding='0px' border="0" src="img/bar_graph_scores.png" width="100%" height="80%" />
</center>

## FAQ

#### Q: Why remove the outliers in the train data?

Trips that are outside of the 99th quantile for trip length will unduly skew all of our numbers and results. Let's remove them. This will remove only 14593 out of the nearly 1.5 million trips from the train dataset.

Some of the trips might have a high extremely trip duration.
When we check those points, some of the passengers are traveling into the Atlantic ocean. Not only are these points outliers, they also probably don’t correspond to real travel information. By cutting out extremal values, we can train a regressor that is a better fit for most values.

#### Q: Why is `dropoff_datetime` present in the train data but not in the test data?

According to the Kaggle website:
> The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.

Since the `dropoff_datetime` was not present in the test dataset, we removed it. It also doesn’t make sense to use it since a taxi driver wouldn’t necessarily know how long a trip when picking someone up.

#### Q: What is `drop_contains`?

It is a list of strings which will tell DFS to drop any features which match the strings.

#### Q: Why is `trips.test_data` in `drop_contains`?

We don't want any features to be generated on the `test_data` column. The column is simply there to differentiate between train and test data. By putting the entity, followed by a dot, and the column name, it tell DFS to drop any aggregation features of `test_data`. If we had put just `test_data` in `drop_contains`, then it would have dropped the `test_data` column and the aggregation features of `test_data`.

#### Q: What is the model being used?

XGBoost, which stands for e<b>X</b>treme <b>G</b>radient <b>B</b>oosting, is the model used. It is a very popular machine learning algorithm in Kaggle competitions for structured or tabular data. More infromation can be found
<a href="https://xgboost.readthedocs.io/en/latest/">here</a>.

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Featuretools is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).

### Contact

Any questions can be directed to help@featurelabs.com
