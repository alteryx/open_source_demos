# Predicting a customer's next purchase using automated feature engineering

<a style="margin:30px" href="https://github.com/FeatureLabs/featuretools">
    <img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</a>

**As customers use your product, they leave behind a trail of behaviors that indicate how they will act in the future. Through automated feature engineering we can identify the predictive patterns in granular customer behavioral data that can be used to improve the customer's experience and generate additional revenue for your business.**

In this tutorial, we show how [Featuretools](https://github.com/FeatureLabs/featuretools) can be used to perform feature engineering on a multi-table dataset of 3 million online grocery orders provided by Instacart to train an accurate machine learning model to predict what product a customer buys next.

You can find this tutorial on the Featuretools [site](https://www.featuretools.com/project/predict-next-purchase/).

*Note: If you are running this notebook yourself, refer to the README.md on Github for instructions to download the Instacart dataset*

## Highlights

* We automatically generate 150+ features using Deep Feature Synthesis and select the 20 most important features for predictive modeling
* We automatically generate label times using [Compose](https://github.com/FeatureLabs/compose) which can be reused for numerous prediction problems (you can try this yourself!)
* We quickly develop a model on a subset of the data and validate on the entire dataset in a scalable manner using [Dask](http://dask.pydata.org/en/latest/).

## Running the tutorial

*Note: In order to run the tutorial, you will need to use Featuretools version 0.16.0 or newer.*

You can download the data directly from Instacart [here](https://www.kaggle.com/c/instacart-market-basket-analysis/data).

After downloading the data, save the CSVs to a directory called `data` in the root of this repository. 

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Featuretools is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact).

### Contact

Any questions can be directed to help@featurelabs.com
