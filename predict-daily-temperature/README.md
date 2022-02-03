# Predicting Daily Temperature

<p style="margin:30px">
    <img style="display:inline; margin-right:50px" width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
    <img width=50% src="https://evalml-web-images.s3.amazonaws.com/evalml_horizontal.svg" alt="Featuretools" />
</p>

This dataset in this problem only contains two colmns--a time index and a target daily temperatures column. Solving this problem as a univariate time series problem lets us build a rich set of features and a useful machine learning model. We can automate the process using both [Featuretools](https://www.featuretools.com) for time series feature engineering and [EvalML](https://github.com/alteryx/evalml) for performing automated time series regression.

We'll demonstrate several end-to-end workflows, starting with a baseline workflow, then moving to one where we build our features using Featuretools, and finally handling everything inside of EvalML.

## Highlights

- Understand what makes a time series problem different from other machine learning problems
- Show the impact that time series feature engineering has on our model
- Quickly make end-to-end workflow using time-series data and time series modeling concepts
- Use AutoMLSearch to perform automated time series machine learning

## Running the tutorial

The data can be found in the `dataset` directory, so the notebook can be run without having to download any additional data.

## Alteryx

![OpenSource](img/OpenSource_Logo-01.png)
Featuretools and EvalML were created by the developers at [Alteryx](https://www.alteryx.com). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.alteryx.com/contact-us/).
