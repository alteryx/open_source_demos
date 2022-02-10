# Predicting Daily Temperature

<a style="margin:30px" href="https://www.featuretools.com">
    <img width=40% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</a>
<a style="margin:30px" href="https://evalml.alteryx.com/en/stable/index.html">
    <img width=40% src="https://evalml-web-images.s3.amazonaws.com/evalml_horizontal.svg" alt="EvalML" />
</a>

In this tutorial, we show how Featuretools and EvalML can be used to automate time series forecasting bu predicting future daily temperatures using historical daily temperature data.

This dataset in this tutorial only contains two columns--a time index and a target daily temperatures column. Solving this problem as a univariate time series problem lets us build a rich set of features and a useful machine learning model. We can automate the process using both [Featuretools](https://www.featuretools.com) for time series feature engineering and [EvalML](https://github.com/alteryx/evalml) for performing automated time series regression.

We'll demonstrate several end-to-end workflows, starting with a baseline workflow, then moving to one where we build our features using Featuretools, and finally handling everything inside of EvalML.

## Highlights

- Understand what makes a time series problem different from other machine learning problems
- Show the impact that time series feature engineering has on our model
- Quickly make end-to-end workflow using time series data and time series modeling concepts
- Use AutoMLSearch to perform automated time series machine learning

## Running the tutorial

The data can be found in the `dataset` directory, so the notebook can be run without having to download any additional data.

## Alteryx

<p align="center">
<img width=50% src="https://alteryx-open-source-images.s3.amazonaws.com/OpenSource_Logo-01.jpg" alt="ayx_os" />
</p>

This is a demo created & maintained by [Alteryx](https://www.alteryx.com). It uses **Featuretools** and **EvalML**, which are open source libraries maintained by Alteryx. To see the other open source projects we’re working on visit Alteryx Open Source. If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.alteryx.com/contact-us/).
