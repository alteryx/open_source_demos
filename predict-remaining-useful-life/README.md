# Predicting Remaining Useful Life
<p style="margin:30px">
    <img style="display:inline; margin-right:50px" width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
    <img style="display:inline" width=15% src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" alt="NASA" />
</p>

The general setup for the problem is a common one: we have a single table of sensor observations over time. Now that collecting information is easier than ever, most industries have already generated *time-series* type problems by the way that they store data. As such, it is crucial to be able to handle data in this form. Thankfully, built-in functionality from [Featuretools](https://www.featuretools.com) handles time varying data well.

We'll demonstrate an end-to-end workflow using a [Turbofan Engine Degradation Simulation Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) from NASA. This notebook demonstrates a rapid way to predict the Remaining Useful Life (RUL) of an engine using an initial dataframe of time-series data. There are three sections of the notebook:
1. [Understand the Data](#Step-1:-Understanding-the-Data)
2. [Generate features](#Step-2:-DFS-and-Creating-a-Model)
3. [Make predictions with Machine Learning](#Step-3:-Using-the-Model)

You can find this tutorial on the Featuretools [site](https://www.featuretools.com/project/predict-remaining-useful-life/).

*To run the notebooks, you need to download the data yourself. Download and unzip the file from [https://ti.arc.nasa.gov/c/6/](https://ti.arc.nasa.gov/c/6/). Then create a 'data' directory and place the files in the 'data' directory.*

## Highlights
* Quickly make end-to-end workflow using time-series data
* Find interesting automatically generated features
* An advanced notebook using custom primitives and hyper-parameter tuning

## Running the tutorial

The data is from the NASA Turbofan Engine Degradation Simulation Data Set
and is available [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

To run the notebooks, place the following files in the 'data' directory:
`train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

Run the Tutorials notebooks:

    - [Simple Featuretools RUL Demo](Simple%20Featuretools%20RUL%20Demo.ipynb)<br/>
    - [Advanced Featuretools RUL](Advanced%20Featuretools%20RUL.ipynb)<br/>

    *The `utils.py` script contains a number of useful helper functions.*


## Alteryx
![OpenSource](img/OpenSource_Logo-01.png)
Featuretools was created by the developers at [Alteryx](https://www.alteryx.com). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.alteryx.com/contact-us/).
