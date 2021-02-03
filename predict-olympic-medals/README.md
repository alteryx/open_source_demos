# Predict how many medals a country will win at the Olympics using automated feature engineering


<a style="margin:30px" href="https://www.featuretools.com">
    <img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</a>

We will investigate the medals won by each country at each historical Olympic Games (dataset pulled from [Kaggle](https://www.kaggle.com/the-guardian/olympic-games)). The dataset contains each medal won at each Olympic Games, including the medaling athlete, their gender, and their country and sport.

We will generate a model using Featuretools that predicts whether or not a country will score more than 10 medals at the next Olympics. While it's possible to have some predictive accuracy without machine learning, feature engineering is necessary to improve the score.

You can find this tutorial on the Featuretools [site](https://www.featuretools.com/project/predict-olympic-medals/).

## Highlights
- We make predictions for the medals won at various points throughout history. Using just the average number of medals won has an average AUC score of 0.79.
- Use automated feature engineering, to generate hundred of features and improve the score to 0.95 on average


## Running the tutorial

You can download the data directly from [Kaggle](https://www.kaggle.com/the-guardian/olympic-games/data).

After downloading the data Copy the three csv files into the structure directory `data/olympic_games_data/` in the root of this repository.

Run the notebooks:

    - [BaselineSolution](BaselineSolution.ipynb)
    - [PredictOlympicMedals](PredictOlympicMedals.ipynb)

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

Featuretools is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).

### Contact

Any questions can be directed to help@featurelabs.com
