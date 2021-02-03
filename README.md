# Alteryx Open Source Demos 

<div class="row">
    <a href="https://evalml.alteryx.com/">
        <img width=40% src="https://evalml-web-images.s3.amazonaws.com/evalml_horizontal.svg" alt="EvalML" />
    </a>
    <a href="https://www.featurelabs.com/">
        <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
    </a>
</div>

This repository consists of a series of demos that leverage **EvalML**, **Featuretools**, **Woodwork**, and **Compose**. The demos rely on a different subset of these libraries and to various levels of complexity.

Building an accurate machine learning model requires several important steps. One of the most complex and time consuming is extracting information through features. Finding the right features is a crucial component of both interpreting the dataset as a whole as well as building a model with great predictive power. Another core component of any machine learning process is selecting the right estimator to use for the problem at hand. By combining the best features with the most accurate estimator and its corresponding hyperparameters, we can build a machine learning model that can generalize well to unknown data. Just as the process of feature engineering is made simple by Featuretools, we have made automated machine learning easy to implement using EvalML.

## Running these tutorials

1. Clone the repository.

    `git clone https://github.com/alteryx/open-source-demos`

2. Install the requirements. It's recommended to create a new environment to work in to install these libraries separately.

    `pip install -r requirements.txt`
    
    In order to properly execute the demos, please install **Graphviz** according to the Featuretools [documentation](https://featuretools.alteryx.com/en/stable/install.html?highlight=graphviz#installing-graphviz).

3. Download the data.

    You can download the data for each demo by following the instructions in each tutorial. The dataset will usually be kept in a folder named `data` within the         project structure.

4. The tutorials can be run in Jupyter Notebook.

    `jupyter notebook`
