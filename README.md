# Data Science - Shipping Ecommerce Sales Predictor
## Overview
* A tool to predict the on-time delivery of a package shipped by a company, describes as: Yes, will be delivered on time or No, will not.
* Dataset of 11,000 packages shipped containing package and customer information
* Tested many different classification algorithms including Naive Bayes-Gaussian, Logistic Regression, KNN, SVM, Random Forest, and Gradient Boosting.
* Optimized XGBoost Classifier to reach a model with an accuracy of 67%
* Built and integrated model into RESTful API using Flask

## Code and Resource Reference
**Python Version:** 3.9
**Packages:** Pandas, Numpy, Sklearn, Lightgbm, XGBoost, Matplotlib, Seaborn, Flask, Json, Pickle
**Web Framework Requirements:** '''pip install -r requirements.txt'''
**Kaggle Dataset:** https://www.kaggle.com/datasets/ulrikthygepedersen/shipping-ecommerce
**Flask Production:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## EDA
In the data analysis I looked at how the data was distibuted to find any outliers. I also looked how the features were correlated to each other, to find
the most fit for use in model building. Here are a few highlights.

## Building the models
In total I looked at six models to see which performed the best. All models were evaluated using Sklearn's 'accuracy_score' metric, giving me a nice accuracy %.

Six Models Used:
*XGBoost - Usually the best performing and used as a Baseline.
*Light Gradient Boost - Faster and more optimal than XGboost, used to compare.
*Naive Bayes (Guassian) - Very reliable and simple Classification algorithm.
*Logistic Regression - Commonly used for Binary Classsification.
*K-nearst Neighbors - Simple but was not expecting much with this limited dataset.
*Support Vector Machine - Again more useful in datasets with more features, but I was curious on how it would perform.

## Model Performance
* **XGBoost:** Accuracy - 67%
* **Light Gradient Boost:** Accuracy - 66%
* **Naive Bayes (Guassian):** Accuracy - 64%
* **Logistic Regression:** Accuracy - 63%
* **K-nearest Neighbors:** Accuracy - 64%
* **Support Vector Machine:** Accuracy - 64%

## Production using Flask
Similar to my 'House Price Estimator' I built an API using Flask on a local server that can easily scale to a public server. The API take in a request with a list of shipping details and returns whether or not the package will be delivered on time.
