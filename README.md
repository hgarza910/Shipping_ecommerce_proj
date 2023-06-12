# E-commerce Shipping Predictor
## Overview
This project focuses on building a predictive calssification model for the E-commerce shipping industry. The goal is to classify shipments as delivered on time or not by leveraging customer and shipment data obtained from an e-commerce 'company'. 
* A tool to predict the on-time delivery of a package shipped by a company, describes as: Yes or No, the package will be delivered on time.
* Dataset of 11,000 packages shipped containing package and customer information.
* Tested many different classification algorithms including Naive Bayes-Gaussian, Logistic Regression, KNN, SVM, Random Forest, and Gradient Boosting.
* Optimized XGBoost Classifier to reach a model with an accuracy of 67%.
* Built and integrated model into RESTful API using Flask.

## Motivation
The motivation behind this project is to help an international e-commerce company gain key insights by employing advanced machine learning techniques to analyze customer data and gain valuable insights. By exploring vaious machine learning models and tuning techniques, the objective is to identify the best performing mdoel that can accurately predict whether or not a shipment will be delivered on time. 

## Data Collection and Cleaning
The dataset used for this project consists of customers and shipped product information ovtained for Kaggle. It comprises data from approximately 11,000 customers and their associated shipments. Fortunately, the dataset required minimal cleaning and preprocessing, allowing for EDA and model building right away.

## Exploritory Data Analysis (EDA)
Before proceeding with the model building phase, an exploritory data analysis (EDA) was conducted to gain a better understanding of the dataset. 
The follow aspects were explored:
* Data types of the variables
* Distributions of Customer Care Calls, Customer Ratings, Prior Purchases, Discounts Offered, and Package Weights
* The spread of aforementioned variables as well as outliers
* Correlations between each variable using a heatmap
* Comparisons of categorical variables
* Analysis of delivered shipments in relation to independent variables using pivot tables

## Model Exploration and Building
Since the problem at hand involved classification, a range of classification models were explored.
These models included:
* Gradient Boost (XGBoost) - Usually the best performing and used as a baseline.
* Light Gradient Boost - Faster and more optimal than XGBoost, used to compare it.
*  Naive Bayes-Gaussian - Very reliable and simple classification algorithm.
*  Logistic Regression - Commonly used for binary classification.
*  K-Nearest Neighbors (KNN) - Simple but was not expecting much with this limited dataset.
*  Support Vector Machine (SVM) - Again, more useful in datasets that are larger, but curious as to how it would perform.

The objective was to experiment with models that I had not previously encountered and evaluate their performance. 

## Model Testing
Once the models were trained, they were tested and evaluated using various metrics. The Mean Absolute Error (MAE) was recorded as well as the negative mean absolute error using cross-validation. Additionally, the accuracy score for each model was calculated to asses their performance.
Here is how each model performed:
* XGBoost: MAE - , Accuracy - 67%
* Light Gradient Boost: MAE - , Accuracy - 66%
* Naive Bayes-Guassian: MAE - , Accuracy - 64%
* Logistic Regression: MAE - , Accuracy - 63%
* K-Nearest Neighbors: MAE, Accuracy - 64%
* Support Vector Machine: MAE- , Accuracy - 64%

## Evaluation of Best Model
The XGBoost model demonstrated the best performance based on evaluation metrics. To further enhance its performance, a deeper analysis of the model was conducted. Log-lss
## Code and Resource Reference
**Python Version:** 3.9
**Packages:** Pandas, Numpy, Sklearn, Lightgbm, XGBoost, Matplotlib, Seaborn, Flask, Json, Pickle
**Web Framework Requirements:** '''pip install -r requirements.txt'''
**Kaggle Dataset:** https://www.kaggle.com/datasets/ulrikthygepedersen/shipping-ecommerce
**Flask Production:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Putting the Model Into Production
In order to gain practical experience in deploying the model, a Restful API was implemneted using Flask. The API was set up locally for testing purposes. With the API clients can send requests with the required parameters and recieve a binary response indicating whether the package will be delivered on time or not. The model achieved a testing accuracy of 67% in this production enviornment.

## Conclusion, Thoughts and Considerations
Although the model achieved some level of accuracy, it falls short of the desired performance. Future imporvements could involve incorporating backwards elimination to determine if the model can be enhanced further. Overall, this project provides a valuable opportunity to work with predictive classification models in the context of shipping commerce and exploring advanced machine learning techniques. 
