import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import models


df = pd.read_csv("shipping_ecommerce.csv")
#%%
df.head(10)
#%%
df.describe()
#%%
df_model = df
df_dum = pd.get_dummies(df_model)

X = df_dum.drop('Class', axis=1)
y = df_dum.Class.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

mods = []
#%%
# Class can either be 1 for on time or 0 for late
# lets look at classification models for binary solution

xg_model = models.xgboost_class(X_train, y_train, XGBClassifier)
mods.append(xg_model)
#xg_tune = models.xg_tune_class(X_train, X_test, y_train, y_test, XGBClassifier)
#mods.append(xg_tune)
lgbm_model = models.lgbm_class(X_train, y_train, LGBMClassifier)
mods.append(lgbm_model)
nbg_model = models.nbg_class(X_train, y_train, GaussianNB)
mods.append(nbg_model)
log_model = models.log_class(X_train, y_train, LogisticRegression)
mods.append(log_model)
knn_model = models.knn_class(X_train, y_train, KNeighborsClassifier)
mods.append(knn_model)
svm_model = models.svm_class(X_train, y_train, SVC)
mods.append(svm_model)

#%%
kern_model_2 = models.kern_reg(X_train, y_train, KernelRidge)
#%%
# models are built lets see how they perform
maes = []

for mod in mods:
    model_name = type(mod).__name__
    mae = mean_absolute_error(y_test, mod.predict(X_test))
    tup = (model_name, mae)
    maes.append(tup)

maes = sorted(maes, key=lambda x: x[1], reverse=True)
#%%
# lets also look if we can get better results with a cross value score
scores = []
for mod in mods:
    model_name = type(mod).__name__
    score = np.mean(cross_val_score(mod, X_train, y_train, scoring='neg_mean_absolute_error'))
    tup = (model_name, score)
    scores.append(tup)

scores = sorted(scores, key=lambda x: x[1], reverse=True)
# scores are negligible
#%%
# test the models

training_predictions = []
testing_predictions = []
for mod in mods:
    model_name = type(mod).__name__
    training_pred = mod.predict(X_train)
    testing_pred = mod.predict(X_test)
    training_preds = [round(value) for value in training_pred]
    testing_preds = [round(value) for value in testing_pred]
    training_accuracy = accuracy_score(y_train, training_preds)
    testing_accuracy = accuracy_score(y_test, testing_preds)
    training_tup = (model_name, training_accuracy)
    testing_tup = (model_name, testing_accuracy)
    training_predictions.append(training_tup)
    testing_predictions.append(testing_tup)

training_predictions = sorted(training_predictions, key=lambda x: x[1], reverse=True)
testing_predictions = sorted(testing_predictions, key=lambda x: x[1], reverse=True)

#%%
# print MAE's
for mae in maes:
    print('Model: {:<25} MAE: {:.2f}'.format(mae[0], mae[1]))
print('\n')
# print cv scores
for score in scores:
    print('Model: {:<25} CV Score: {:.2f}'.format(score[0], score[1]))
print('\n')
# print training scores
for prediction in training_predictions:
    print('Model: {:<25} Training Accuracy: {:.2f}'.format(prediction[0], prediction[1]))
print('\n')
# print testing scores
for prediction in testing_predictions:
    print('Model: {:<25} Testing Accuracy: {:.2f}'.format(prediction[0], prediction[1]))