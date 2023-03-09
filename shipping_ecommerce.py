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
xg_tune = models.xg_tune_class(X_train, X_test, y_train, y_test, XGBClassifier)
mods.append(xg_tune)
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
# models are built lets see how they perform
maes = []
for mod in mods:
    mae = mean_absolute_error(y_test, mod.predict(X_test))
    maes.append(mae)
maes.sort()
print(maes)
#%%
# lets also look if we can get better results with a cross value score
scores = []
for mod in mods:
    score = np.mean(cross_val_score(mod, X_train, y_train, scoring='neg_mean_absolute_error'))
    scores.append(score)

print(scores)
# scores are negligible
#%%
# test the models

preds = []
model_name = []
for mod in mods:
    model_name.append(type(mod).__name__)
    prediction = mod.predict(X_test)
    predictions = [round(value) for value in prediction]
    accuracy = accuracy_score(y_test, predictions)
    preds.append(accuracy)
#%%
print('Model        Accuracy')
for i in range(len(model_name)):
    print(model_name[i] + '\t' + str(round(preds[i], 2)*100)+'%')
