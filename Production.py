import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pickle

df = pd.read_csv("shipping_ecommerce.csv")
df_model = df
df_dum = pd.get_dummies(df_model)

X = df_dum.drop('Class', axis=1)
y = df_dum.Class.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
#%%
xg_model = XGBClassifier(objective='binary:logistic', random_state=42, early_stopping_rounds=20, n_estimators=1000, learning_rate=0.05, n_jobs=4)
eval_set = [(X_test, y_test)]
xg_model.fit(X_train, y_train, eval_set=eval_set)
#%%
pickl = {'model': xg_model}
pickle.dump(pickl, open('model_file' + '.p', 'wb'))

file_name = 'FlaskApi/models/model_file.p'
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']




