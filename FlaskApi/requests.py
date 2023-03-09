import requests
from data_input import data_in
#%%

data_in = [3, 1, 2, 7, 4375, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]
URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

r = requests.get(URL, headers=headers, json=data)
#%%
r.json()