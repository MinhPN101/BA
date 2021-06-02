import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import pickle

train=pd.read_csv('data/train.csv')
features=pd.read_csv('data/features.csv')
stores = pd.read_csv('data/stores.csv')
test = pd.read_csv('data/test.csv')

data = train.merge(features, on=['Store', 'Date'], how='inner').merge(stores, on=['Store'], how='inner')
data = data.fillna(0)
data = data[data['Weekly_Sales'] >= 0]
data.drop('IsHoliday_y', axis = 1, inplace = True)
data['IsHoliday'] = data['IsHoliday_x']
data.drop('IsHoliday_x', axis = 1, inplace = True)

train_data = [data]
type_mapping = {"A": 1, "B": 2, "C": 3}
for dataset in train_data:
    dataset['Type'] = dataset['Type'].map(type_mapping)

type_mapping = {False: 0, True: 1}
for dataset in train_data:
     dataset['IsHoliday'] = dataset['IsHoliday'].map(type_mapping)

data.drop(['Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace = True)
y = data['Weekly_Sales']
X = data.drop(['Weekly_Sales'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.25)

best_rf_model = RandomForestRegressor(max_depth = 30, n_estimators=100).fit(X, y)
filename = 'finalized_model.sav'
pickle.dump(best_rf_model, open(filename, 'wb'))
# print(X.shape)