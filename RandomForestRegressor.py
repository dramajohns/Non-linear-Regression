# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:47:23 2023

@author: fedib
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from xgboost import XGBRegressor


dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp2/house-prices.csv')

label=dataset['Price']
data=dataset.drop(['Price'],axis=1)
data[['Brick', 'Neighborhood']] =  data[['Brick', 'Neighborhood']].apply(LabelEncoder().fit_transform)
from sklearn.model_selection import train_test_split

#train_data,test_data,train_label,test_label=train_test_split(data,label,test_size = 0.33,random_state = 0)


#train classifier
#clf = RandomForestRegressor(n_estimators=50,random_state=0)
clf = XGBRegressor(n_estimators=100,max_depth=5,objective='reg:linear',learning_rate=0.3)

clf=clf.fit(data,label) 

prediction = clf.predict(data) 



print(metrics.r2_score(label,prediction))
from sklearn.metrics import explained_variance_score
EV=explained_variance_score(label,prediction)
print("explained variance : %f" %(EV))
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#sqrt(mean_squared_error(label, prediction))

