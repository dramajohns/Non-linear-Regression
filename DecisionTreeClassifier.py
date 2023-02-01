# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:24:10 2022

@author: fedib
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree

dataset = pd.read_csv('C:/STUDY/SEMESTRE 2/machine learning/tp/tp1/iris.csv')

label=dataset['variety']
data=dataset.drop(['variety'],axis=1)

from sklearn.model_selection import train_test_split

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size = 0.33,random_state = 0)


#train classifier
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0)

clf=clf.fit(train_data,train_label) 

prediction = clf.predict(test_data) 

from sklearn.metrics import accuracy_score

ACC=accuracy_score(test_label, prediction)*100
print(ACC)

tree.plot_tree(clf) 

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(test_label, prediction)
print(CM)











