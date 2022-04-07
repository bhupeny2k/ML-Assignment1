# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:18:39 2022

@author: ACER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
iris=pd.read_csv("iris.csv")
X=iris.iloc[:,:-1]
Y=iris.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
s=model.predict(X_test)
from sklearn.metrics import accuracy_score
u=accuracy_score(Y_test,s)
plt.plot(X,Y)
