# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:05:04 2016

@author: ZFang
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 1], [1, 2], [2, 3], [4, 4]])
y = np.array([2, 3, 4, 5])
tscv = TimeSeriesSplit(n_splits=2)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]