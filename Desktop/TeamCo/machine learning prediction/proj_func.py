# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:08:41 2016

@author: ZFang
"""

import pandas as pd
import os
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

    


def gen_df(dataframe):
    n = 10
    dataframe['SMA_10'] = dataframe['Adj Close'].rolling(window=n).mean()
    mo_arr = dataframe['Adj Close'][9:].values - dataframe['Adj Close'][:-9].values
    dataframe['Momentum'] = np.zeros(len(dataframe.index))
    dataframe.loc[9:,'Momentum'] = mo_arr
    dataframe['LL'] = dataframe['Adj Close'].rolling(window=n).min()
    dataframe['HH'] = dataframe['Adj Close'].rolling(window=n).max()
    dataframe['stoch_K'] = 100 * (dataframe['Adj Close']- dataframe['LL'])/(dataframe['HH']- dataframe['LL'])
    for i in range(9,len(dataframe.index)):
        dataframe.loc[i,'WMA_10'] = (10*dataframe.loc[i,'Adj Close']+9*dataframe.loc[i-1,'Adj Close']+
                            8*dataframe.loc[i-2,'Adj Close']+7*dataframe.loc[i-3,'Adj Close']+
                            6*dataframe.loc[i-4,'Adj Close']+5*dataframe.loc[i-5,'Adj Close']+
                            4*dataframe.loc[i-6,'Adj Close']+3*dataframe.loc[i-7,'Adj Close']+
                            2*dataframe.loc[i-8,'Adj Close']+dataframe.loc[i-9,'Adj Close'])/sum(range(1,11,1))
    dataframe['EMA_12'] = dataframe['Adj Close'].ewm(span=12).mean()
    dataframe['EMA_26'] = dataframe['Adj Close'].ewm(span=26).mean()
    dataframe['DIFF'] = dataframe['EMA_12'] - dataframe['EMA_26']
    dataframe['MACD'] = np.zeros(len(dataframe.index))
    dataframe['A/D'] = np.zeros(len(dataframe.index))
    for i in range(1,len(dataframe.index)):
        dataframe.loc[i,'MACD'] = dataframe.loc[i-1,'MACD'] + 2/(n+1)*(dataframe.loc[i,'DIFF']-dataframe.loc[i-1,'MACD'])
        dataframe.loc[i,'A/D'] = (dataframe.loc[i,'High']-dataframe.loc[i-1,'Adj Close'])/(dataframe.loc[i,'High']-dataframe.loc[i,'Low'])
    
    return dataframe
    

def gen_op_df(dataframe):
    op_df = pd.DataFrame(np.zeros((len(dataframe),9)), columns=['Date', 'SMA_10', 'Momentum', 
                         'stoch_K', 'WMA_10', 'MACD', 'A/D', 'Volume', 'Adj Close', 'Adj Close Value'])
    op_df['Date'] = dataframe['Date']
    op_df['Adj Close Value'] = dataframe['Adj Close']
    op_df['Year'] = [datetime.strptime(i, '%m/%d/%Y').year for i in op_df['Date'].values]
    for i in range(10,len(dataframe.index)-1):
        op_df.loc[i,'SMA_10']=1 if (dataframe.loc[i,'Adj Close']>dataframe.loc[i,'SMA_10']) else 0
        op_df.loc[i,'WMA_10']=1 if (dataframe.loc[i,'Adj Close']>dataframe.loc[i,'WMA_10']) else 0
        op_df.loc[i,'MACD']=1 if (dataframe.loc[i,'MACD']>dataframe.loc[i-1,'MACD']) else 0
        op_df.loc[i,'stoch_K']=1 if (dataframe.loc[i,'stoch_K']>dataframe.loc[i-1,'stoch_K']) else 0
        op_df.loc[i,'Momentum']=1 if (dataframe.loc[i,'Momentum']>0) else 0
        op_df.loc[i,'A/D']=1 if (dataframe.loc[i,'A/D']>dataframe.loc[i-1,'A/D']) else 0
        op_df.loc[i,'Volume']=1 if (dataframe.loc[i,'Volume']>dataframe.loc[i-1,'Volume']) else 0
        op_df.loc[i,'Adj Close']=1 if (dataframe.loc[i+1,'Adj Close']/dataframe.loc[i,'Adj Close']>1) else 0
    # drop first 10 columns due to nan
    op_df = op_df[10:]
    op_df.index = range(len(op_df))
    return op_df
    
    
def tune_para(dataframe):
    
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D' , 'Volume']
    X = dataframe[columns].as_matrix()
    y = dataframe['Adj Close'].as_matrix()
    
    # Time split 
    tscv = TimeSeriesSplit(n_splits=1)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test
    
    '''
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.7,0.8,0.9,1],
                             'C': [0.7,0.8,0.9,1]},
                            {'kernel': ['linear'], 'C': [0.7,0.8,0.9,1]},
                             {'kernel':['poly'], 'degree': [2,3,4,5]}]
        
        
        
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=3)
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        report = classification_report(y_true, y_pred)
        print(report)
        print()
        
        ### Calculate Output number
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
            
        return X_train, y_train, accuracy, precision, recall
        # Note the problem is too easy: the hyperparameter plateau is too flat and the
        # output model is the same for precision and recall with ties in quality.
    '''

def para_svm(dataframe):
    ### Training and Testing Set
    random.seed(0) 
    sample_index = random.sample(list(dataframe.index),int(1*len(dataframe.index)))
    para_index = random.sample(sample_index, int(0.5*len(sample_index)))
    op_df_train = dataframe.ix[para_index]
    op_df_holdout = dataframe.drop(para_index)
    columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D' , 'Volume']
    X = op_df_train[columns].as_matrix()
    Y = op_df_train['Adj Close'].as_matrix()
    
    
    ### Train four kinds of SVM model
    C = 1  # SVM regularization parameter
    svc = svm.SVC(cache_size = 1000, kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(cache_size = 1000, kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(cache_size = 1000, kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False, C=C).fit(X, Y)
    
    X_holdout = op_df_holdout[columns].as_matrix()
    Y_holdout = op_df_holdout['Adj Close'].as_matrix()
    Z = pd.DataFrame(np.zeros((1,4)), columns = ['SVC with linear kernel','LinearSVC (linear kernel)',
                                                 'SVC with RBF kernel','SVC with polynomial'])
    Y_result = Y_holdout
    
    
    ### Make the prediction
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        pred = clf.predict(X_holdout)
        Y_result = np.vstack((Y_result, np.array(pred))) # append prediction on Y_result
        Z.iloc[0,i] = sum(pred==Y_holdout)/len(pred)
    Y_result = Y_result.T
    return Z, Y_result
    
    
def para_ann(dataframe):
    ### Training and Testing Set
    random.seed(0) 
    sample_index = random.sample(list(dataframe.index),int(1*len(dataframe.index)))
    para_index = random.sample(sample_index, int(0.5*len(sample_index)))
    op_df_train = dataframe.ix[para_index]
    op_df_holdout = dataframe.drop(para_index)
    columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D' , 'Volume']
    X = op_df_train[columns].as_matrix()
    Y = op_df_train['Adj Close'].as_matrix()
    
    ### ANN model
    lbfgs_ann = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(X,Y)
    X_holdout = op_df_holdout[columns].as_matrix()
    Y_holdout = op_df_holdout['Adj Close'].as_matrix()
    Z = pd.DataFrame(np.zeros((1,1)), columns = ['ANN with backpropagation'])
    Y_result = Y_holdout
    
    pred = lbfgs_ann.predict(X_holdout)
    Y_result = np.vstack((Y_result, np.array(pred)))
    Z.iloc[0,0] = sum(pred==Y_holdout)/len(pred)
    Y_result = Y_result.T
    return Z, Y_result
    
    

