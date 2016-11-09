# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:07:34 2016

@author: ZFang
"""
import pandas as pd
import os
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    ### file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\\')
    sp_df = pd.read_csv('sp500.csv')
    
    
    ### Generate the columns, calculate the technical indactors
    n = 10
    sp_df['SMA_10'] = sp_df['Adj Close'].rolling(window=n).mean()
    mo_arr = sp_df['Adj Close'][9:].values - sp_df['Adj Close'][:-9].values
    sp_df['Momentum'] = np.zeros(len(sp_df.index))
    sp_df.loc[9:,'Momentum'] = mo_arr
    sp_df['LL'] = sp_df['Adj Close'].rolling(window=n).min()
    sp_df['HH'] = sp_df['Adj Close'].rolling(window=n).max()
    sp_df['stoch_K'] = 100 * (sp_df['Adj Close']- sp_df['LL'])/(sp_df['HH']- sp_df['LL'])
    for i in range(9,len(sp_df.index)):
        sp_df.loc[i,'WMA_10'] = (10*sp_df.loc[i,'Adj Close']+9*sp_df.loc[i-1,'Adj Close']+
                            8*sp_df.loc[i-2,'Adj Close']+7*sp_df.loc[i-3,'Adj Close']+
                            6*sp_df.loc[i-4,'Adj Close']+5*sp_df.loc[i-5,'Adj Close']+
                            4*sp_df.loc[i-6,'Adj Close']+3*sp_df.loc[i-7,'Adj Close']+
                            2*sp_df.loc[i-8,'Adj Close']+sp_df.loc[i-9,'Adj Close'])/sum(range(1,11,1))
    sp_df['EMA_12'] = sp_df['Adj Close'].ewm(span=12).mean()
    sp_df['EMA_26'] = sp_df['Adj Close'].ewm(span=26).mean()
    sp_df['DIFF'] = sp_df['EMA_12'] - sp_df['EMA_26']
    sp_df['MACD'] = np.zeros(len(sp_df.index))
    sp_df['A/D'] = np.zeros(len(sp_df.index))
    for i in range(1,len(sp_df.index)):
        sp_df.loc[i,'MACD'] = sp_df.loc[i-1,'MACD'] + 2/(n+1)*(sp_df.loc[i,'DIFF']-sp_df.loc[i-1,'MACD'])
        sp_df.loc[i,'A/D'] = (sp_df.loc[i,'High']-sp_df.loc[i-1,'Adj Close'])/(sp_df.loc[i,'High']-sp_df.loc[i,'Low'])
        
        
    ### Generate opinion dataframe, which use (-1,1) to measure the upward and downward trend
    op_df = pd.DataFrame(np.zeros((2769,7)), columns=['SMA_10', 'Momentum', 
                         'stoch_K', 'WMA_10', 'MACD', 'A/D', 'Adj Close'])
    for i in range(10,len(sp_df.index)):
        op_df.loc[i,'SMA_10']=1 if (sp_df.loc[i,'Adj Close']>sp_df.loc[i,'SMA_10']) else 0
        op_df.loc[i,'WMA_10']=1 if (sp_df.loc[i,'Adj Close']>sp_df.loc[i,'WMA_10']) else 0
        op_df.loc[i,'MACD']=1 if (sp_df.loc[i,'MACD']>sp_df.loc[i-1,'MACD']) else 0
        op_df.loc[i,'stoch_K']=1 if (sp_df.loc[i,'stoch_K']>sp_df.loc[i-1,'stoch_K']) else 0
        op_df.loc[i,'Momentum']=1 if (sp_df.loc[i,'Momentum']>0) else 0
        op_df.loc[i,'A/D']=1 if (sp_df.loc[i,'A/D']>sp_df.loc[i-1,'A/D']) else 0
        op_df.loc[i,'Adj Close']=1 if (sp_df.loc[i,'Adj Close']>sp_df.loc[i-1,'Adj Close']) else 0
    # drop first 10 columns due to nan
    op_df = op_df[10:]


    ### Training and Testing Set
    random.seed(0)
    sample_index = random.sample(list(op_df.index),2000)
    op_df_train = op_df.ix[sample_index]
    op_df_test = op_df.drop(sample_index)
    columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D']
    X = op_df_train[columns].as_matrix()
    Y = op_df_train['Adj Close'].as_matrix()
    
    
    ### Train four kinds of SVM model
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
    lin_svc = svm.LinearSVC(C=C).fit(X, Y)
    
    X_test = op_df_test[columns].as_matrix()
    Y_test = op_df_test['Adj Close'].as_matrix()
    Z = pd.DataFrame(np.zeros((1,4)), columns = ['SVC with linear kernel','LinearSVC (linear kernel)',
                                                 'SVC with RBF kernel','SVC with polynomial'])
    Y_result = Y_test
    
    
    ### Make the prediction
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        Y_result = np.vstack((Y_result, np.array(clf.predict(X_test)))) # append prediction on Y_result
        Z.iloc[0,i] = sum(clf.predict(X_test)==Y_test)/len(clf.predict(X_test))
    Y_result = Y_result.T
    
    
    ### Output
    writer = pd.ExcelWriter('result.xlsx', engine = 'xlsxwriter')
    pd.DataFrame(Y_result, columns = ['True value','SVC with linear kernel','LinearSVC (linear kernel)',
                                      'SVC with RBF kernel','SVC with polynomial']).to_excel(writer, '%s' %'result')
    writer.save()
    
    