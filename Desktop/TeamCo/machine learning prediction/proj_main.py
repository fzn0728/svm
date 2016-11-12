# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:19:14 2016

@author: ZFang
"""

# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random
from datetime import datetime



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
                         'stoch_K', 'WMA_10', 'MACD', 'A/D', 'Volume', 'Adj Close'])
    op_df['Year'] = [datetime.strptime(i, '%Y-%m-%d').year for i in dataframe['Date'].values]
    for i in range(10,len(ra_df.index)-1):
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
        Y_result = np.vstack((Y_result, np.array(clf.predict(X_holdout)))) # append prediction on Y_result
        Z.iloc[0,i] = sum(clf.predict(X_holdout)==Y_holdout)/len(clf.predict(X_holdout))
    Y_result = Y_result.T
    return Z, Y_result


if __name__ == '__main__':
    ### file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\\')
    file_name = 'appl.csv'
    new_file_name = 'appl_op.csv'
    orig_df = pd.read_csv(file_name)
    
    
    ### Generate the columns, calculate the technical indactors
    ra_df = gen_df(orig_df)  
        
    ### Generate opinion dataframe, which use (-1,1) to measure the upward and downward trend
    # op_df = gen_op_df(ra_df)
    # op_df.to_csv(new_file_name, index=True, header=True, index_label='index')
    
    op_df = pd.read_csv(new_file_name, index_col='index')
    accuracy_df = pd.DataFrame(np.zeros((1,4)), columns = ['SVC with linear kernel','LinearSVC (linear kernel)',
                                             'SVC with RBF kernel','SVC with polynomial'])
    for i in set(op_df['Year']):
        yr_df = op_df[op_df['Year']==i]
        Z, Y_result = para_svm(yr_df)
        accuracy_df = accuracy_df.append(Z, ignore_index=True)
    
    
'''
    
    ### Output
    writer = pd.ExcelWriter('result.xlsx', engine = 'xlsxwriter')
    pd.DataFrame(Y_result, columns = ['True value','SVC with linear kernel','LinearSVC (linear kernel)',
                                      'SVC with RBF kernel','SVC with polynomial']).to_excel(writer, '%s' %'result')
    writer.save()
    
'''