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



def gen_df(dataframe):
    n = 10
    dataframe['SMA_10'] = dataframe['Adj Close'].rolling(window=n).mean()
    mo_arr = dataframe['Adj Close'][9:].values - dataframe['Adj Close'][:-9].values
    dataframe['Momentum'] = np.zeros(len(dataframe.index))
    dataframe.loc[9:,'Momentum'] = mo_arr
    dataframe['LL'] = sp_df['Adj Close'].rolling(window=n).min()
    dataframe['HH'] = sp_df['Adj Close'].rolling(window=n).max()
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
    op_df = pd.DataFrame(np.zeros((len(dataframe),10)), columns=['SMA_10', 'Momentum', 
                         'stoch_K', 'WMA_10', 'MACD', 'A/D', 'Volume', 'Fed Rate', 'Un Empl', 'Adj Close'])
    for i in range(10,len(sp_df.index)-1):
        op_df.loc[i,'SMA_10']=1 if (dataframe.loc[i,'Adj Close']>dataframe.loc[i,'SMA_10']) else 0
        op_df.loc[i,'WMA_10']=1 if (dataframe.loc[i,'Adj Close']>dataframe.loc[i,'WMA_10']) else 0
        op_df.loc[i,'MACD']=1 if (dataframe.loc[i,'MACD']>dataframe.loc[i-1,'MACD']) else 0
        op_df.loc[i,'stoch_K']=1 if (dataframe.loc[i,'stoch_K']>dataframe.loc[i-1,'stoch_K']) else 0
        op_df.loc[i,'Momentum']=1 if (dataframe.loc[i,'Momentum']>0) else 0
        op_df.loc[i,'A/D']=1 if (dataframe.loc[i,'A/D']>dataframe.loc[i-1,'A/D']) else 0
        op_df.loc[i,'Volume']=1 if (dataframe.loc[i,'Volume']>dataframe.loc[i-1,'Volume']) else 0
        op_df.loc[i,'Fed Rate']=1 if (dataframe.loc[i,'Fed Rate']>dataframe.loc[i-1,'Fed Rate']) else 0
        op_df.loc[i,'Un Empl']=1 if (dataframe.loc[i,'Un Empl']>dataframe.loc[i-1,'Un Empl']) else 0
        op_df.loc[i,'Adj Close']=1 if (dataframe.loc[i+1,'Adj Close']>dataframe.loc[i,'Adj Close']) else 0
    # drop first 10 columns due to nan
    op_df = op_df[10:]
    return op_df

if __name__ == '__main__':
    ### file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\\')
    file_name = 'sp500_monthly.csv'
    new_file_name = 'sp500_monthly_op.csv'
    sp_df = pd.read_csv(file_name)
    
    
    ### Generate the columns, calculate the technical indactors
    sp_df = gen_df(sp_df)  
        
    ### Generate opinion dataframe, which use (-1,1) to measure the upward and downward trend
    op_df = gen_op_df(sp_df)
    op_df.to_csv(new_file_name, index=True, header=True, index_label='index')
    
    # op_df = pd.read_csv(new_file_name, index_col='index')
    ### Training and Testing Set
    random.seed(0)
    sample_index = random.sample(list(op_df.index),int(0.7*len(op_df.index)))
    op_df_train = op_df.ix[sample_index]
    op_df_test = op_df.drop(sample_index)
    columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D', 'Volume', 'Fed Rate', 'Un Empl']
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
    
    