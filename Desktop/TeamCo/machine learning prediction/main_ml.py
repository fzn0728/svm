# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:07:34 2016

@author: ZFang
"""


from datetime import datetime
import pandas_datareader.data as wb
import pandas as pd
import numpy as np
from sklearn import svm


def cal_return(p):
    daily_return = (p['Adj Close'] - p['Adj Close'].shift(1))/p['Adj Close']
    return daily_return


def get_etf(stocklist, start):
    '''
    Pull the data using Yahoo Fiannce API, given ticker(could be a list) and start date 
    '''
    start = datetime.strptime(start, '%m/%d/%Y')
    # end = datetime.strptime(end, '%m/%d/%Y')
    end = datetime.today()
    p = wb.DataReader(stocklist,'yahoo',start,end)
    p['Return'] = cal_return(p)
    return p

def gen_ind(dataframe, n=10):
    '''
    Generate the technical indicator from price
    '''
    dataframe['SMA_10'] = dataframe['Adj Close'].rolling(window=n).mean()
    mo_arr = dataframe['Adj Close'][9:].values - dataframe['Adj Close'][:-9].values
    dataframe['Momentum'] = np.zeros(len(dataframe.index))
    dataframe.ix[9:,'Momentum'] = mo_arr
    dataframe['LL'] = dataframe['Adj Close'].rolling(window=n).min()
    dataframe['HH'] = dataframe['Adj Close'].rolling(window=n).max()
    dataframe['stoch_K'] = 100 * (dataframe['Adj Close']- dataframe['LL'])/(dataframe['HH']- dataframe['LL'])
    for i in range(9,len(dataframe.index)):
        dataframe.ix[i,'WMA_10'] = (10*dataframe.ix[i,'Adj Close']+9*dataframe.ix[i-1,'Adj Close']+
                            8*dataframe.ix[i-2,'Adj Close']+7*dataframe.ix[i-3,'Adj Close']+
                            6*dataframe.ix[i-4,'Adj Close']+5*dataframe.ix[i-5,'Adj Close']+
                            4*dataframe.ix[i-6,'Adj Close']+3*dataframe.ix[i-7,'Adj Close']+
                            2*dataframe.ix[i-8,'Adj Close']+dataframe.ix[i-9,'Adj Close'])/sum(range(1,11,1))
    dataframe['EMA_12'] = dataframe['Adj Close'].ewm(span=12).mean()
    dataframe['EMA_26'] = dataframe['Adj Close'].ewm(span=26).mean()
    dataframe['DIFF'] = dataframe['EMA_12'] - dataframe['EMA_26']
    dataframe['MACD'] = np.zeros(len(dataframe.index))
    dataframe['A/D'] = np.zeros(len(dataframe.index))
    for i in range(1,len(dataframe.index)):
        dataframe.ix[i,'MACD'] = dataframe.ix[i-1,'MACD'] + 2/(n+1)*(dataframe.ix[i,'DIFF']-dataframe.ix[i-1,'MACD'])
        dataframe.ix[i,'A/D'] = (dataframe.ix[i,'High']-dataframe.ix[i-1,'Adj Close'])/(dataframe.ix[i,'High']-dataframe.ix[i,'Low'])
    return dataframe

def gen_op_df(dataframe):
    '''
    Generate binary indicator based on technical indicator value
    '''
    op_df = pd.DataFrame(np.zeros((len(dataframe),10)), columns=['SMA_10', 'Momentum', 
                         'stoch_K', 'WMA_10', 'MACD', 'A/D', 'Volume', 'Adj Close', 'Adj Close Value', 'Return'])
    op_df.index = dataframe.index
    op_df['Adj Close Value'] = dataframe['Adj Close']
    op_df['Return'] = dataframe['Return']
    op_df['Year'] = dataframe.index.year
    for i in range(10,len(dataframe.index)-1):
        op_df.ix[i,'SMA_10']=1 if (dataframe.ix[i,'Adj Close']>dataframe.ix[i,'SMA_10']) else -1
        op_df.ix[i,'WMA_10']=1 if (dataframe.ix[i,'Adj Close']>dataframe.ix[i,'WMA_10']) else -1
        op_df.ix[i,'MACD']=1 if (dataframe.ix[i,'MACD']>dataframe.ix[i-1,'MACD']) else -1
        op_df.ix[i,'stoch_K']=1 if (dataframe.ix[i,'stoch_K']>dataframe.ix[i-1,'stoch_K']) else -1
        op_df.ix[i,'Momentum']=1 if (dataframe.ix[i,'Momentum']>0) else -1
        op_df.ix[i,'A/D']=1 if (dataframe.ix[i,'A/D']>dataframe.ix[i-1,'A/D']) else -1
        op_df.ix[i,'Volume']=1 if (dataframe.ix[i,'Volume']>dataframe.ix[i-1,'Volume']) else -1
        op_df.ix[i,'Adj Close']=1 if (dataframe.ix[i+1,'Adj Close']/dataframe.ix[i,'Adj Close']>1) else -1
    # drop first 10 columns due to nan
    op_df = op_df[10:]
    return op_df

def tune_para(dataframe, i):
    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    columns = ['SMA_10','Momentum','stoch_K','WMA_10','MACD','A/D','Volume']
    X = dataframe[columns].as_matrix()
    y = dataframe['Adj Close'].as_matrix()
    X_train = X[i-200:i]
    y_train = y[i-200:i]
    X_test = X[i:i+1]
    y_test = y[i:i+1]
    
    ### Train four kinds of SVM model
    C = 1  # SVM regularization parameter
    svc = svm.SVC(cache_size = 1000, kernel='linear', C=C).fit(X_train, y_train)
    rbf_svc = svm.SVC(cache_size = 1000, kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
    poly_svc = svm.SVC(cache_size = 1000, kernel='poly', degree=3, C=C).fit(X_train, y_train)
    lin_svc = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False, C=C).fit(X_train, y_train)
    Y_result = y_test
    
    
    ### Make the prediction
    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        pred = clf.predict(X_test)
        Y_result = np.vstack((Y_result, np.array(pred))) # append prediction on Y_result
    return Y_result.T
    
def svm_result(op_df):
    result_arr = np.zeros((1,5))    
    for i in range(200, len(op_df.index)):
        Y_result = tune_para(op_df, i)
        result_arr = np.append(result_arr, Y_result, axis=0)
    Y_result_df = pd.DataFrame(result_arr, columns = ['True value', 'SVC with linear kernel','LinearSVC (linear kernel)', 'SVC with RBF kernel','SVC with polynomial'])
    Y_result_df['Return'] = op_df.ix[199:,'Return'].values
    return Y_result_df
    
if __name__ == '__main__':
    new_file_name = 'appl_op.csv'
    price_df = get_etf('AMZN','12/05/2010')
    indicator_df =  gen_ind(price_df, n=10)
    op_df = gen_op_df(indicator_df)
    
    # Save the file and read it next time
    op_df.to_csv(new_file_name, index=True, header=True, index_label='index')
    op_df = pd.read_csv(new_file_name, index_col='index')
    Y_result_df = svm_result(op_df)


    ### Calculate the return based on the strategy
    Accuracy = pd.DataFrame(np.zeros((2,5)), columns = ['True value','SVC with linear kernel','LinearSVC (linear kernel)',
                                             'SVC with RBF kernel','SVC with polynomial'])
    for i in range(5):
        Accuracy.iloc[0,i] = sum(Y_result_df.iloc[:,0]==Y_result_df.iloc[:,i])/len(Y_result_df.iloc[:,0])
        if i==0:
            Accuracy.iloc[1,i] = np.prod(1+ Y_result_df['Return'][2:].values)
        else:
            Accuracy.iloc[1,i] = np.prod(1+ Y_result_df.iloc[1:-1,i].values*Y_result_df['Return'][2:].values)



    # bol = 1+ Y_result_df.iloc[1:-1,i] * Y_result_df['Return'].shift(1)[2:]
    # bol.prod()










