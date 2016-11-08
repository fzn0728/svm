# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:07:34 2016

@author: ZFang
"""
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\\')
    sp_df = pd.read_csv('sp500.csv')
    
    
    
    # SImple Moving Average
    sp_df['SMA_10'] = sp_df['Adj Close'].rolling(window=10).mean()
    mo_arr = sp_df['Adj Close'][9:].values - sp_df['Adj Close'][:-9].values
    sp_df['Momentum'] = 0
    sp_df.loc[9:,'Momentum'] = mo_arr
    sp_df['LL'] = sp_df['Adj Close'].rolling(window=10).min()
    sp_df['HH'] = sp_df['Adj Close'].rolling(window=10).max()
    sp_df['stoch_K'] = 100 * (sp_df['Adj Close']- sp_df['LL'])/(sp_df['HH']- sp_df['LL'])
    