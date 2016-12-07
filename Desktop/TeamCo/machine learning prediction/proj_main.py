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
import proj_func as f


if __name__ == '__main__':
    ### file path
    os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\\')
    file_name = 'appl.csv'
    new_file_name = 'appl_op.csv'
    orig_df = pd.read_csv(file_name)
    
    ### Generate the columns, calculate the technical indactors
    ra_df = f.gen_df(orig_df)  
        
    ### Generate opinion dataframe, which use (-1,1) to measure the upward and downward trend
    # op_df = f.gen_op_df(ra_df)
    # op_df.to_csv(new_file_name, index=True, header=True, index_label='index')
    
    op_df = pd.read_csv(new_file_name, index_col='index')
    
    result_arr = np.zeros((1,5))
    accuracy_df_ann =pd.DataFrame(np.zeros((1,1)), columns = ['ANN with backpropagation'])
    
    
    for i in range(201, len(op_df.index)):
        Y_result = f.tune_para(op_df, i)
        result_arr = np.append(result_arr, Y_result, axis=0)
    Y_result_df = pd.DataFrame(result_arr, columns = ['True value', 'SVC with linear kernel','LinearSVC (linear kernel)', 'SVC with RBF kernel','SVC with polynomial'])
    
    # Calculate Accuracy
    Accuracy = pd.DataFrame(np.zeros((1,4)), columns = ['SVC with linear kernel','LinearSVC (linear kernel)',
                                             'SVC with RBF kernel','SVC with polynomial'])
    for i in range(4):
        Accuracy.iloc[0,i] = sum(Y_result_df.iloc[:,0] == Y_result_df.iloc[:,i])/len(Y_result_df.iloc[:,0])
    
    
'''
    j = 0
    for i in set(op_df['Year']):
        print("Year_%s" %i)
        yr_df = op_df[op_df['Year']==i]
        X_train, y_train, accuracy, precision, recall = f.tune_para(yr_df)
        result_df_svm.iloc[j,] = [accuracy, precision, recall]
        j += 1

    
    for i in set(op_df['Year']):
        yr_df = op_df[op_df['Year']==i]
        Z, Y_result = f.para_ann(yr_df)
        accuracy_df_ann = accuracy_df_ann.append(Z, ignore_index=True)

    
    ### Output
    writer = pd.ExcelWriter('result.xlsx', engine = 'xlsxwriter')
    pd.DataFrame(Y_result, columns = ['True value','SVC with linear kernel','LinearSVC (linear kernel)',
                                      'SVC with RBF kernel','SVC with polynomial']).to_excel(writer, '%s' %'result')
    writer.save()
    
'''
