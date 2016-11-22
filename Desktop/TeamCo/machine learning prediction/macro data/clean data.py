# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:10:21 2016

@author: ZFang
"""

import pandas as pd
import os



os.chdir(r'C:\Users\ZFang\Desktop\TeamCo\machine learning prediction\macro data\\')
df = pd.pandas.read_csv('temp.csv')

for i in range(len(df['DATE'])):
    for j in range(len(df['DATE.3'])):
        if df['DATE.3'].iloc[j] ==  df['DATE'].iloc[i]:
            df['New'].iloc[i:i+5] = df['STLFSI'].iloc[j]
            print('Pass_%d' %i)

            
df.to_csv('clean data.csv')
