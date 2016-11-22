# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:25:34 2016

@author: ZFang
"""

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.svm import SVC
import os 
import pandas as pd
import proj_func as f

print(__doc__)



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

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
columns = ['SMA_10','Momentum','stoch_K', 'WMA_10', 'MACD','A/D' , 'Volume']
X = op_df[columns].as_matrix()[0:100]
Y = op_df['Adj Close'].as_matrix()[0:100]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.7,1],
                     'C': [1.0]},
                    {'kernel': ['linear'], 'C': [1.0], 'cache_size': [1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2,
                       scoring='%s_macro' % score)
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
    print(classification_report(y_true, y_pred))
    print()

    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(precision)
    print(recall)
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.