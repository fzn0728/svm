# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:57:04 2016

@author: ZFang
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


iris = datasets.load_iris()
X = iris.data[:,:2]

y = iris.target

h = .02 # step size in the mesh

## We create oan instance of SVM and fit out data. We do not scale our 
## data since we want to plot the support vectors

C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X,y)
rbf_svc = svm.SVC(kernel='rbf',gamma=0.7, C=C).fit(X,y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,y)
lin_svc = svm.LinearSVC(C=C).fit(X,y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx,yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.RdYlGn, alpha=0.8)
    
    # plot also the training poits
    plt.scatter(X[:, 0],X[:, 1],c=y, cmap=plt.cm.RdYlGn)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    