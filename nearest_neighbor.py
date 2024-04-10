# Description: This file contains the implementation of the nearest neighbor classifier.
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold

def nearest(X_train, y_train, X_test, y_test):
    skf = StratifiedKFold(n_splits=50, random_state=42, shuffle=True)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    knn = KNeighborsClassifier(350, algorithm='brute')
    for train_index, train2_index in skf.split(X_train, y_train):
        X_train_fold, X_train2_fold = X_train[train_index], X_train[train2_index]
        y_train_fold, y_train2_fold = y_train[train_index], y_train[train2_index]
        knn.fit(X_train_fold, y_train_fold)
        knn.fit(X_train2_fold, y_train2_fold)

    y_pred = knn.predict(X_test)
   
    return y_pred
    
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)