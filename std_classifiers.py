import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

import xgboost as xgb

# Navigation folders
currentPath = str(Path().absolute())
filePath = currentPath + '\\modifiedData\\'
dataList = os.listdir('modifiedData')

# Initialize labels array
y = np.zeros(len(dataList))
X = np.empty((1,990), float)

def prepare_dataset(X, labels):
    try:
        i = 0
        # Prepare dataset
        for filename in dataList:
            if 'rect' in filename:
                y[i] = 1
            data = np.genfromtxt(filePath + filename, delimiter=',').flatten()
            data = np.reshape(data, (-1,990))
            X = np.concatenate((X, data), axis=0)

            i += 1
    except:
        print(filename)

    # Clear first row
    X = np.delete(X, 0, axis=0)

    return X, y

X, y = prepare_dataset(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

def randomForestCalc(X_train, X_test, y_train, y_test):
    print('-------------------------------------------')
    print('------------- Random Forest ---------------')

    # Random Forest Algorithm
    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    rf_acc = metrics.accuracy_score(y_test, y_pred)
    # acc_list.append(rf_acc)
    # if rf_acc > max_acc:
    #     max_acc = rf_acc
    #     max_est = n_est
    rf_prec = metrics.precision_score(y_test, y_pred)
    rf_rec = metrics.recall_score(y_test, y_pred)
    print('Accuracy: ', rf_acc)
    print('Precision: ', rf_prec)
    print('Recall: ', rf_rec)
    Fscore = 2 * rf_rec * rf_prec / (rf_rec + rf_prec)
    print('F-score: ', Fscore)
    con_mat = metrics.confusion_matrix(y_test, y_pred)
    print(con_mat)

def XGBoostCalc(X_train, X_test, y_train, y_test):
    print('-------------------------------------------')
    print('--------------- XGBoost -------------------')

    # XGBoost Algorithm
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    param = {
        'eta': 0.2,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': 2}
    steps = 100
    model = xgb.train(param, D_train, steps)
    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    xgb_acc = accuracy_score(y_test, best_preds)
    xgb_prec = precision_score(y_test, best_preds, average='macro')
    xgb_rec = recall_score(y_test, best_preds, average='macro')
    print('Accuracy = {}'.format(xgb_acc))
    print('Precision = {}'.format(xgb_prec))
    print('Recall = {}'.format(xgb_rec))
    Fscore = 2 * xgb_rec * xgb_prec / (xgb_rec + xgb_prec)
    print('F-score: ', Fscore)
    con_mat = metrics.confusion_matrix(y_test, best_preds)
    print(con_mat)
    print('-------------------------------------------')

# acc_list = []
# est_list = list(range(1,800,10))
# max_acc = 0
# max_est = 0
# for n_est in est_list:
# randomForestCalc(X_train, X_test, y_train, y_test)
XGBoostCalc(X_train, X_test, y_train, y_test)
# plt.plot(est_list, acc_list)
# plt.xlabel('Number of estimators')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.show()
