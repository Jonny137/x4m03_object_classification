import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import InputLayer, Dense, ReLU, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Method to prepare the dataset
def prepare_dataset(X, labels):
    i = 0
    # Prepare dataset
    for filename in dataList:
        if 'rect' in filename:
            y[i] = 1
        data = np.genfromtxt(filePath + filename, delimiter=',').flatten()
        data = np.reshape(data, (-1,990))
        X = np.concatenate((X, data), axis=0)
        i += 1
    # Clear first row
    X = np.delete(X, 0, axis=0)

    return X, y

# Navigation folders
currentPath = str(Path().absolute())
filePath = currentPath + '\\modifiedData\\'
dataList = os.listdir('modifiedData')

# Initialize labels array
y = np.zeros(len(dataList))
X = np.empty((1,990), float)

# Call the method to adjust the dataset and split it into train/test parts
X, y = prepare_dataset(X, y)

# Initialize number of estimations for fold, cross_validation score list and
# confusion matrix init
n_est = 5
crs_val_fold = []
iter = 0
conf_arr = []
history_rec = []

# Prepare the splitting based on number of estimations for StratifiedKFold
for train_ind, test_ind in StratifiedKFold(n_est, shuffle=True).split(X, y):
    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]

    # Setup and train the model
    model = Sequential([
        InputLayer((990,)),
        Dense(64),
        Dropout(0.5),
        ReLU(),
        Dense(32),
        Dropout(0.5),
        ReLU(),
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['acc'])

    weight_name = 'weights' + str(iter) + '.h5'

    # Train the model
    # Two callbackes added for weight saving and stopping training
    # if the accuraccy is not improveed over 30 iterations
    save_best = ModelCheckpoint(filepath='nnData/' + weight_name,
                                verbose=0, save_best_only=True)
    early_stopping = EarlyStopping(patience=30)
    history = model.fit(X_train, y_train,
              epochs=400,
              validation_data=(X_test, y_test),
              callbacks=[save_best, early_stopping])

    # Append model results for further processing
    history_rec.append(history)

    # Evaluate the test set for fold
    scores = model.evaluate(X_test, y_test)

    # Plot the confusion matrix and score for each fold
    iter += 1
    print("%s: %.2f%%" % ('Accuracy: ', scores[1]*100))
    crs_val_fold.append(scores[1] * 100)
    y_pred = model.predict_classes(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_arr.append(conf_matrix)

# Show the results after folds
print()
print("Total Cross Validation Acc: %.2f%% (+/- %.2f%%)"
                                % (np.mean(crs_val_fold), np.std(crs_val_fold)))

# Plot the confusion matrix for all folds (total conf_mat is sum of each folds)
print('Confusion matrix: ')
total_conf = np.zeros((2,2))
for mat in conf_arr:
    total_conf += np.array(mat)
print(total_conf)

# Compare the each fold model results and plot the best one
# Basic method of max extraction
best_acc = history_rec[0]
for elem in history_rec:
    if np.mean(elem.history['acc']) > np.mean(best_acc.history['acc']):
        best_acc = elem

# Plot the train and test accuraccy score
plt.plot(best_acc.history['acc'], label='train')
plt.plot(best_acc.history['val_acc'], label='test')
plt.plot(best_acc.history['loss'], label='train')
plt.plot(best_acc.history['val_loss'], label='test')
plt.title('Train/Test Accuracy Graph')
plt.legend()
plt.grid(True)
plt.show()
