import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import InputLayer, Dense, ReLU, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


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
filePath = currentPath + '\\testData\\'
dataList = os.listdir('testData')

# Initialize labels array
y = np.zeros(len(dataList))
X = np.empty((1,990), float)

# Call the method to adjust the dataset and split it into train/test parts
X_test, y_test = prepare_dataset(X, y)

# Setup the model
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
# Load weights
model.load_weights("nnData/weights4.h5")

model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['acc'])


score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
