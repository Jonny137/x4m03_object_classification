import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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

# Navigation folder
currentPath = str(Path().absolute())

# Read the single frame to be evaluated
filepath = currentPath + '\\singleFrame\\single_frame.csv'
X_pred = np.genfromtxt(filepath, delimiter=',')
X_pred = np.reshape(X_pred, (1,990))

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
# Show summary
model.summary()
# Load weights
model.load_weights("nnData/weights.h5")

model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['acc'])

# Perform prediction on a single set frame
# Evaluation is based on end sigmoid actiavtion function with threshold 0.5
# If the mod_pred is above 0.5 -> result is 1 and object is considered rectangular
# Else result is 0 and object is considered round
prediction = model.predict(X_pred)
mod_pred = round(float(prediction[0]))
if mod_pred == 1:
    print('Rectangular object, Predicted Acc: %2.2f' % (prediction[0]))
else:
    print('Round object, Predicted Acc: %2.2f' % (prediction[0]))
