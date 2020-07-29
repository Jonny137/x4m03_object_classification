# THIS SCRIPT IS USED FOR PREPARING A SINGLE FRAME FOR CALL PREDICT INSIDE
# THE singlePredict.py SCRIPT
import os
import numpy as np
from pathlib import Path

# Navigation folders
currentPath = str(Path().absolute())
filePath = currentPath + '\\recordedData\\'
singleFolder = 'singleFrame/'

# Folder and file related variables
recList = os.listdir('recordedData')
arr = np.empty((2,99), float)

new_frame = input('Insert filename for preparation: ')
for filename in recList:
    if new_frame in recList:
        os.remove(singleFolder + 'single_frame.csv')
        break
    else:
        continue

for filename in recList:
    if new_frame in filename and 'phi' not in filename:
        readLoc = filePath + filename
        data = np.genfromtxt(readLoc,  delimiter=',').transpose()
        arr = np.vstack((arr,data))

empty_lst = [0,1]
arr = np.delete(arr, empty_lst, axis=0)
num_lst = [i for i in range(arr.shape[0]) if (i % 2) == 0]
arr = np.delete(arr, num_lst, axis=0)

newName = singleFolder + 'single_frame.csv'
np.savetxt(newName, arr, delimiter=',')
