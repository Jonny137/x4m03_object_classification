import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Navigation folders
currentPath = str(Path().absolute())
filePath = currentPath + '\\recordedData\\'

# Folder and file related variables
recList = os.listdir('recordedData')
modList = os.listdir('modifiedData')
dataFolder = 'recordedData/'
labelFolder = 'labeledData/'
modifiedFolder = 'modifiedData/'
uniqueFiles = []
fileSet = []

# Read all recorded files and form a set of unique names for further processing
def sort_rec_files():
    for name in recList:
        temp = name.split('_')[0]
        if temp not in uniqueFiles:
            uniqueFiles.append(temp)

    for unique in uniqueFiles:
        temp = []
        for name in recList:
            if (unique in name and 'phi' not in name
                                         and unique == name.split('_')[0]):
                temp.append(name)

        fileSet.append(temp)


# Read data from the fileSet variable and combine data into single csv with the
# unique name
def read_save_unique_data(files):
    arr = np.empty((2,99), float)

    for filename in files:
        readLoc = dataFolder + filename
        data = np.genfromtxt(readLoc,  delimiter=',').transpose()
        arr = np.vstack((arr,data))

    empty_lst = [0,1]
    arr = np.delete(arr, empty_lst, axis=0)
    num_lst = [i for i in range(arr.shape[0]) if (i % 2) == 0]
    arr = np.delete(arr, num_lst, axis=0)

    newName = modifiedFolder + files[0].split('_')[0] + '.csv'
    np.savetxt(newName, arr, delimiter=',')


# Append classification label and save new .csv in labeledData folder
def append_class_label():
    # Label value arrays to append as last column
    # Rectangular objects are labeled with 1 while the round objects are 0
    arr_ones = np.ones((10,1), dtype=int)
    arr_zeros = np.zeros((10,1), dtype=int)

    for filename in modList:
        try:
            readLoc = 'modifiedData/' + filename
            labelLoc = labelFolder + filename
            data = np.genfromtxt(readLoc,  delimiter=',')

            if 'rect' in readLoc:
                new_arr = np.append(data, arr_ones, axis=1)
                np.savetxt(labelLoc, new_arr, delimiter=',')
            elif 'round' in readLoc:
                new_arr = np.append(data, arr_zeros, axis=1)
                np.savetxt(labelLoc, new_arr, delimiter=',')
            else:
                print(filename)
        except ValueError:
            print(filename)
# Method calls
sort_rec_files()

for files in fileSet:
    read_save_unique_data(files)

append_class_label()
newlist = os.listdir('modifiedData')
print('Successful Formating!')
print('Total data samples: {}'.format(len(newlist)))
