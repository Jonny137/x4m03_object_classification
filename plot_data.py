import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt


# Navigation folders
currentPath = str(Path().absolute())
filePath = currentPath + '\\recordedData\\'
modPath = currentPath + '\\modifiedData\\'

# Folder and file related variables
recList = os.listdir('recordedData')
modList = os.listdir('modifiedData')
dataFolder = 'recordedData/'
modifiedFolder = 'modifiedData/'

data = []

data.append(np.genfromtxt(dataFolder + 'roundPiggybank_90cm_amp.csv', delimiter=','))

for element in data:
    for i in range(13):
        element[:,1][i] = 0

plt.plot(data[0][:,1], label='roundPiggybank')
plt.legend()
plt.grid()
plt.title('roundPiggybank_90cm_amp.csv')
plt.show()
