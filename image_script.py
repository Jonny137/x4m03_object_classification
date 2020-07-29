import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Navigation folders
currentPath = str(Path().absolute())
origPath = currentPath + '\\recordedData\\'

# Folder with files
origList = os.listdir('recordedData')

def saveBaseData():
    for filename in origList:
        if 'phi' not in filename:
            temp = origPath + filename
            data = np.genfromtxt(temp, delimiter=',')
            # First 18 bins are isolation peak samples
            for i in range(18):
                data[:,1][i] = 0
            plt.plot(data[:, 0], data[:, 1])
            plt.title(filename)
            plt.xlabel('Bin #')
            plt.ylabel('Norm. Amp')
            plt.grid()
            noExt = filename[:-4]
            imageDir = 'radarImages/' + noExt + '.png'
            plt.savefig(imageDir, bbox_inches='tight')
            plt.clf()

def peakExtCalc(temp, filename):
    data = np.genfromtxt(temp, delimiter=',')
    noExt = filename[:-4]

    # First 18 bins are isolation peak samples
    for i in range(18):
        data[:,1][i] = 0
    peak = data[:,1][0]
    peakInd = 0
    j = 0
    for i in data[:,1]:
        if i > peak:
            peak = i
            peakInd = j
        j +=1
    res = 5.05 # resolution in cm's
    distance = res * peakInd
    delay = 2 * distance / 299792458 /100
    print('--------------------------------------------')
    print(noExt)
    print(f'Peak: {peak}, Peak_Index: {peakInd}')
    print(f'Distance: {distance}cm, Delay: {delay}')
    print('--------------------------------------------')

if __name__ == '__main__':
    # saveBaseData()
    for filename in origList:
        if 'phi' not in filename:
            temp = origPath + filename
            peakExtCalc(temp, filename)
