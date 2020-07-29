import os
import glob
from pathlib import Path
import pandas as pd

# Pathways to the folders with csv files and where future file will be stored
currentPath = str(Path().absolute())
storePath = currentPath + '\\' + 'recordedData\\'
modPath = currentPath + '\\' + 'modifiedData\\'
# List of csv filenames to be merged
fileList = os.listdir(storePath)
modList = os.listdir(modPath)

# Add header row to the data, shape column added, 0 for round, 1 for rect and 2
# for not identified
def editLabel(list):
    for filename in list:
        file = pd.read_csv(storePath + filename)
        file.columns = ['Bin_#', 'Norm_Amp']

        if 'round' in filename:
            file['Shape'] = 0
        else:
            file['Shape'] = 1

        file.to_csv(modPath + 'mod_' + filename)


def main():
    # Filter only amplitude recorded data
    ampList = [name for name in fileList if 'amp' in name]
    editLabel(ampList)

    # Merge all modified CSV into one single big CSV file
    combCSV = pd.concat([pd.read_csv(modPath + f) for f in modList])
    del combCSV['Unnamed: 0']
    combCSV.to_csv('inputData.csv')

if __name__ == '__main__':
    main()
