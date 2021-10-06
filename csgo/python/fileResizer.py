import os, sys
from os import curdir, path
import numpy as np

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def splitAndSave(f, out, n):
    arr = np.load(f, allow_pickle=True)
    divisor = len(arr) // n
    for i in range(n):
        newArr = arr[i*divisor: (i+1)*divisor]
        np.save("%s_%d.npy"%(out, i+1), newArr,  allow_pickle=True)

def main():
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    numSplits = int(sys.argv[3])
    subFolders = ['breakpoints', 'data', 'maps', 'scores']

    for folder in subFolders:
        curdir = path.join(inputDir, folder)
        for f in getAllFiles(curdir):
            out = path.join(path.join(outputDir, folder), path.basename(f).split(".")[0])
            splitAndSave(path.join(curdir, f), out, numSplits)
    pass

if __name__ == "__main__":
    main()