import os, sys
from os import path
import numpy as np
from tqdm import tqdm

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def splitAndSave(f, out, n):
    arr = np.load(f, allow_pickle=True)
    divisor = len(arr) // n
    for i in range(n):
        newArr = arr[i*divisor: (i+1)*divisor]
        np.save("%s_%d.npy"%(out, i+1), newArr,  allow_pickle=True)
    os.remove(f)

def main():
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    numSplits = int(sys.argv[3])
    # subFolders = ['breakpoints', 'data', 'maps', 'scores']

    # for folder in subFolders:
    for f in tqdm(getAllFiles(inputDir)):
        out = path.join(outputDir, path.basename(f).split(".")[0])
        splitAndSave(path.join(inputDir, f), out, numSplits)

if __name__ == "__main__":
    main()