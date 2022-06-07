import numpy as np
import sys, os
import os.path as path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputDataPath = sys.argv[1]
    inputLabelPath = sys.argv[2]
    outputPath = sys.argv[3]
    
    
    xfiles = getAllFiles(inputDataPath)
    yfiles = getAllFiles(inputLabelPath)
    for i in range(len(xfiles)):
        xf = xfiles[i]
        yf = yfiles[i]
        print(xf, yf)
        x = np.load(path.join(inputDataPath,xf))
        y = np.load(path.join(inputLabelPath,yf))
        outputSplits = []
        outputLabels = []
        for splitidx in range(x.shape[0]):
            split = x[splitidx]
            labels = y[splitidx]
            fullzero = True
            for second in split:
                if np.sum(second[1]) + np.sum(second[2]) != 0:
                    fullzero = False
            if not fullzero:
                outputSplits.append(split)
                outputLabels.append(labels)
        print(len(outputSplits), len(outputLabels))
        # np.save(path.join(path.join(outputPath,"Data"),xf), np.array(outputSplits))
        np.save(path.join(path.join(outputPath,"SeparateTeamLabels"),yf), np.array(outputLabels))
            

if __name__ == "__main__":
    main()