import numpy as np
import sys, os
import os.path as path
from matplotlib import pyplot as plt
from PIL import Image

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    t1labels = []
    t2labels = []
    bothlabels = []
    
    files = getAllFiles(inputPath)
    for f in files:
        x = np.load(path.join(inputPath,f))
        print(x.shape)
        for split in x:
            for second in split:
                averageTeam1Count = np.sum(second[1,:,:])
                averageTeam2Count = np.sum(second[2,:,:])
                t1labels.append(averageTeam1Count)
                t2labels.append(averageTeam2Count)
                bothlabels.append(averageTeam1Count+averageTeam2Count)
        print(len(t1labels), len(t2labels), len(bothlabels))
    t1labels = np.array(t1labels)
    t2labels = np.array(t2labels)
    bothlabels = np.array(bothlabels)
    
    print(t1labels.shape, t2labels.shape, bothlabels.shape)
    np.save(path.join(outputPath, "team1CountLabels.npy"), t1labels)
    np.save(path.join(outputPath, "team2CountLabels.npy"), t2labels)
    np.save(path.join(outputPath, "bothCountLabels.npy"), bothlabels)

if __name__ == "__main__":
    main()