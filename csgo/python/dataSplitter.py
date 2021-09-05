import sys
import os
import numpy as np

NUM_BATCHES = 40

def main():
    dPath = sys.argv[1]
    mPath = sys.argv[2]
    tPath = sys.argv[3]
    bPath = sys.argv[4]

    data = np.load(dPath + "\data.npy")
    splitAndSave(data, dPath, "data")
    maps = np.load(mPath + "\maps.npy")
    splitAndSave(maps, mPath, "maps")
    targets = np.load(tPath + "\scores.npy")
    splitAndSave(targets, tPath, "scores")
    breakPoints = np.load(bPath + "\\breakpoints.npy", allow_pickle=True)
    splitAndSave(breakPoints, bPath, "breakpoints")

def splitAndSave(data, dPath, name):
    print(dPath)
    extra = data.shape[0] % NUM_BATCHES
    if extra != 0:
        data = data[:-extra]
    print(data.shape)
    for i, arr in enumerate(np.split(data, NUM_BATCHES)):
        path = os.path.join(dPath, "%s_%d.npy"%(name,i))
        np.save(path, arr)

if __name__ == "__main__":
    main()