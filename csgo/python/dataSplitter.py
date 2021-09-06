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
    data2 = np.load(dPath + "\data_pt2.npy")
    splitAndSave(data2, dPath, "data2")
    maps = np.load(mPath + "\maps.npy")
    splitAndSave(maps, mPath, "maps")
    maps2 = np.load(mPath + "\maps_pt2.npy")
    splitAndSave(maps2, mPath, "maps2")
    targets = np.load(tPath + "\scores.npy")
    splitAndSave(targets, tPath, "scores")
    targets2 = np.load(tPath + "\scores_pt2.npy")
    splitAndSave(targets2, tPath, "scores2")
    breakPoints = np.load(bPath + "\\breakpoints.npy", allow_pickle=True)
    splitAndSave(breakPoints, bPath, "breakpoints")
    breakPoints2 = np.load(bPath + "\\breakpoints_pt2.npy", allow_pickle=True)
    splitAndSave(breakPoints2, bPath, "breakpoints2")

def splitAndSave(data, dPath, name):
    extra = data.shape[0] % NUM_BATCHES
    if extra != 0:
        data = data[:-extra]
    for i, arr in enumerate(np.split(data, NUM_BATCHES)):
        path = os.path.join(dPath, "%s_%d.npy"%(name,i))
        np.save(path, arr)

if __name__ == "__main__":
    main()