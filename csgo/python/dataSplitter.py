import sys
import os
import numpy as np

def main():
    dPath = sys.argv[1]
    mPath = sys.argv[2]
    tPath = sys.argv[3]

    data = np.load(dPath)
    print(data.shape)
    splitAndSave(data, dPath)
    maps = np.load(mPath)
    splitAndSave(maps, mPath)
    targets = np.load(tPath)
    splitAndSave(targets, tPath)

def splitAndSave(data, dPath):
    extra = data.shape[0] % 20
    if extra != 0:
        data = data[:-extra]
    print(data.shape)
    for i, arr in enumerate(np.split(data, 20)):
        path = os.path.split(dPath)[0]
        path = os.path.join(path, "data_%d.npy"%i)
        np.save(path, arr)

if __name__ == "__main__":
    main()