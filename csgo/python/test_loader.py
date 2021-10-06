import numpy as np
import os.path as path
import os

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def toOneHot(arr):
    shape = (arr.size, arr.max() + 1)
    oneHot = np.zeros(shape)
    rows = np.arange(arr.size)
    oneHot[rows, arr] = 1
    return oneHot

MAP_DEPTH = 5
MAP_SCALE = 128

x = '../first_data_set/vectors/reducedTestSplits/scores'
out = 'D:\Projects\CSGO_Map_Generator\Images_data_set\TestTargets'
d = getAllFiles(x)
for i in range(len(d)):
    p = path.join(out, "targets_%d.npy"%i)
    z = np.load(path.join(x, d[i]))
    np.save(p, z)
