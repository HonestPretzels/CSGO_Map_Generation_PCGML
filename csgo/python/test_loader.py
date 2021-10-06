import numpy as np

def toOneHot(arr):
    shape = (arr.size, arr.max() + 1)
    oneHot = np.zeros(shape)
    rows = np.arange(arr.size)
    oneHot[rows, arr] = 1
    return oneHot

MAP_DEPTH = 5
MAP_SCALE = 128

d = np.load('..\\first_data_set\\vectors\\reducedTestSplits\\scores\\scores_1_1.npy', allow_pickle=True)

print(toOneHot(d).shape)
# finishedMaps = np.empty((len(maps), MAP_DEPTH, MAP_SCALE, MAP_SCALE))
# for i in range(len(maps)):
#     finishedMaps[i] = mapMapping[maps[i]]
