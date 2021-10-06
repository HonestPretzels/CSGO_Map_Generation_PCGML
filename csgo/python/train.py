from tensorflow import keras
import numpy as np
import tensorflow as tf
import sys, os
import os.path as path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

# From here https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python on 06/10/2021
def toOneHot(arr):
    shape = (arr.size, arr.max() + 1)
    oneHot = np.zeros(shape)
    rows = np.arange(arr.size)
    oneHot[rows, arr] = 1
    return oneHot

# Adapted from here https://towardsdatascience.com/thousands-of-csv-files-keras-and-tensorflow-96182f7fabac on 06/10/2021
def generate_batches(dataFiles, targetFiles, batch_size):
    counter = 0

    while True:
        dfname = dataFiles[counter]
        tfname = targetFiles[counter]
        counter = (counter + 1) % len(dataFiles)

        data = np.load(dfname, allow_pickle=True)
        target = np.load(tfname, allow_pickle=True)

        for idx in range(0, data.shape[0], batch_size):
            x = data[idx:(idx+batch_size)]
            y = toOneHot(target[idx:(idx+batch_size)])

            yield x, y


def main():
    dataDir = sys.argv[0]
    targetDir = sys.argv[0]
    pass

if __name__ == "__main__":
    main()