from ntpath import join
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
    oneHot = np.zeros(shape, dtype=np.uint8)
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
    trainDataDir = sys.argv[1]
    trainTargetDir = sys.argv[2]

    tdFiles = [path.join(trainDataDir, f) for f in getAllFiles(trainDataDir)]
    ttFiles = [path.join(trainTargetDir, f) for f in getAllFiles(trainTargetDir)]
    
    trainDataSet = tf.data.Dataset.from_generator(
        generator=lambda: generate_batches(tdFiles, ttFiles, 256),
        output_types=(np.uint8, np.uint8),
        output_shapes=([None, 30, 15, 128, 128], [None, 2])
    )
    for x, y in trainDataSet:
        print(x.shape, y.shape)

if __name__ == "__main__":
    main()