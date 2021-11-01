import os
import keras
from keras import layers
import numpy as np
import tensorflow as tf
import sys
import os.path as path

FLAT_LENGTH = 15*128*128

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
def generate_batches(dataFiles, batch_size):
    counter = 0

    while True:
        dfname = dataFiles[counter]
        counter = (counter + 1) % len(dataFiles)
        data = np.load(dfname, allow_pickle=True)
        data = data.reshape(data.shape[0]*30, 15, 128, 128)
        maxLength = data.shape[0] - (data.shape[0] % batch_size)
        for i in range(0, maxLength, batch_size):
            x = data[i:i+batch_size]
            yield x, x

def genModel():
    # Currently based off the architecture here: https://keras.io/examples/vision/autoencoder/
    input_img = keras.Input(shape=(15,128,128))
    # Encoder
    encoded = layers.Conv2D(32, (3,3), activation="relu", padding="same", data_format="channels_first")(input_img)
    encoded = layers.MaxPooling2D((2,2), padding="same", data_format="channels_first")(encoded)
    encoded = layers.Conv2D(32, (3,3), activation="relu", padding="same", data_format="channels_first")(encoded)
    encoded = layers.MaxPooling2D((2,2), padding="same", data_format="channels_first")(encoded)

    # Decoder
    decoded = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(encoded)
    decoded = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2D(15, (3,3), activation="sigmoid", padding="same", data_format="channels_first")(decoded)
    autoencoder = keras.Model(input_img, decoded)

    return autoencoder

def train(model, trainData, batchSize):
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
    model.fit(trainData, trainData,
        epochs=50,
        batch_size=batchSize,
        shuffle=True,
    )

def main():
    trainDataDir = sys.argv[1]

    trainDataFiles = [path.join(trainDataDir, f) for f in getAllFiles(trainDataDir)]

    
    batchSize = 30 # Number of single seconds to look at

    trainDataSet = np.load(trainDataFiles[0])
    trainDataSet = trainDataSet.reshape(trainDataSet.shape[0]*30, 15, 128, 128)

    trainDataSet = trainDataSet[:360]
    print(trainDataSet.shape, trainDataSet.dtype)

    # trainDataSet = tf.data.Dataset.from_generator(
    #     generator=lambda: generate_batches(trainDataFiles, batchSize),
    #     output_types=(np.uint8, np.uint8),
    #     output_shapes=([batchSize,15,128,128], [batchSize,15,128,128])
    # )

    # testDataSet = tf.data.Dataset.from_generator(
    #     generator=lambda: generate_batches(testDataFiles, batchSize),
    #     output_types=(np.uint8, np.uint8),
    #     output_shapes=([batchSize,15,128,128], [batchSize,15,128,128])
    # )

    autoencoder = genModel()
    train(autoencoder, trainDataSet, batchSize)

if __name__ == "__main__":
    with tf.device("gpu:0"):
        main()