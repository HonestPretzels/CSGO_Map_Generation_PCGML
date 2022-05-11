import os
from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf
import sys
import os.path as path

FLAT_LENGTH = 15*128*128
BATCH_SIZE = 2
WEIGHT_CONSTANT = np.ones((BATCH_SIZE, 15, 128, 128), dtype=np.float32)
# Multiply only the player dimensions by 1000 to increase their importance
# WEIGHT_CONSTANT[:,5:15,:,:] *= 100
# Multiply the Non-main levels by 100 to increase their importance relative to the main floor
WEIGHT_CONSTANT[:,0:5,:,:] *= 0
WEIGHT_CONSTANT = tf.convert_to_tensor(WEIGHT_CONSTANT, dtype=tf.float32)

def weighted_MSE(y_true, y_pred):
    # Copy to keep the originals as they are supposed to be
    y_true_weighted = y_true * WEIGHT_CONSTANT
    y_pred_weighted = y_pred * WEIGHT_CONSTANT
    # MSE across all axes except the batch. Returns 30 MSEs with a batch size of 30
    loss = tf.reduce_mean(tf.square(y_true_weighted - y_pred_weighted), axis=[1,2,3])
    return loss


def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

# Adapted from here https://towardsdatascience.com/thousands-of-csv-files-keras-and-tensorflow-96182f7fabac on 06/10/2021
def generate_batches(dataFiles, labelFiles, batch_size):
    counter = 0

    while counter < len(dataFiles):
        xname = dataFiles[counter]
        yname = labelFiles[counter]
        counter += 1
        data = np.load(xname, allow_pickle=True)
        data = data.reshape(data.shape[0]*30, 3, 128, 128)
        labels = np.load(yname, allow_pickle=True)
        labels = labels.reshape(labels.shape[0]*30, 3, 128, 128)
        maxLength = data.shape[0] - (data.shape[0] % batch_size)
        for i in range(0, maxLength, batch_size):
            x = data[i:i+batch_size]
            y = labels[i:i+batch_size]
            
            yield x, y

def genModel():
    # Currently based off the architecture here: https://keras.io/examples/vision/autoencoder/
    input_img = keras.Input(shape=(3,128,128))
    # Encoder
    encoded = layers.Conv2D(8, (3,3), strides=2, activation="relu", padding="same", data_format="channels_first")(input_img)
    encoded = layers.Conv2D(16, (3,3), strides=2, activation="relu", padding="same", data_format="channels_first")(encoded)
    encoded = layers.Conv2D(32, (3,3), strides=2, activation="relu", padding="same", data_format="channels_first")(encoded)
    encoded = layers.Conv2D(32, (3,3), strides=2, activation="relu", padding="same", data_format="channels_first")(encoded)
    encoded = layers.Conv2D(32, (3,3), strides=2, activation="relu", padding="same", data_format="channels_first")(encoded)
    latent = layers.Flatten()(encoded)
    latent = layers.Dense(512, activation="softmax", name="latentspace")(latent)
    

    # Decoder
    decoded = layers.Reshape((32,4,4))(latent)
    decoded = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2DTranspose(16, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2DTranspose(8, (3,3), strides=2, activation="relu", data_format="channels_first", padding="same")(decoded)
    decoded = layers.Conv2D(3, (3,3), activation="relu", padding="same", data_format="channels_first")(decoded)
    # encoded = layers.Flatten()(encoded)
    # decoded = layers.Dense(32, activation="relu")(encoded)
    # decoded = layers.Dense(1, activation="linear")(encoded)
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, latent)

    return autoencoder, encoder

def train(model: keras.Model, trainData, batchSize, checkpointPath):
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        save_weights_only=True)

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="mean_squared_error")
    model.summary()
    if os.listdir(checkpointPath):
        model.load_weights(checkpointPath)
    model.fit(trainData,
        verbose=1,
        epochs=15,
        batch_size=batchSize,
        shuffle=True,
        callbacks=[model_checkpoint_callback]
    )

def main():
    trainDataDir = sys.argv[1]
    trainLabelsDir = sys.argv[2]
    checkpointPath = sys.argv[3]

    trainDataX = [path.join(trainDataDir, f) for f in getAllFiles(trainDataDir)]
    trainDataY = [path.join(trainLabelsDir, f) for f in getAllFiles(trainLabelsDir)]

    
    batchSize = BATCH_SIZE # Number of single seconds to look at

    

    trainDataSet = tf.data.Dataset.from_generator(
        generator=lambda: generate_batches(trainDataX, trainDataY, batchSize),
        output_types=(np.float32, np.float32),
        output_shapes=([batchSize,3,128,128], [batchSize,3,128,128])
    )

    autoencoder, _ = genModel()
    train(autoencoder, trainDataSet, 30, checkpointPath)

if __name__ == "__main__":
    with tf.device("gpu:0"):
        main()