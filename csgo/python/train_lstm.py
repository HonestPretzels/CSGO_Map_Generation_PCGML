import os, sys
import os.path as path
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def getModel():
    inputLayer = layers.Input(512)
    
    x = layers.LSTM(128, dropout=0.8, return_sequences=True)(inputLayer)
    x = layers.LSTM(128)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(2, activation="sigmoid")(x)
    
    model = keras.Model(inputLayer, out)
    model.compile(optimizer="adam", learning_rate=0.0001, loss="binary_crossentropy")
    
    return model
    
def generate_batches(dataFiles, labelFiles, batchSize):
    counter = 0

    while counter < len(dataFiles):
        xname = dataFiles[counter]
        yname = labelFiles[counter]
        counter += 1
        data = np.load(xname, allow_pickle=True)
        data = data.reshape(data.shape[0]*30, 3, 128, 128)
        labels = np.load(yname, allow_pickle=True)
        labels = labels.reshape(labels.shape[0]*30, 2, 128, 128)
        maxLength = data.shape[0] - (data.shape[0] % batchSize)
        for i in range(0, maxLength, batchSize):
            x = data[i:i+batchSize]
            y = labels[i:i+batchSize]
            
            yield x, y

def train(model: keras.Model, trainData, batchSize, checkpointPath):
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        save_weights_only=True)
    
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
    
    
    trainDataSet = tf.data.Dataset.from_generator(
        generator=lambda: generate_batches(trainDataX, trainDataY, 30),
        output_types=(np.float32, np.float32),
        output_shapes=([30,3,128,128], [30,2,128,128])
    )

    autoencoder = getModel()
    train(autoencoder, trainDataSet, 30, checkpointPath)

if __name__ == "__main__":
    main()