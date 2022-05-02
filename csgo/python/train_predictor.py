import os, sys
import os.path as path
import numpy as np
from keras import layers
from tensorflow import keras

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def getModel():
    inputLayer = layers.Input(512)
    
    x = layers.Dense(256, activation="relu")(inputLayer)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(2, activation="sigmoid")(x)
    
    model = keras.Model(inputLayer, out)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model
    
def generate_batches(dataFiles, labelFiles, batchSize):
    counter = 0

    while counter < len(dataFiles):
        xname = dataFiles[counter]
        yname = labelFiles[counter]
        counter += 1
        data = np.load(xname, allow_pickle=True)
        labels = np.load(yname, allow_pickle=True)
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
    model.fit(trainData[0], trainData[1],
        validation_split=0.1,
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

    trainDataX = np.load([path.join(trainDataDir, f) for f in getAllFiles(trainDataDir)][0])
    trainDataY = np.load([path.join(trainLabelsDir, f) for f in getAllFiles(trainLabelsDir)][0])
    trainDataSet = (trainDataX, trainDataY)
    
    ## Uncomment this code to use mutlifile data
    # trainDataX = [path.join(trainDataDir, f) for f in getAllFiles(trainDataDir)]
    # trainDataY = [path.join(trainLabelsDir, f) for f in getAllFiles(trainLabelsDir)]
    
    
    # trainDataSet = tf.data.Dataset.from_generator(
    #     generator=lambda: generate_batches(trainDataX, trainDataY, 32),
    #     output_types=(np.float32, np.float32),
    #     output_shapes=([32,512], [32,2])
    # )

    autoencoder = getModel()
    train(autoencoder, trainDataSet, 32, checkpointPath)

if __name__ == "__main__":
    main()