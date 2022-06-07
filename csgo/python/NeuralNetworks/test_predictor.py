import os, sys
import os.path as path
import numpy as np
from keras import layers
from tensorflow import keras

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1337

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def getModel():
    inputLayer = layers.Input((30,512))
    
    x = layers.LSTM(64, return_sequences=True)(inputLayer)
    x = layers.LSTM(64)(inputLayer)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(2, activation="sigmoid")(x)
    
    model = keras.Model(inputLayer, out)
    opt = keras.optimizers.Adam(learning_rate=0.00001)
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

def test(model: keras.Model, trainData, batchSize, checkpointPath):
    model.summary()
    model.load_weights(checkpointPath).expect_partial()
    preds = model.predict(trainData[0],
        batch_size=batchSize,
    )
    wins = 0
    losses = 0
    avgwins = 0
    avgLoss = 0
    for i in preds:
        avgwins += i[0]
        avgLoss += i[1]
        if i[0] > i[1]:
            wins += 1
        else:
            losses += 1
            
    print(wins, losses, avgwins/len(preds), avgLoss/len(preds))

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
    test(autoencoder, trainDataSet, 1, checkpointPath)

if __name__ == "__main__":
    main()