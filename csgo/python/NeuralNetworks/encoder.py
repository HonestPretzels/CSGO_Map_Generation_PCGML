import sys
import numpy as np
import gc
import keras.backend as K
from matplotlib import pyplot as plt
from PIL import Image
import os.path as path
import tensorflow as tf
from train_autoencoder import generate_batches, getAllFiles, genModel
from keras.utils.np_utils import to_categorical  

# gameBreaks refers to the delinieators between the games and is an array of the number of splits in each game

def visualize(preds, reals):
    for y in range(preds.shape[0]): # Sub sample of states
        for t in range(preds.shape[1]):
            for x in range(preds.shape[2]):
                # Display the real values
                # r = reals[y][t][x]
                r = np.zeros((128,128))
                print(r.shape)
                # Third layer is empty
                g = np.zeros((128,128))
                # Blue for preds
                b = preds[y][t][x]
                print(b.shape)
                
                s = sum(b.flatten())
                if s > 0.1:
                    b *= 255
                    r *= 255
                    img = np.dstack((r,g,b))
                    r_img = Image.fromarray((img).astype(np.uint8))
                    plt.imshow(r_img)
                    plt.show()
            
    
def test(datasetPath, labelsPath, checkpointPath, output):
    testDataFiles = [path.join(datasetPath, f) for f in getAllFiles(datasetPath)]
    testDataLabels = [path.join(labelsPath, f) for f in getAllFiles(labelsPath)]
    print(testDataFiles)
    AE, Encoder = genModel()
    AE.load_weights(checkpointPath).expect_partial()
    AE.summary()
    Encoder.summary()
    for i in range(len(testDataFiles)):
        x = np.load(testDataFiles[i])
        x = x.reshape(x.shape[0]*30, 3, 128, 128)
        y = np.load(testDataLabels[i])
        y = np.repeat(y, 30)
        y = y.reshape(y.shape[0], 1)
        y = to_categorical(y, 2)
        print(testDataFiles[i], testDataLabels[i], y.shape, x.shape)
        preds = Encoder.predict(x, verbose=1)
        np.save(path.join(path.join(output,"Data"), path.basename(testDataFiles[i])), preds)
        np.save(path.join(path.join(output, "Labels"), path.basename(testDataLabels[i])), y)
        gc.collect()
        K.clear_session()
    

if __name__ == "__main__":
    dataSet = sys.argv[1]
    labels = sys.argv[2]
    checkpoint = sys.argv[3]
    output = sys.argv[4]
    
    test(dataSet, labels, checkpoint, output)