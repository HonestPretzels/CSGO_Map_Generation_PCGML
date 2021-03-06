import sys
import numpy as np
import gc
import keras.backend as K
from matplotlib import pyplot as plt
from PIL import Image
import os.path as path
import tensorflow as tf
from train_autoencoder import generate_batches, getAllFiles, genModel

# gameBreaks refers to the delinieators between the games and is an array of the number of splits in each game

def visualize(preds, reals):
    avgdiffs1 = 0
    count1 = 0
    avgdiffs2 = 0
    count2 = 0
    for y in range(preds.shape[0]): # Sub sample of states
        for t in range(preds.shape[1]):
            for x in range(preds.shape[2]):
                # Display the real values
                r = reals[y][t][x]
                # Third layer is empty
                g = np.zeros((128,128))
                # Blue for preds
                b = preds[y][t][x]
                
                if x == 1 :
                    count1 += 1
                    s = sum(r.flatten())/(128*128)
                    u = sum(b.flatten())/(128*128)
                    avgdiffs1 += (s-u)
                elif x == 2 :
                    count2 += 1
                    s = sum(r.flatten())/(128*128)
                    u = sum(b.flatten())/(128*128)
                    avgdiffs2 += (s-u)
                    
                    # if s > 0.1:
                    #     r *= 255
                    #     b *= 255
                    #     img = np.dstack((r,g,b))
                    #     r_img = Image.fromarray((img).astype(np.uint8))
                    #     plt.imshow(r_img)
                    #     plt.show()
    print('avg1: %.3f avg2: %.3f'%(avgdiffs1/count1, avgdiffs2/count2))  
    
def test(datasetPath, labelsPath, checkpointPath):
    testDataFiles = [path.join(datasetPath, f) for f in getAllFiles(datasetPath)]
    testDataLabels = [path.join(labelsPath, f) for f in getAllFiles(labelsPath)]
    
    batchSize = 30
    
    testDataSet = tf.data.Dataset.from_generator(
        generator=lambda: generate_batches(testDataFiles, testDataLabels, batchSize),
        output_types=(np.float32, np.float32),
        output_shapes=([batchSize,3,128,128], [batchSize,3,128,128])
    )
    
    AE,_ = genModel()
    AE.load_weights(checkpointPath).expect_partial()
    AE.summary()
    preds = []
    real = []
    count = 0
    for batch in testDataSet:
        count += 1
        # Memory leak happens if I don't do this batch for loop and clear the session
        preds.append(AE.predict(batch[0], verbose=1))
        real.append(batch[1])
        gc.collect()
        K.clear_session()
        # # Memory constraint hack for now
        if count >= 200:
            break
    # avg_diff = 0
    # max_diff = 0
    # for i in range(len(preds)):
    #     for j in range(len(preds[0])):
            # diff = abs(preds[i][j] - real[i][j])
            # avg_diff += diff / (len(preds)*30)
            # if diff > max_diff:
            #     max_diff = diff
                
    visualize(np.array(preds[1:]), np.array(real[1:]))

if __name__ == "__main__":
    dataSet = sys.argv[1]
    labels = sys.argv[2]
    checkpoint = sys.argv[3]
    
    test(dataSet, labels, checkpoint)