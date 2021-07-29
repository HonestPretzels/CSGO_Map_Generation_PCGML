import numpy as np
import tflearn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tflearn import conv_2d, input_data
import sys, os
from tflearn.layers.conv import max_pool_2d
import tensorflow as tf

from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.models.dnn import DNN
from demo_parse_loader import MAP_MAP

SECONDS = 30
PLAYERS = 10
STATE_LENGTH = 23
MAP_DEPTH = 5
MAP_SCALE = 128


def load_data(dataPath, mapPath, targetPath, mapMapping):
    '''
    Load the 3 data vectors and split them out into train-test splits
    '''
    print('loading data')
    data = np.load(dataPath)
    print('loading maps')
    maps = np.load(mapPath)
    finishedMaps = np.empty((len(maps), MAP_DEPTH, MAP_SCALE, MAP_SCALE))
    for i in range(len(maps)):
        finishedMaps[i] = mapMapping[maps[i]]
    print('loading scores')
    scores = np.load(targetPath)
    target = np.empty((len(scores), 2))
    for i in range(len(scores)):
        target[i] = [1,0] if scores[i] == 1 else [0,1]
    
   

    print('splitting')
    return train_test_split(data, finishedMaps, target, test_size=0.1, random_state=1337)

def generate_map_mapping(mapPath):
    '''
    Load the images needed for the maps and generate the mapping of map_ids to those images
    The 5 dimensions represent:
        0. main floor
        1. main objects
        2. second floor
        3. second objects
        4. bomb sites
    '''
    outMap = {}
    for mapName in MAP_MAP.keys():
        outMap[MAP_MAP[mapName]] = np.empty((MAP_DEPTH,MAP_SCALE,MAP_SCALE))
        img = Image.open(os.path.join(mapPath, mapName+"_main_floor.png"))
        outMap[MAP_MAP[mapName]][0] = processImage(img)
        img = Image.open(os.path.join(mapPath, mapName+"_main_objects.png"))
        outMap[MAP_MAP[mapName]][1] = processImage(img)
        img = Image.open(os.path.join(mapPath, mapName+"_floor_2.png"))
        outMap[MAP_MAP[mapName]][2] = processImage(img)
        img = Image.open(os.path.join(mapPath, mapName+"_objects_2.png"))
        outMap[MAP_MAP[mapName]][3] = processImage(img)
        img = Image.open(os.path.join(mapPath, mapName+"_bomb_sites.png"))
        outMap[MAP_MAP[mapName]][4] = processImage(img)
    return outMap

def processImage(image):
    '''
    Returns a 2d numpy array from an image
    '''
    return np.asarray_chkfinite(ImageOps.grayscale(image))

def main():
    data = sys.argv[1]
    maps = sys.argv[2]
    target = sys.argv[3]
    imagePath = sys.argv[4]

    mapMapping = generate_map_mapping(imagePath)

    xTrain, xTest, mapTrain, mapTest, yTrain, yTest = load_data(data, maps, target, mapMapping)
    
    xTrain = np.reshape(xTrain, [-1, SECONDS*PLAYERS*STATE_LENGTH])
    mapTrain = np.reshape(mapTrain, [-1, MAP_DEPTH*MAP_SCALE*MAP_SCALE])
    fullTrain = np.concatenate((xTrain, mapTrain), axis=1)

    xValidate = np.reshape(xTest, [-1, SECONDS*PLAYERS*STATE_LENGTH])
    mapValidate = np.reshape(mapTest, [-1, MAP_DEPTH*MAP_SCALE*MAP_SCALE])
    fullValidate = np.concatenate((xValidate, mapValidate), axis=1)

    
    model = initNetwork(fullTrain.shape)
    model.fit(fullTrain, yTrain, validation_set=(fullValidate, yTest), show_metric=True, batch_size=32)

def initNetwork(inputShape):

    # SPLIT
    netInput = input_data([None, inputShape[1]])

    dataFlat = tf.slice(netInput, [0,0], [-1, SECONDS*PLAYERS*STATE_LENGTH])
    mapFlat = tf.slice(netInput, [0, SECONDS*PLAYERS*STATE_LENGTH], [-1, -1])
    
    dataInput = tf.reshape(dataFlat, [-1, SECONDS, PLAYERS*STATE_LENGTH])
    mapInput = tf.reshape(mapFlat, [-1, MAP_DEPTH, MAP_SCALE, MAP_SCALE])

    # MAP HEAD
    conv1 = conv_2d(mapInput,16, 3, activation="relu")
    dropped = dropout(conv1, 0.5)
    pooled1 = max_pool_2d(dropped, 2, strides=2)
    conv2 = conv_2d(pooled1, 32, 3, activation="relu")
    pooled2 = max_pool_2d(conv2, 2, strides=2)
    flatMap = tflearn.flatten(pooled2)

    # DATA HEAD
    lstm1 = lstm(dataInput, 128, dropout=0.8, return_seq=True)
    lstm2 = lstm(lstm1, 128)
    flatData = tflearn.flatten(lstm2)

    # RECOMBINE 
    combined = tf.concat([flatData, flatMap], 1)
    fc1 = fully_connected(combined, 128)
    fc2 = fully_connected(fc1, 64)
    fc3 = fully_connected(fc2, 2, activation="softmax")
    out = regression(fc3, optimizer="adam", learning_rate=0.001, loss="categorical_crossentropy")

    model = DNN(out)
    return model

if __name__ == "__main__":
    main()