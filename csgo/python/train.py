from genericpath import isfile
import numpy as np
import tflearn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tflearn import conv_2d, input_data
import sys, os
from tflearn.layers.conv import max_pool_2d
import tensorflow as tf
import random

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

random.seed(1337)
tf.random.set_seed(1337)

def load_data(dataPath, mapPath, targetPath, mapMapping, doSplit=True):
    '''
    Load the 3 data vectors and split them out into train-test splits
    '''
    data = np.load(dataPath)
    maps = np.load(mapPath)
    finishedMaps = np.empty((len(maps), MAP_DEPTH, MAP_SCALE, MAP_SCALE))
    for i in range(len(maps)):
        finishedMaps[i] = mapMapping[maps[i]]
    scores = np.load(targetPath)
    target = np.empty((len(scores), 2))
    for i in range(len(scores)):
        target[i] = [1,0] if scores[i] == 1 else [0,1]
    
   

    if doSplit:
        print('splitting')
        xTrain, xTest, mapTrain, mapTest, yTrain, yTest = train_test_split(data, finishedMaps, target, test_size=0.1, random_state=1337)

        xTrain = np.reshape(xTrain, [-1, SECONDS*PLAYERS*STATE_LENGTH])
        mapTrain = np.reshape(mapTrain, [-1, MAP_DEPTH*MAP_SCALE*MAP_SCALE])
        fullTrain = np.concatenate((xTrain, mapTrain), axis=1)

        xValidate = np.reshape(xTest, [-1, SECONDS*PLAYERS*STATE_LENGTH])
        mapValidate = np.reshape(mapTest, [-1, MAP_DEPTH*MAP_SCALE*MAP_SCALE])
        fullValidate = np.concatenate((xValidate, mapValidate), axis=1)

        return (fullTrain, yTrain), (fullValidate, yTest)
    
    else:
        x = np.reshape(data, [-1, SECONDS*PLAYERS*STATE_LENGTH])
        maps= np.reshape(finishedMaps, [-1, MAP_DEPTH*MAP_SCALE*MAP_SCALE])
        full = np.concatenate((x, maps), axis=1)
        return (full, target)


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

def getSets(dPath, mPath, tPath, gPath = None):
    '''
    Returns the paths to the data, maps, and targets for each file in the directories given
    as a simple dictionary
    '''
    data = {i: path for i , path in enumerate([os.path.join(dPath, f) for f in os.listdir(dPath)])}
    maps = {i: path for i , path in enumerate([os.path.join(mPath, f) for f in os.listdir(mPath)])}
    targets = {i: path for i , path in enumerate([os.path.join(tPath, f) for f in os.listdir(tPath)])}
    if gPath != None:
        gameBreaks = {i: path for i , path in enumerate([os.path.join(gPath, f) for f in os.listdir(gPath)])}
        return data, maps, targets, gameBreaks
    else:
        return data, maps, targets

def initNetwork(inputShape):
    '''
    Create the network architecture
    '''

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

def train():
    '''
    Main training loop
    '''
    dataPath = sys.argv[1]
    mapsPath = sys.argv[2]
    targetPath = sys.argv[3]
    imagePath = sys.argv[4]
    checkPointPath = sys.argv[5]

    mapMapping = generate_map_mapping(imagePath)
    data, maps, targets = getSets(dataPath, mapsPath, targetPath)

    # SET ONE
    (xTrain, yTrain) , (xValidate, yValidate) = load_data(data[0], maps[0], targets[0], mapMapping)
    model = initNetwork(xTrain.shape)
    model.fit(xTrain, yTrain, validation_set=(xValidate, yValidate), show_metric=True, batch_size=32)
    model.save(checkPointPath)

    for i in range(len(data))[1:]:
        del xTrain
        del yTrain
        del xValidate
        del yValidate
        (xTrain, yTrain) , (xValidate, yValidate) = load_data(data[i], maps[i], targets[i], mapMapping)
        model.load(checkPointPath)
        model.fit(xTrain, yTrain, validation_set=(xValidate, yValidate), show_metric=True, batch_size=32)
        model.save(checkPointPath)
    model.save(os.path.join(os.path.split(checkPointPath)[0],"final.model"))

if __name__ == "__main__":
    train()