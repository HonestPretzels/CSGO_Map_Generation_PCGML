import numpy as np
import tflearn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import sys, os
from demo_parse_loader import MAP_MAP

def load_data(dataPath, mapPath, targetPath):
    '''
    Load the 3 data vectors and split them out into train-test splits
    '''
    print('loading data')
    data = np.load(dataPath)
    print('loading maps')
    maps = np.load(mapPath)
    print('loading scores')
    target = np.load(targetPath)
   

    print('splitting')
    return train_test_split(data, maps, target, test_size=0.1, random_state=1337)

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
        outMap[MAP_MAP[mapName]] = np.empty((5,128,128))
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

    # xTrain, xTest, mapTrain, mapTest, yTrain, yTest = load_data(data, maps, target)

if __name__ == "__main__":
    main()