import numpy as np
import os.path as path
import os
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical  

def getAllFilePaths(p):
    return [path.join(p,f) for f in os.listdir(p) if path.isfile(path.join(p, f))]

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

xfiles = getAllFilePaths('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\ReducedImagesNoZeros\\Labels\\')
x = np.load(xfiles[1])
print(x.shape)
# x = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\Latent_encoding\\latent_encoding_3.npy', allow_pickle=True)
# print(x.shape)

# d = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\PreProcessed\\data\\data_11.npy')
# print(d.shape)

# y = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\PreProcessed\\scores\\scores_11.npy')
# print(y.shape)
# y = np.repeat(y, 30)
# print(y.shape)
# y = y.reshape(y.shape[0], 1)
# print(y.shape)
# y = to_categorical(y, 2)
# print(y.shape)
