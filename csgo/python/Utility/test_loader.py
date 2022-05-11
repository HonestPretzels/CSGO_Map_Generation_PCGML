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

# xfiles = getAllFilePaths('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\Latent_encodingNoZeros\\CombinedLabelsSplitFormat\\')
# x = np.load(xfiles[0])
# wins = 0
# losses = 0
# for i in x:
#     if i[0] == 1:
#         wins += 1
#     else:
#         losses += 1
# print(wins, losses)
x = np.load('F:\\CSGO_Map_Generator\\Dev_data_subset\\FullAutoencoderDataSet\\training_data\\data_0_2.npy', allow_pickle=True)
y = np.load('F:\\CSGO_Map_Generator\\Dev_data_subset\\FullAutoencoderDataSet\\training_labels\\data_0_2.npy', allow_pickle=True)
print(x.shape, y.shape)

# Visualization Code
for second in range(x.shape[1]):
    for team in range(x.shape[2]):
        r = x[0][second][team] * 255
        g = y[0][second][team] * 255
        b = np.zeros((128,128))
        img = np.dstack((r,g,b))
        r_img = Image.fromarray((img).astype(np.uint8))
        plt.imshow(r_img)
        plt.show()
