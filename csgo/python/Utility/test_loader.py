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
x = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\Latent_encodingNoZeros_2Teams\\CombinedTeamLabels_SplitFormat\\bothCountLabels.npy', allow_pickle=True)
print(x[0:10])

# y = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\ReducedImages\\SeparateTeamLabels\\data_1.npy', allow_pickle=True)

# print(x.shape, y.shape)

# # Visualization Code
# for split in range(x.shape[0]):
#     for second in range(x.shape[1]):
#         for team in range(x.shape[2]):
#             r = x[split][second][team]
#             g = y[split][second][team]
#             print(np.max(g))
#             b = np.zeros((128,128))
            
#             s = sum(g.flatten())
#             if s > 1:
#                 r *= 255
#                 g *= 255
#                 img = np.dstack((r,g,b))
#                 r_img = Image.fromarray((img).astype(np.uint8))
#                 plt.imshow(r_img)
#                 plt.show()
