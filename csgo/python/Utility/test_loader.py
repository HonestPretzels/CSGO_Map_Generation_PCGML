import numpy as np
import os.path as path
import os
from PIL import Image
from matplotlib import pyplot as plt

x = np.load('F:/CSGO_Map_Generator/Dev_data_subset/MapAndPlayersTest/data_103_1.npy')
y = np.load('F:/CSGO_Map_Generator/Dev_data_subset/MapAndPlayersTest_labels/data_103_1.npy')
x = x.reshape(x.shape[0]*30, 3, 128, 128)
y = y.reshape(y.shape[0]*30, 2, 128, 128)
for i in range(1000,x.shape[0]):
    for j in range(y.shape[1]):
        r = np.zeros((128,128))
        g = y[i][j]*255
        b = x[i][j]*255
    
        img = np.dstack((r,g,b))
        r_img = Image.fromarray((img).astype(np.uint8))
        plt.imshow(r_img)
        plt.show()