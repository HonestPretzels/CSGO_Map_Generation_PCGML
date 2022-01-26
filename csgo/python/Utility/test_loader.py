import numpy as np
import os.path as path
import os
from PIL import Image
from matplotlib import pyplot as plt

x = np.load('F:/CSGO_Map_Generator/Dev_data_subset/Training_splits/data_0_2.npy')
x = x.reshape(x.shape[0]*30, 15, 128, 128)
print(sum(x[0,5,:,:].flatten()))
print(x[0,0,40,:])
print(x.shape)
for j in range(x.shape[1]):
    # Display the real values
    r = np.zeros((128,128))
    # Third layer is empty
    g = np.zeros((128,128))
    # Blue for preds
    b = x[0][j]
    print(sum(b.flatten()))
    
    img = np.dstack((r,g,b))
    r_img = Image.fromarray((img).astype(np.uint8))
    plt.imshow(r_img)
    plt.show()