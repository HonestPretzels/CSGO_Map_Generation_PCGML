import numpy as np
import os.path as path
import os
from PIL import Image
from matplotlib import pyplot as plt

x = np.load('F:\\CSGO_Map_Generator\\CleanDataSetApr2022\\PreProcessed\\scores\\scores_1.npy', allow_pickle=True)
print(x.shape)
