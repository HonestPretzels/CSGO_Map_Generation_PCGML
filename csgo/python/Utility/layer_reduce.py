import numpy as np
import sys
import os
import os.path as path
from matplotlib import pyplot as plt
from PIL import Image
import gc


def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    for f in getAllFiles(inputPath):
        ifp = path.join(inputPath, f)
        ofp = path.join(outputPath, f)
        # Get only player layers
        x = np.load(ifp)[:,:,5:15]
        # Separate teams
        x = np.reshape(x, (x.shape[0], x.shape[1], 2, 5, x.shape[3], x.shape[4]))
        # Sum the layers
        y = np.sum(x, axis=3)
        # Clamp to 0 or 1
        z = np.where(y > 0.0001, 1, 0)
        
        ## Visualization Code
        # for second in range(z.shape[1]):
        #     for team in range(2):
        #         r = z[0][second][team] * 255
        #         g = np.zeros((128,128))
        #         b = np.zeros((128,128))
        #         img = np.dstack((r,g,b))
        #         r_img = Image.fromarray((img).astype(np.uint8))
        #         plt.imshow(r_img)
        #         plt.show()
        
        np.save(ofp, z)
        gc.collect()
    

if __name__ == "__main__":
    main()