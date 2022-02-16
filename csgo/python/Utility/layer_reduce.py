import numpy as np
import sys
import os
import os.path as path
from matplotlib import pyplot as plt
from PIL import Image
import gc
import keras.backend as K


def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    for f in getAllFiles(inputPath):
        ifp = path.join(inputPath, f)
        ofp = path.join(outputPath, f)
        # Get only player layers
        x = np.load(ifp)
        players = x[:,:,5:15]
        level = x[:,:,0] / 255
        # Separate teams
        players = np.reshape(players, (players.shape[0], players.shape[1], 2, 5, players.shape[3], players.shape[4]))
        # Sum the layers
        players = np.sum(players, axis=3)
        # Clamp to 0 or 1
        players = np.where(players > 0.0001, 1, 0)
        players = np.reshape(players, (players.shape[0], players.shape[1], 2*128*128))
        players = np.sum(players, axis=2)/10
        
        fullPlayers = np.zeros((level.shape))
        for split in range(len(players)):
            for second in range(len(players[split])):
                fullPlayers[split,second].fill(players[split,second])
        
        output = np.stack((level, fullPlayers), axis=2)
        print(output.shape)

        
        # ## Visualization Code
        # for second in range(output.shape[1]):
        #     for team in range(2):
        #         r = output[0][second][team] * 255
        #         g = np.zeros((128,128))
        #         b = np.zeros((128,128))
        #         img = np.dstack((r,g,b))
        #         r_img = Image.fromarray((img).astype(np.uint8))
        #         plt.imshow(r_img)
        #         plt.show()
        
        np.save(ofp, output)
        gc.collect()
        K.clear_session()
    

if __name__ == "__main__":
    main()