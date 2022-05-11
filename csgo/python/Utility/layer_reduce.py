import numpy as np
import sys
import os
import os.path as path
import gc
import keras.backend as K


def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main(isLabel = False):
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    for f in getAllFiles(inputPath):
        ifp = path.join(inputPath, f)
        ofp = path.join(outputPath, f)
        # Get only player layers
        x = np.load(ifp)
        players = x[:,:,5:15]
        level = x[:,:,0]
        # Separate teams
        players = np.reshape(players, (players.shape[0], players.shape[1], 2, 5, players.shape[3], players.shape[4]))
        # Sum the layers
        players = np.sum(players, axis=3)
        # Clamp to 0 or 1
        players = np.where(players > 0.0001, 1, 0)
        if isLabel:
            players = np.reshape(players, (players.shape[0], players.shape[1], 2, 128*128))
            players = np.sum(players, axis=3)/5
            
            fullPlayers = np.zeros((level.shape[0], level.shape[1], 2, level.shape[2], level.shape[3]))
            for split in range(len(players)):
                for second in range(len(players[split])):
                    for team in range(len(players[split,second])):
                        fullPlayers[split,second,team].fill(players[split,second,team])
                        
            level = np.reshape(level, (level.shape[0], level.shape[1], 1, level.shape[2], level.shape[3]))
            output = np.concatenate((level, fullPlayers), axis=2)
            
            print(level.shape, players.shape, fullPlayers.shape, output.shape)
        else:
            level = np.reshape(level, (level.shape[0], level.shape[1], 1, 128, 128))
            output = np.concatenate((level, players), axis=2)
            print(level.shape, players.shape, output.shape)

        
        
        np.save(ofp, output)
        gc.collect()
        K.clear_session()
    

if __name__ == "__main__":
    main(True)