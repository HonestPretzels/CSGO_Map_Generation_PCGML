import numpy as np
import sys, os
from os import path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    files = getAllFiles(sys.argv[1])
    print(files)
    for f in files:
        p = path.join(sys.argv[1], f)
        x = np.load(p)
        players = np.take(x, [1,2], axis=2)
        map = np.take(x, [0], axis=2) * (1/255)
        print(players.shape, map.shape, np.max(map), np.max(players))
        x = np.concatenate((map, players), axis=2)
        print(x.shape, np.max(x))
        # x = np.delete(x, np.arange(1,30), axis=1)
        # x = x.reshape((x.shape[0],-1))
        np.save(path.join(sys.argv[2], f), x)
        
if __name__ == "__main__":
    main()