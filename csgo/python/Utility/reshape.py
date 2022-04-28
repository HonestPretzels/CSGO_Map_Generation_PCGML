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
        x = x.reshape((x.shape[0]//30, 30,-1))
        np.save(p, x)
        
if __name__ == "__main__":
    main()