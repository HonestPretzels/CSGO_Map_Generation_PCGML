import numpy as np
import sys
import os
import os.path as path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    
    for f in getAllFiles(inputDir):
        ifp = path.join(inputDir, f)
        ofp = path.join(outputDir, f)
        
        x = np.load(ifp)
        y = np.reshape(x, (x.shape[0], x.shape[1], 2*128*128))
        z = np.sum(y, axis=2)
        z = z/10
        np.save(ofp, z)

if __name__ == "__main__":
    main()
    