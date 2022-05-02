import numpy as np
import sys, os
import os.path as path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    out = []
    for file in getAllFiles(inputPath):
        out.append(np.load(path.join(inputPath, file)))
    print([x.shape for x in out])
    out = np.concatenate(out, axis=0)
    print(out.shape)
    np.save(outputPath, out)
    
    

if __name__ == "__main__":
    main()