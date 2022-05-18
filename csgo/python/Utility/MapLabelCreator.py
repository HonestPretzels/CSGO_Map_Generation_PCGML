import numpy as np
import sys, os
import os.path as path
from matplotlib import pyplot as plt
from PIL import Image

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def main():
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]
    
    maps = []
    labels = []
    
    files = getAllFiles(inputPath)
    for f in files:
        x = np.load(path.join(inputPath,f))
        print(x.shape)
        for split in x:
            map = split[0][0]
            # Append Maps
            exists = False
            for i in range(len(maps)):
                map2 = maps[i]
                if (map == map2).all():
                    exists = True
            if not exists:
                maps.append(map)
                      
            for i in range(len(maps)):
                map2 = maps[i]
                if (map == map2).all():
                    label = i
            labels.append(label)
    labels = np.array(labels)
    for map in maps:
        r_img = Image.fromarray((map).astype(np.uint8))
        plt.imshow(r_img)
        plt.show()
    print(labels.shape)
    np.save(path.join(outputPath, "mapLabels.npy"), labels)

if __name__ == "__main__":
    main()