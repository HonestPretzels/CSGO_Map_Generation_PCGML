import numpy as np
import sys

data = sys.argv[1]
maps = sys.argv[2]
target = sys.argv[3]

dataOut = sys.argv[4]
mapsOut = sys.argv[5]
targetOut = sys.argv[6]

print('loading data')
data = np.load(data)
np.save(dataOut, data[:200])
print('loading maps')
maps = np.load(maps)
np.save(mapsOut, maps[:200])
print('loading scores')
target = np.load(target)
np.save(targetOut, target[:200])
