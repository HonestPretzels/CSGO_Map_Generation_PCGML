import numpy as np
from train import generate_map_mapping

MAP_DEPTH = 5
MAP_SCALE = 128

mapMapping = generate_map_mapping(".\\maps\\Resized128")
maps = np.load('.\\vectors\\trainingSequences\\fullSequences\\train_data.npy', allow_pickle=True)

# finishedMaps = np.empty((len(maps), MAP_DEPTH, MAP_SCALE, MAP_SCALE))
# for i in range(len(maps)):
#     finishedMaps[i] = mapMapping[maps[i]]
