import numpy as np
from PIL import Image
import PIL, sys, os
from os import path
from tqdm import tqdm

MAP_SCALE = 128

MAP_MAP = {
    "de_dust2": 0,
    "de_mirage": 1,
    "de_vertigo": 2,
    "de_inferno": 3,
    "de_overpass": 4,
    "de_train": 5,
}

REVERSE_MAP_MAP = {
    0: "de_dust2",
    1: "de_mirage",
    2: "de_vertigo",
    3: "de_inferno",
    4: "de_overpass",
    5: "de_train",
}

PLAYER_TYPES = {
    "Utility": 0,
    "Shooting": 1,
    "Jumping": 2,
    "Crouching": 3,
    "Standard": 4,
}

PLAYER_OPACITY = 0.2 

def getAllFilePaths(p):
    return [path.join(p,f) for f in os.listdir(p) if path.isfile(path.join(p, f))]

def usedUtility(p):
    if p[16] == 1.0:
        return True
    if p[17] == 1.0:
        return True
    if p[18] == 1.0:
        return True
    if p[19] == 1.0:
        return True
    if p[20] == 1.0:
        return True
    return False

def processArray(seq, maps, images, outPath):

    # 30 Seconds, 
    # 15 depth (L1, L2, L1Obj, L2Obj, Bombs, T1_Standard, T1_Crouching, T1_Jumping, T1_Shooting, T1_Utility, T2_Standard, T2_Crouching, T2_Jumping, T2_Shooting, T2_Utility),
    # Mapsize, 
    # Mapsize
    split_outPut = np.zeros((len(seq),30,15, MAP_SCALE, MAP_SCALE), dtype=np.float32)
    for split in tqdm(range(len(seq))):
        split_data = seq[split]
        split_map = REVERSE_MAP_MAP[maps[split]]
        L1 = "%s_main_floor.png"%split_map
        L2 = "%s_floor_2.png"%split_map
        L1Obj = "%s_main_objects.png"%split_map
        L2Obj = "%s_objects_2.png"%split_map
        bombs = "%s_bomb_sites.png"%split_map
        for imgPath in images:
            if L1 in imgPath:
                L1_arr = np.asarray(Image.open(imgPath).convert('L'))
                L1_arr = L1_arr * (1/255)
            elif L2 in imgPath:
                L2_arr = np.asarray(Image.open(imgPath).convert('L'))
                L2_arr = L2_arr * (1/255)
            elif L1Obj in imgPath:
                L1Obj_arr = np.asarray(Image.open(imgPath).convert('L'))
                L1Obj_arr = L1Obj_arr * (1/255)
            elif L2Obj in imgPath:
                L2Obj_arr = np.asarray(Image.open(imgPath).convert('L'))
                L2Obj_arr = L2Obj_arr * (1/255)
            elif bombs in imgPath:
                bombs_arr = np.asarray(Image.open(imgPath).convert('L'))
                bombs_arr = bombs_arr * (1/255)

        for second in range(len(split_data)):
            split_outPut[split][second][0] = L1_arr
            split_outPut[split][second][1] = L1Obj_arr
            split_outPut[split][second][2] = L2_arr
            split_outPut[split][second][3] = L2Obj_arr
            split_outPut[split][second][4] = bombs_arr

            for player in range(len(split_data[second])):
                if player < 5:
                    teamOffset = 5 # Team 1
                else:
                    teamOffset = 10 # Team 2

                player_data = split_data[second][player]
                if player_data[8] <= 0: # Player is dead
                    continue
                else:
                    posX = round(player_data[0])
                    posY = round(player_data[1])
                    # Bump players 1 pixel away from the edge of map to allow for full representations
                    posX = 126 if posX >= 127 else posX
                    posY = 126 if posY >= 127 else posY
                    posX = 1 if posX <= 0 else posX
                    posY = 1 if posY <= 0 else posY
                if usedUtility(player_data): # Utility
                    split_outPut[split][second][teamOffset + 4][posY][posX] += PLAYER_OPACITY
                elif player_data[15] == 1.0: # Shooting
                    split_outPut[split][second][teamOffset + 3][posY][posX] += PLAYER_OPACITY
                elif player_data[14] == 1.0: # Jumped
                    split_outPut[split][second][teamOffset + 2][posY][posX] += PLAYER_OPACITY
                elif player_data[13] == 1.0: # Crouched
                    split_outPut[split][second][teamOffset + 1][posY][posX] += PLAYER_OPACITY
                else: # Standard
                    split_outPut[split][second][teamOffset][posY][posX] += PLAYER_OPACITY

    np.save(outPath, split_outPut)


def main():
    splitFolder = sys.argv[1]
    mapsFolder = sys.argv[2]
    imageFolder = sys.argv[3]
    outFolder = sys.argv[4]


    # TODO: Iterate through files
    SplitFiles = getAllFilePaths(splitFolder)
    MapFiles = getAllFilePaths(mapsFolder)
    images = getAllFilePaths(imageFolder)
    for idx in tqdm(range(len(SplitFiles))):
        sequenceArr = np.load(SplitFiles[idx])
        mapsArr = np.load(MapFiles[idx])

        processArray(sequenceArr, mapsArr, images, path.join(outFolder, "data_%d.npy"%idx))

if __name__ == "__main__":
    main()
