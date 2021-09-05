import numpy as np
import sys, os
import json
from tqdm import tqdm

SPLIT_LENGTH = 30

MAP_MAP = {
    "de_dust2": 0,
    "de_mirage": 1,
    "de_vertigo": 2,
    "de_inferno": 3,
    "de_overpass": 4,
    "de_train": 5,
}

def getMap(id, metaDataPath):
    with open(os.path.join(metaDataPath, id + ".json")) as f:
        metaData = json.load(f)
        return MAP_MAP[metaData["voting"]["map"]["pick"][0]]


def ProcessOneDemo(seqPath, roundsPath):
    rounds = np.load(roundsPath)
    seq = np.load(seqPath)
    out = []
    outScores = []
    splitsPerRound = []

    # First round is knife, last round is padding
    for i in range(1, len(rounds)):
        round_start_second = rounds[i-1][0] + 1
        round_end_second = rounds[i][0]
        round = seq[round_start_second:round_end_second+1]
        # Needs to be padded
        if len(round) % SPLIT_LENGTH != 0:
            splits = np.zeros(((round.shape[0] // SPLIT_LENGTH) + 1, SPLIT_LENGTH, 10, 23))
            j = 0
            for j in range(0, len(round) - (len(round) % SPLIT_LENGTH), SPLIT_LENGTH):
                splits[i//SPLIT_LENGTH] = round[j:j+SPLIT_LENGTH]
            for k in range(len(round[j+SPLIT_LENGTH:])):
                splits[j//SPLIT_LENGTH+1][k] = round[j+k]
        # Does not need to be padded
        else:
            splits = np.zeros(((round.shape[0] // SPLIT_LENGTH), SPLIT_LENGTH, 10, 23))
            for j in range(0, len(round), SPLIT_LENGTH):
                splits[j//SPLIT_LENGTH] = round[j:j+SPLIT_LENGTH]

        # Save splits
        out.extend(splits)
        splitsPerRound.append(len(splits))
        outScores.extend([rounds[i][1]]*len(splits))

    out = np.array(out)
    outScores = np.array(outScores)
    return out, outScores, splitsPerRound
    
        

def main():
    roundsPath = sys.argv[1]
    sequencesPath = sys.argv[2]
    metaDataPath = sys.argv[3]
    outputPath = sys.argv[4]
    round_files = [f for f in os.listdir(roundsPath) if os.path.isfile(os.path.join(roundsPath, f))]
    i = 0
    for f in round_files:
        i += 1
        print('Game %d: %s'%(i, f))
    fullData = []
    fullScores = []
    fullMaps = []
    roundsPerGame = []
    for f in tqdm(round_files):
        sf = os.path.join(os.path.join(sequencesPath), os.path.basename(f).split('_')[0] + '.npy')
        rf = os.path.join(os.path.join(roundsPath), f)
        data, scores, splitsPerRound = ProcessOneDemo(sf, rf)
        fullData.extend(data)
        fullScores.extend(scores)
        roundsPerGame.append(splitsPerRound)
        fullMaps.extend([getMap(os.path.basename(rf).split('_')[0], metaDataPath)]*len(data))

    np.save(os.path.join(os.path.join(outputPath,"data"),"data.npy"), np.array(fullData))
    np.save(os.path.join(os.path.join(outputPath,"scores"), "scores.npy"), np.array(fullScores))
    np.save(os.path.join(os.path.join(outputPath,"maps"), "maps.npy"), np.array(fullMaps))
    # 2D array of the game [Number of rounds][Splits per round]
    np.save(os.path.join(os.path.join(outputPath,"breakpoints"),"breakpoints.npy"), np.array(roundsPerGame, dtype=object))

if __name__ == "__main__":
    main()