import numpy as np
import sys, os
from tqdm import tqdm

SPLIT_LENGTH = 30

def ProcessOneDemo(seqPath, roundsPath):
    rounds = np.load(roundsPath)
    seq = np.load(seqPath)
    out = []
    outScores = []

    # First round is knife, last round is padding
    for i in range(len(rounds[1:-1])):
        round_start_second = rounds[i-1][0]
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
        outScores.extend([rounds[i][1]]*len(splits))

    out = np.array(out)
    outScores = np.array(outScores)
    return out, outScores
    
        

def main():
    logPath = sys.argv[1]
    round_files = [f for f in os.listdir(os.path.join(logPath, "fullGames\\rounds")) if os.path.isfile(os.path.join(logPath, "fullGames\\rounds\\" + f))]
    fullData = []
    fullScores = []
    for f in tqdm(round_files):
        sf = os.path.join(os.path.join(logPath, "fullGames\\sequences"), os.path.basename(f).split('_')[0] + '.npy')
        rf = os.path.join(os.path.join(logPath, "fullGames\\rounds"), f)
        data, scores = ProcessOneDemo(sf, rf)
        fullData.extend(data)
        fullScores.extend(scores)
        

    outputPath = os.path.join(logPath, "trainingSequences")
    np.save(os.path.join(os.path.join(outputPath,"data"),"train_data.npy"), np.array(fullData))
    np.save(os.path.join(os.path.join(outputPath,"scores"), "train_scores.npy"), np.array(fullScores))

if __name__ == "__main__":
    main()