import numpy as np
import sys, os


def ProcessOneDemo(seq, rounds):
    rounds = np.load(rounds)
    seq = np.load(seq)
    print(seq.shape)

def main():
    logPath = sys.argv[1]
    round_files = [f for f in os.listdir(os.path.join(logPath, "rounds")) if os.path.isfile(os.path.join(os.path.join(logPath, "rounds"), f))]
    for f in round_files:
        sf = os.path.join(os.path.join(logPath, "sequences"), os.path.basename(f).split('_')[0] + '.npy')
        rf = os.path.join(os.path.join(logPath, "rounds"), f)
        ProcessOneDemo(sf, rf)
            

if __name__ == "__main__":
    main()