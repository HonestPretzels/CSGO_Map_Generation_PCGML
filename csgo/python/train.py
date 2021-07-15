import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import sys

def load_data(dataPath, targetPath):
    data = np.load(dataPath)
    target = np.load(targetPath)
    
    return train_test_split(data, target, test_size=0.1, random_state=1337)

def main():
    data = sys.argv[1]
    target = sys.argv[2]

    xTrain, xTest, yTrain, yTest = load_data(data, target)

if __name__ == "__main__":
    main()