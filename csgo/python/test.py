import sys
import numpy as np
from matplotlib import pyplot as plt
from train import initNetwork, generate_map_mapping, load_data, getSets

# gameBreaks refers to the delinieators between the games and is an array of the number of splits in each game

def evaluatePred(pred, actual, gameBreaks):
    correct = 0
    total = 0

    i = 0
    g = 0
    for game in gameBreaks:
        g += 1
        game_split = 0
        fig = plt.figure(figsize=(20,4))
        game_preds_x = []
        game_preds_y = []
        game_actuals_x = []
        game_actuals_y = []
        round_delineators = []
        for j in range(len(game)):
            r = game[j]
            round_delineators.append(sum(game[:j+1]))
            for split in range(r):
                i += 1
                game_split += 1
                winnerPred = np.argmax(pred[i])
                winnerActual = np.argmax(actual[i])
                total += 1
                if winnerPred == winnerActual:
                    correct += 1
                game_preds_y.append(winnerPred)
                game_actuals_y.append(winnerActual)
                game_preds_x.append(game_split)
                game_actuals_x.append(game_split)
        for x in round_delineators:
            plt.axvline(x, color="black")
        plt.plot(game_preds_x, game_preds_y, marker='o', label="predicted")
        plt.plot(game_actuals_x, game_actuals_y, marker='*', linestyle="--", label="actual")
        plt.legend(loc="upper right")
        plt.ylim(-0.1, 1.5)
        plt.savefig('./testPlots/test_game_%d.png'%g)

    print('Accuracy on test set is %.3f'%(correct/total))
    # TODO: GRAPH THE PREDICTION VS THE ACTUAL

def test():
    testDataPath = sys.argv[1]
    testMapsPath = sys.argv[2]
    testTargetsPath = sys.argv[3]
    gameBreaksPath = sys.argv[4]
    imagePath = sys.argv[5]
    modelPath = sys.argv[6]

    mapMapping = generate_map_mapping(imagePath)
    data, maps, targets = getSets(testDataPath, testMapsPath, testTargetsPath)
    (x, y) = load_data(data[0], maps[0], targets[0], mapMapping, doSplit=False)

    model = initNetwork(x.shape)
    model.load(modelPath, weights_only=True)
    pred = model.predict(x)

    gameBreaks = np.load(gameBreaksPath, allow_pickle=True)
    evaluatePred(pred, y, gameBreaks)

if __name__ == "__main__":
    test()