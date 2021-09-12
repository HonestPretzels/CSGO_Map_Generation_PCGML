import sys
import numpy as np
from matplotlib import pyplot as plt
from train import initNetwork, generate_map_mapping, load_data, getSets

# gameBreaks refers to the delinieators between the games and is an array of the number of splits in each game

def evaluatePred(pred, actual, gameBreaks):
    '''
    Graph the predictions vs the actual for each game
    '''
    correct = 0
    total = 0
    wins = 0

    i = 0
    g = 0
    for game in gameBreaks:
        g += 1
        game_split = 0
        # fig = plt.figure(figsize=(20,4))
        game_preds_x = []
        game_preds_y = []
        game_actuals_x = []
        game_actuals_y = []
        round_delineators = []
        for j in range(len(game)):
            r = game[j]
            round_delineators.append(sum(game[:j+1]))
            for split in range(r):
                game_split += 1
                winnerPred = pred[i][0]
                winnerActual = actual[i][0]
                i += 1
                total += 1
                if round(winnerPred) == round(winnerActual):
                    correct += 1
                if winnerActual == 1:
                    wins += 1
                game_preds_y.append(winnerPred)
                game_actuals_y.append(winnerActual)
                game_preds_x.append(game_split)
                game_actuals_x.append(game_split)
        # for x in round_delineators:
        #     plt.axvline(x, color="black")
        # plt.plot(game_preds_x, game_preds_y, marker='o', label="predicted")
        # plt.plot(game_actuals_x, game_actuals_y, marker='*', linestyle="--", label="actual")
        # plt.legend(loc="upper right")
        # plt.ylim(-0.1, 1.5)
        # plt.savefig('./testPlots/test_game_%d.png'%g)

    return correct, wins, total

def test():
    '''
    Main test loop
    '''
    testDataPath = sys.argv[1]
    testMapsPath = sys.argv[2]
    testTargetsPath = sys.argv[3]
    gameBreaksPath = sys.argv[4]
    imagePath = sys.argv[5]
    modelPath = sys.argv[6]

    mapMapping = generate_map_mapping(imagePath)
    data, maps, targets, gameBreaks = getSets(testDataPath, testMapsPath, testTargetsPath, gameBreaksPath)
    (x, y) = load_data(data[0], maps[0], targets[0], mapMapping, doSplit=False)

    model = initNetwork(x.shape)
    model.load(modelPath, weights_only=True)
    

    correctCount = 0
    totalCount = 0
    winsCount = 0
    for i in range(len(data)):
        (x, y) = load_data(data[i], maps[i], targets[i], mapMapping, doSplit=False)
        pred = model.predict(x)
        g = np.load(gameBreaks[i], allow_pickle=True)
        count = 0
        for game in g:
            for r in game:
                count += r
        c, w, t = evaluatePred(pred, y, g)
        correctCount += c
        winsCount += w
        totalCount += t

    print("Accuracy: %.3f, Wins: %.3f"%(correctCount/totalCount, winsCount/totalCount))

if __name__ == "__main__":
    test()