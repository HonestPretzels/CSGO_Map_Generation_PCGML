import os
import shutil
import sys
from os import path

def getAllFiles(p):
    return [f for f in os.listdir(p) if path.isfile(path.join(p, f))]

def cleanDemos(p):
    demos = getAllFiles(p)
    zipPath = path.join(p, "batch_demos")
    os.mkdir(zipPath)
    for f in demos:
        if path.isfile(path.join(p, f)) and (path.basename(path.join(p, f)).split('.')[-1] == "dem"):
                shutil.move(path.join(p, f), path.join(zipPath, f))
    shutil.rmtree(zipPath)


def main():
    doScrape = False
    if "-scrape" in sys.argv:
        doScrape = True

    offset = int(sys.argv[2])

    # Create the folder structure
    rootFolder = sys.argv[1]
    demoPath = path.join(rootFolder, "demos")
    testDemoPath = path.join(demoPath, "testDemos")
    trainDemoPath = path.join(demoPath, "trainDemos")
    metaDataPath = path.join(rootFolder, "metaData")
    testMetaDataPath = path.join(metaDataPath, "testMetaData")
    trainMetaDataPath = path.join(metaDataPath, "trainMetaData")
    vectorPath = path.join(rootFolder, "vectors")
    testVectorPath = path.join(vectorPath, "testVectors")
    testFullGameVectorPath = path.join(testVectorPath, "fullGames")
    testFullGameRoundVectorPath = path.join(testFullGameVectorPath, "rounds")
    testFullGameSequenceVectorPath = path.join(testFullGameVectorPath, "sequences")
    trainVectorPath = path.join(vectorPath, "trainVectors")
    trainFullGameVectorPath = path.join(trainVectorPath, "fullGames")
    trainFullGameRoundVectorPath = path.join(trainFullGameVectorPath, "rounds")
    trainFullGameSequenceVectorPath = path.join(trainFullGameVectorPath, "sequences")
    testSplitPath = path.join(testVectorPath, "splits")
    testSplitDataPath = path.join(testSplitPath, "data")
    testSplitMapsPath = path.join(testSplitPath, "maps")
    testSplitScoresPath = path.join(testSplitPath, "scores")
    testSplitBreakPointsPath = path.join(testSplitPath, "breakpoints")
    trainSplitPath = path.join(trainVectorPath, "splits")
    trainSplitDataPath = path.join(trainSplitPath, "data")
    trainSplitMapsPath = path.join(trainSplitPath, "maps")
    trainSplitScoresPath = path.join(trainSplitPath, "scores")
    trainSplitBreakPointsPath = path.join(trainSplitPath, "breakpoints")


    if os.path.exists(rootFolder):
        pass
    else:
        os.mkdir(rootFolder)
        os.mkdir(demoPath)
        os.mkdir(trainDemoPath)
        os.mkdir(testDemoPath)
        os.mkdir(metaDataPath)
        os.mkdir(testMetaDataPath)
        os.mkdir(trainMetaDataPath)
        os.mkdir(vectorPath)
        os.mkdir(testVectorPath)
        os.mkdir(trainVectorPath)
        os.mkdir(testFullGameVectorPath)
        os.mkdir(testFullGameRoundVectorPath)
        os.mkdir(testFullGameSequenceVectorPath)
        os.mkdir(trainFullGameVectorPath)
        os.mkdir(trainFullGameRoundVectorPath)
        os.mkdir(trainFullGameSequenceVectorPath)
        os.mkdir(trainSplitPath)
        os.mkdir(trainSplitDataPath)
        os.mkdir(trainSplitMapsPath)
        os.mkdir(trainSplitScoresPath)
        os.mkdir(trainSplitBreakPointsPath)
        os.mkdir(testSplitPath)
        os.mkdir(testSplitDataPath)
        os.mkdir(testSplitMapsPath)
        os.mkdir(testSplitScoresPath)
        os.mkdir(testSplitBreakPointsPath)

    # process demos in batches
    for batch in range(max(1,offset), 11):
        # fetch the demos
        if doScrape:
            os.system("python .\python\Faceit_log_scraper.py %d %d %s %s -b -s -g -d"%(batch, batch-1, demoPath, metaDataPath))

        # Split out the test set
        demos = getAllFiles(demoPath)
        for i in range(len(demos)):
            metaDataFileName = path.basename(demos[i]).split('.')[0] + ".json"
            if i % 10 == 0:
                shutil.move(path.join(demoPath, demos[i]), path.join(testDemoPath, demos[i]))
                shutil.move(path.join(metaDataPath, metaDataFileName), path.join(testMetaDataPath, metaDataFileName))
            else:
                shutil.move(path.join(demoPath, demos[i]), path.join(trainDemoPath, demos[i]))
                shutil.move(path.join(metaDataPath, metaDataFileName), path.join(trainMetaDataPath, metaDataFileName))

        # Parse the demos
        if os.getcwd() == "E:\Projects\GRAIL_PCGML_tmaurer_summer_2021\csgo":
            os.chdir('go')
        os.system("go run demoParser.go %s %s %s"%('..\\' + trainDemoPath, '..\\' + trainFullGameSequenceVectorPath, '..\\' + trainFullGameRoundVectorPath))
        os.system("go run demoParser.go %s %s %s"%('..\\' + testDemoPath, '..\\' + testFullGameSequenceVectorPath, '..\\' + testFullGameRoundVectorPath))
        os.chdir('..')

        # zip the demos and remove originals for space saving
        cleanDemos(testDemoPath)
        cleanDemos(trainDemoPath)

    # Parse the scraped demos into splits
    os.system('python .\python\\demo_parse_loader.py %s %s %s %s'%(testFullGameRoundVectorPath, testFullGameSequenceVectorPath, testMetaDataPath, testSplitPath))
    os.system('python .\python\\demo_parse_loader.py %s %s %s %s'%(trainFullGameRoundVectorPath, trainFullGameSequenceVectorPath, trainMetaDataPath, trainSplitPath))

    # Create the Numpy batch files for training
    os.system('python .\python\\dataSplitter.py %s %s %s %s'%(testSplitDataPath, testSplitMapsPath, testSplitScoresPath, testSplitBreakPointsPath))
    os.system('python .\python\\dataSplitter.py %s %s %s %s'%(trainSplitDataPath, trainSplitMapsPath, trainSplitScoresPath, trainSplitBreakPointsPath))

if __name__ == "__main__":
    main()