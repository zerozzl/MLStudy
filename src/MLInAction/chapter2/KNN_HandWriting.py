import numpy
import os
import operator

def imgToVector(filepath):
    vect = numpy.zeros((1, 1024))
    fr = open(filepath)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vect[0, 32 * i + j] = int(line[j])
    return vect

def classify(target, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(target, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def handwritingClassTest(rootFolder):
    trainingDataFolder = rootFolder + "trainingDigits"
    testDataFolder = rootFolder + "testDigits"
    hwLabels = []
    trainingFileList = os.listdir(trainingDataFolder)
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        filepath = trainingFileList[i]
        filename = filepath.split(".")[0]
        classNum = int(filename.split("_")[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = imgToVector(trainingDataFolder + "/%s" % filepath)
    
    testFileList = os.listdir(testDataFolder)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filepath = testFileList[i]
        filename = filepath.split(".")[0]
        classNum = int(filename.split("_")[0])
        vectorUnderTest = imgToVector(testDataFolder + "/%s" % filepath)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "The classifier came back with: %d, the real answer is: %d" %  (classifierResult, classNum)
        if(classifierResult != classNum): errorCount += 1.0
        
    print "\nThe total number of errors id: %d" % errorCount
    print "\nThe total error rate is: %f" % (errorCount/float(mTest))
