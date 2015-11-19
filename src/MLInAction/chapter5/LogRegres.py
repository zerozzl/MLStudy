import numpy
import random
import matplotlib.pyplot as plt

def loadDataSet(filepath):
    dataMat = []
    label = []
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append(int(lineArr[2]))
    return dataMat, label

def sigmoid(inX):
    return 1.0 / (1 + numpy.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    n = numpy.shape(dataMatrix)[1]
    alpha = 0.001
    maxCycles = 500
    weights = numpy.ones((n, 1))
    while(maxCycles > 0):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        maxCycles -= 1
    return weights

def stocGradAscent0(dataMat, classLabels, numIter=150):
    m, n = numpy.shape(dataMat)
    alpha = 0.01
    weights = numpy.ones(n)
    for j in range(numIter):
        for i in range(m):
            h = sigmoid(sum(dataMat[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMat[i]
    return weights

def stocGradAscent1(dataMat, classLabels, numIter=150):
    m, n = numpy.shape(dataMat)
    weights = numpy.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(dataMat, labelMat, weights):
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def classify(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0;
    
def colicTest(folder):
    frTrain = open(folder + "/horseColicTraining.txt")
    frTest = open(folder + "/horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    
    trainWeights = stocGradAscent1(numpy.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classify(numpy.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    
    errorRate = (float(errorCount) / numTestVec)
    print "The error rate of the test is %f" % errorRate
    return errorRate

def multiTest(folder):
    numTests = 10
    errorSum = 0.0
    i = numTests
    while(i > 0):
        errorSum += colicTest(folder)
        i -= 1
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))

    