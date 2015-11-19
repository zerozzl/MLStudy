import numpy
from numpy import inf
import matplotlib.pyplot as plt

def loadSimpleData():
    dataMat = numpy.matrix([[1.0, 2.1],
                            [2.0, 1.1],
                            [1.3, 1.0],
                            [1.0, 1.0],
                            [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = numpy.ones((numpy.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = 1.0
    return retArray

def buildStump(dataArr, classLables, D):
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(classLables).T
    m, n = numpy.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = numpy.mat(numpy.zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = numpy.mat(numpy.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = numpy.shape(dataArr)[0]
    D = numpy.mat(numpy.ones((m, 1)) / m)
    aggClassEst = numpy.mat(numpy.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print "D: ", D.T
        alpha = float(0.5 * numpy.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print "classEst: ", classEst.T
        expon = numpy.multiply(-1 * alpha * numpy.mat(classLabels).T, classEst)
        D = numpy.multiply(D, numpy.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        # print "aggClassEst: ", aggClassEst.T
        aggErrors = numpy.multiply(numpy.sign(aggClassEst) != numpy.mat(classLabels).T, numpy.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print "total error: ", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(dataToClass, classifierArr):
    dataMat = numpy.mat(dataToClass)
    m = numpy.shape(dataMat)[0]
    aggClassEst = numpy.mat(numpy.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return numpy.sign(aggClassEst)

def loadDataSet(filepath):
    numFeat = len(open(filepath).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotROC(predStrengths, classLabes):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(numpy.array(classLabes) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabes) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabes[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rae')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    print "The Area Under the Curve is: ", ySum * xStep
    plt.show()
