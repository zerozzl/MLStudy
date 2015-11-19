import numpy
import operator
import matplotlib.pyplot as plt

def loadDatas(filepath):
    fr = open(filepath)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = numpy.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFormLine = line.split("\t")
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector

def printOriginalData(datingDataMat, datingLabels):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
               (15.0 * numpy.array(datingLabels)), (15.0 * numpy.array(datingLabels)))
    ax2 = fig.add_subplot(222)
    ax2.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               (15.0 * numpy.array(datingLabels)), (15.0 * numpy.array(datingLabels)))
    ax3 = fig.add_subplot(223)
    ax3.scatter(datingDataMat[:, 0], datingDataMat[:, 2],
               (15.0 * numpy.array(datingLabels)), (15.0 * numpy.array(datingLabels)))
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify(target, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(target, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest(filepath):
    hoRatio = 0.10
    dataMat, labels = loadDatas(filepath)
    normMat, ranges, minVals = autoNorm(dataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, labels[i])
        if(classifierResult != labels[i]) : errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

def classifyPerson(filepath):
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, labels = loadDatas(filepath)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = numpy.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, labels, 3)
    print "You will probably like this person: %s" % resultList[classifierResult - 1]

