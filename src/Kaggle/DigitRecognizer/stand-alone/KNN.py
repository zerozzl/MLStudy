import numpy as np;
import operator;

def classify(target, dataSet, lables, k):
    m = dataSet.shape[0];
    diffMat = np.tile(target, (m, 1)) - dataSet;
    sqDiffMat = diffMat ** 2;
    sqDistinct = sqDiffMat.sum(axis=1);
    distinct = sqDistinct ** 0.5;
    sortIndex = distinct.argsort();
    classCount = {};
    for i in range(k):
        voteLabel = lables[sortIndex[i]];
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1;
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def process(testData, dataSet, lables):
    print 'begin classify data...'
    k = 5; # 5 is better than 10
    m = testData.shape[0];
    result = [];
    for i in range(m):
        num = classify(testData[i], dataSet, lables, k);
        result.append(num);
    return result;
