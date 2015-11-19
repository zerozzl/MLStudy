import numpy

def loadData(filepath):
    dataMat = []
    fr = open(filepath)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[numpy.nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[numpy.nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1

def regLeaf(dataSet):
    return numpy.mean(dataSet[:, -1])

def regErr(dataSet):
    return numpy.var(dataSet[:, -1]) * numpy.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    n = numpy.shape(dataSet)[1]
    S = errType(dataSet)
    bestS = numpy.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1): 
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if(numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if(numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if numpy.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(numpy.power(lSet[:, -1] - tree['left'], 2)) + sum(numpy.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(numpy.power(testData[:, -1] - treeMean, 2))
        if(errorMerge < errorNoMerge):
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m, n = numpy.shape(dataSet)
    X = numpy.mat(numpy.ones((m, n)))
    Y = numpy.mat(numpy.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0 : n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if numpy.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, connot do inverse,\n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(numpy.power(Y - yHat, 2))

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = numpy.shape(inDat)[1]
    X = numpy.mat(numpy.ones((1, n + 1)))
    X[:, 1: n + 1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = numpy.mat(numpy.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, numpy.mat(testData[i]), modelEval)
    return yHat

def effectContrast(trainData, testData):
    trainMat = numpy.mat(loadData(trainData))
    testMat = numpy.mat(loadData(testData))
    
    myTree_1 = createTree(trainMat, ops=(1, 20))
    yHat_1 = createForeCast(myTree_1, testMat[:, 0])
    corr_1 = numpy.corrcoef(yHat_1, testMat[:, 1], rowvar=0)[0, 1]
    
    myTree_2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat_2 = createForeCast(myTree_2, testMat[:, 0], modelTreeEval)
    corr_2 = numpy.corrcoef(yHat_2, testMat[:, 1], rowvar=0)[0, 1]
    
    ws, X, Y = linearSolve(trainMat)
    yHat_3 = numpy.mat(numpy.zeros((numpy.shape(testMat)[0], 1)))
    for i in range(numpy.shape(testMat)[0]):
        yHat_3[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    corr_3 = numpy.corrcoef(yHat_3, testMat[:, 1], rowvar=0)[0, 1]
    
    print "RegTree: %f\nModelTree: %f\nLinear: %f" % (corr_1, corr_2, corr_3)

