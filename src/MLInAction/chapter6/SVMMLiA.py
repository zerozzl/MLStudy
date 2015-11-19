import random
import numpy
import os

def loadDataSet(filepath):
    dataMat = []
    labelMat = []
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    b = 0
    m = numpy.shape(dataMat)[0]
    alphas = numpy.mat(numpy.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(numpy.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(numpy.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print "L == H"
                    continue
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMat[j, :] * dataMat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration num: %d" % iter
    return b, alphas

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = numpy.shape(dataMatIn)[0]
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))
        self.b = 0
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))
        self.K = numpy.mat(numpy.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    fXk = float(numpy.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = numpy.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print "L == H"
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print "eta >= 0"
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
    
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(numpy.mat(dataMatIn), numpy.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:
            nonBoundIs = numpy.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print "non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(X)
    w = numpy.zeros((n, 1))
    for i in range(m):
        w += numpy.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def kernelTrans(X, A, kTup):
    m = numpy.shape(X)[0]
    K = numpy.mat(numpy.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = numpy.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def testRbf(trainingData, testData, k1=1.3):
    dataArr, labelArr = loadDataSet(trainingData)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "There are %d Support Vectors" % numpy.shape(sVs)[0]
    m, n = numpy.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kenelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kenelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print "The training error rate is: %f" % (float(errorCount) / m)
    
    dataArr, labelArr = loadDataSet(testData)
    errorCount = 0
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    m, n = numpy.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kenelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print "The test error rate is: %f" % (float(errorCount) / m)

def imgToVector(filepath):
    vect = numpy.zeros((1, 1024))
    fr = open(filepath)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vect[0, 32 * i + j] = int(line[j])
    return vect

def loadImages(folder):
    hwLabels = []
    trainingFileList = os.listdir(folder)
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        file = trainingFileList[i]
        fileName = file.split(".")[0]
        classNum = int(fileName.split("_")[0])
        if classNum == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = imgToVector('%s/%s' % (folder, file))
    return trainingMat, hwLabels

def testDigits(trainingData, testData, kTup=('rbf', 10)):
    dataArr, labelArr = loadImages(trainingData)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print "There are %d Support Vectors" % numpy.shape(sVs)[0]
    m, n = numpy.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kenelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kenelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print "The training error rate is: %f" % (float(errorCount) / m)

    dataArr, labelArr = loadImages(testData)
    errorCount = 0
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(labelArr).transpose()
    m, n = numpy.shape(dataMat)
    for i in range(m):
        kenelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kenelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]):
            errorCount += 1
    print "The test error rate is: %f" % (float(errorCount) / m)
