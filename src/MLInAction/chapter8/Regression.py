import numpy
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

def loadDataSet(filepath):
    numFeat = len(open(filepath).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    xTx = xMat.T * xMat
    if numpy.linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def plotStandRegres(xArr, yArr, ws):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    corr = numpy.corrcoef(yHat.T, yMat)
    print "correlation coefficient: ", corr
    plt.show()

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    m = numpy.shape(xMat)[0]
    weights = numpy.mat(numpy.eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = numpy.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if numpy.linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = numpy.shape(testArr)[0]
    yHat = numpy.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotLwlr(xArr, yArr, yHat):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s=2, c='red')
    plt.show()

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + numpy.eye(numpy.shape(xMat)[1]) * lam
    if numpy.linalg.det(denom) == 0.0:
        print "This matrix is singular, canno do reverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    # yMean = numpy.mean(yMat, 0)
    # yMat = yMat - yMean
    xMeans = numpy.mean(xMat, 0)
    xVar = numpy.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = numpy.zeros((numTestPts, numpy.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, numpy.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def plotParamTrend(weights):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weights)
    plt.show()

def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = numpy.mean(inMat, 0)  # calc mean then subtract it off
    inVar = numpy.var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    xMat = regularize(xMat)
    n = numpy.shape(xMat)[1]
    returnMat = numpy.zeros((numIt, n))
    ws = numpy.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = numpy.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def scrapePage(inFile, outFile, yr, numPce, origPrc):
    fr = open(inFile)
    fw = open(outFile, 'a')
    soup = BeautifulSoup(fr.read())
    i = 1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print "item #%d did not sell" % i
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')  # strips out $
            priceStr = priceStr.replace(',', '')  # strips out ,
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
            print "%s\t%d\t%s" % (priceStr, newFlag, title)
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()

def legoDataCollect(folder):
    scrapePage(folder + '/setHtml/lego8288.html', folder + 'legoData.txt', 2006, 800, 49.99)
    scrapePage(folder + '/setHtml/lego10030.html', folder + 'legoData.txt', 2002, 3096, 269.99)
    scrapePage(folder + '/setHtml/lego10179.html', folder + 'legoData.txt', 2007, 5195, 499.99)
    scrapePage(folder + '/setHtml/lego10181.html', folder + 'legoData.txt', 2007, 3428, 199.99)
    scrapePage(folder + '/setHtml/lego10189.html', folder + 'legoData.txt', 2008, 5922, 299.99)
    scrapePage(folder + '/setHtml/lego10196.html', folder + 'legoData.txt', 2009, 3263, 249.99)

def legoStandRegres(dataArr, labelArr):
    m, n = numpy.shape(dataArr)
    dataMat = numpy.mat(numpy.ones((m, n + 1)))
    dataMat[:, 1:5] = numpy.mat(dataArr)
    ws = standRegres(dataMat, labelArr)
    return ws

def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = numpy.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        numpy.random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = numpy.mat(testX)
            matTrainX = numpy.mat(trainX)
            meanTrain = numpy.mean(matTrainX, 0)
            varTrain = numpy.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * numpy.mat(wMat[k, :]).T + numpy.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, numpy.array(testY))
    meanErrors = numpy.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[numpy.nonzero(meanErrors == minMean)]
    xMat = numpy.mat(xArr)
    yMat = numpy.mat(yArr).T
    meanX = numpy.mean(xMat, 0)
    varX = numpy.var(xMat, 0)
    unReg = bestWeights / varX
    print "the best model from Ridge Regression is :\n", unReg
    print "with constant term: ", -1 * sum(numpy.multiply(meanX, unReg)) + numpy.mean(yMat)
    
    
    