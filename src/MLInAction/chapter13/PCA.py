import numpy
import matplotlib.pyplot as plt

def loadDataSet(filepath, delim='\t'):
    fr = open(filepath)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return numpy.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = numpy.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = numpy.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = numpy.linalg.eig(numpy.mat(covMat))
    eigValInd = numpy.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def plotPCA(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

def replaceNanWithMean(filepath):
    dataMat = loadDataSet(filepath, ' ')
    numFeat = numpy.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = numpy.mean(dataMat[numpy.nonzero(~numpy.isnan(dataMat[:, i].A))[0], i])
        dataMat[numpy.nonzero(numpy.isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat
