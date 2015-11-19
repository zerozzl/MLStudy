import PCA
import EXTRAS
import numpy

dataMat = PCA.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch13/testSet.txt")
lowDMat, reconMat = PCA.pca(dataMat, 1)
PCA.plotPCA(dataMat, reconMat)

'''
dataMat = PCA.replaceNanWithMean("E:/TestDatas/MachineLearningInAction/Ch13/secom.data")
meanVals = numpy.mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = numpy.cov(meanRemoved, rowvar=0)
eigVals, eigVects = numpy.linalg.eig(numpy.mat(covMat))
print eigVals
'''

# EXTRAS.plotTestSet("E:/TestDatas/MachineLearningInAction/Ch13/testSet.txt")
# EXTRAS.plotTestSet3("E:/TestDatas/MachineLearningInAction/Ch13/testSet3.txt")
# EXTRAS.plotSecomPCA("E:/TestDatas/MachineLearningInAction/Ch13/secom.data")