import EXTRAS
import AdaBoost
import numpy

# EXTRAS.plotSimpleData()

# dataMat, labels = AdaBoost.loadSimpleData()

# D = numpy.mat(numpy.ones((5, 1)) / 5)

# bestStump, minError, bestClasEst = AdaBoost.buildStump(dataMat, labels, D)

dataMat, labels = AdaBoost.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch07/horseColicTraining2.txt")

classifierArr, aggClassExt = AdaBoost.adaBoostTrainDS(dataMat, labels, 10)

AdaBoost.plotROC(aggClassExt.T, labels)