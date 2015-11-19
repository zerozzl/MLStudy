import Regression

# Regression
'''
# dataMat, labelMat = Regression.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch08/ex0.txt")

# ws = Regression.standRegres(dataMat, labelMat)

# Regression.plotStandRegres(dataMat, labelMat, ws)

# Regression.lwlr(dataMat[0], dataMat, labelMat, 1.0)

# yHat = Regression.lwlrTest(dataMat, dataMat, labelMat, 0.01)

# Regression.plotLwlr(dataMat, labelMat, yHat)
'''

# Reduce coefficient
'''
# dataMat, labelMat = Regression.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch08/abalone.txt")

# ridgeWeights = Regression.ridgeTest(dataMat, labelMat)

# returnMat = Regression.stageWise(dataMat, labelMat, 0.005, 1000)

# Regression.plotParamTrend(returnMat)
'''

# LEGO

# Regression.legoDataCollect("E:/TestDatas/MachineLearningInAction/Ch08/lego/")

dataArr, labelArr = Regression.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch08/lego/legoData.txt")

# ws = Regression.legoStandRegres(dataArr, labelArr)

Regression.crossValidation(dataArr, labelArr, 10)

ridgeWeights = Regression.stageWise(dataArr, labelArr)

