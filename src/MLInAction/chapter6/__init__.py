import SVMMLiA
import EXTRAS

# SVMMLiA

# dataMat, labelMat = SVMMLiA.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch06/testSet.txt")

# b, alphas = SVMMLiA.smoSimple(dataMat, labelMat, 0.6, 0.001, 40)

# b, alphas = SVMMLiA.smoP(dataMat, labelMat, 0.6, 0.001, 40)

# ws = SVMMLiA.calcWs(alphas, dataMat, labelMat)

# SVMMLiA.testRbf("E:/TestDatas/MachineLearningInAction/Ch06/testSetRBF.txt", "E:/TestDatas/MachineLearningInAction/Ch06/testSetRBF2.txt")

SVMMLiA.testDigits("E:/TestDatas/MachineLearningInAction/Ch06/digits/trainingDigits", "E:/TestDatas/MachineLearningInAction/Ch06/digits/testDigits")

# EXTRAS
'''
# EXTRAS.plotRBF("E:/TestDatas/MachineLearningInAction/Ch06/testSetRBF2.txt")
# EXTRAS.plotSupportVectors("E:/TestDatas/MachineLearningInAction/Ch06/testSet.txt")
# EXTRAS.plotSupportVectors("E:/TestDatas/MachineLearningInAction/Ch06/testSet.txt")
'''