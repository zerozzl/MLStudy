import KNN_Test
import KNN_Dating
import KNN_HandWriting
import EXTRAS

# KNN_Test
group, labels = KNN_Test.createDataSet()

tInX = [0, 0]
tLabel = KNN_Test.classify(tInX, group, labels, 3)

print("tInX ", tInX, " label is ", tLabel)

# KNN_Dating
'''
# dataMat, labelVector = KNN_Dating.loadDatas("E:/TestDatas/MachineLearningInAction/Ch02/datingTestSet2.txt")

# KNN_Dating.printOriginalData(dataMat, labelVector)

# KNN_Dating.autoNorm(dataMat)

# KNN_Dating.classifyPerson("E:/TestDatas/MachineLearningInAction/Ch02/datingTestSet.txt")

'''
# KNN_HandWriting
'''
# vect = KNN_HandWriting.imgToVector("E:/TestDatas/MachineLearningInAction/Ch02/digits/trainingDigits/0_0.txt")
# print vect[0, 32:63]

KNN_HandWriting.handwritingClassTest("E:/TestDatas/MachineLearningInAction/Ch02/digits/")
'''

# EXTRAS
'''
# EXTRAS.createDatingDist("E:/TestDatas/MachineLearningInAction/Ch02/EXTRAS/")
# EXTRAS.createDatingDist_2("E:/TestDatas/MachineLearningInAction/Ch02/EXTRAS/")
'''
