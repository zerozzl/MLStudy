import Apriori

dataSet = Apriori.loadDataSet()
# C1 = Apriori.createC1(dataSet)
# D = map(set, dataSet)
# L1, suppData = Apriori.scanD(D, C1, 0.5)
L, suppData = Apriori.apriori(dataSet, minSupport=0.5)
rules = Apriori.generateRules(L, suppData, minConf=0.7)

'''
Apriori.mushTest("E:/TestDatas/MachineLearningInAction/Ch11/mushroom.dat")
'''