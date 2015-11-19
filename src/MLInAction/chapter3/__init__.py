import Trees
import TreePlotter

# Trees

'''
# myDat, labels = Trees.createDataSet()

# shannonEnt = Trees.calcShannonEnt(myDat)
# print shannonEnt

# retDataSet = Trees.splitDataSet(myDat, 0, 1)
# print retDataSet

# bestFeature = Trees.chooseBestFeatureToSplit(myDat)
# print bestFeature

# tree = Trees.createTree(myDat, labels)
# print tree
'''

# TreePlotter
'''
# TreePlotter.createPlot()

tree = TreePlotter.retrieveTree(0)

# numLeafs = TreePlotter.getNumLeafs(tree)
# depth = TreePlotter.getTreeDepth(tree)
# print "leafs nums: %d, depth: %d" % (numLeafs, depth)

TreePlotter.createPlot(tree)
'''

# Classify
'''
# myDat, labels = Trees.createDataSet()
# myTree = TreePlotter.retrieveTree(0)
# result = Trees.classify(myTree, labels, [1, 0])
# print "result is: %s" % result

# Trees.storeTree(myTree, "E:/TestDatas/MachineLearningInAction/Ch03/classifierStorage.txt")
# tree = Trees.grabTree("E:/TestDatas/MachineLearningInAction/Ch03/classifierStorage.txt")
# print tree

# Trees.lensesStudy("E:/TestDatas/MachineLearningInAction/Ch03/lenses.txt")
'''
