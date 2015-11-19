import FPGrowth

'''
rootNode = FPGrowth.treeNode('pyramid', 9, None)
rootNode.children['eye'] = FPGrowth.treeNode('eye', 13, None)
rootNode.children['phoenix'] = FPGrowth.treeNode('phoenix', 3, None)
rootNode.disp()
'''

'''
simpDat = FPGrowth.loadSimpDat()
initSet = FPGrowth.createInitSet(simpDat)
myFPtree, myHeaderTab = FPGrowth.createTree(initSet, 3)
# myFPtree.disp()

# print FPGrowth.findPrefixPath('t', myHeaderTab['t'][1])

freqItems = []
FPGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
print freqItems
'''

FPGrowth.kosarakTest("E:/TestDatas/MachineLearningInAction/Ch12/kosarak.dat")

