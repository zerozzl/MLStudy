import Bayes
import feedparser

listOPosts, listClassed = Bayes.loadDataSet()

vocabList = Bayes.createVocabList(listOPosts)

# print vocabList

# print Bayes.setOfWordsToVec(vocabList, listOPosts[0])

# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(Bayes.setOfWordsToVec(vocabList, postinDoc))
    
# p0V, p1V, pAb = Bayes.trainNB0(trainMat, listClassed)
# print p0V
# print p1V
# print pAb

# Bayes.testingNB()

# Bayes.spamTest("E:/TestDatas/MachineLearningInAction/Ch04/");

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

# vocabList, pNY, pSF = Bayes.localWords(ny, sf)

Bayes.getTopWords(ny, sf)
