import KMeans
import numpy

'''
dataMat = mat(KMeans.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch10/testSet.txt"))
k = 4
centroids, clustAssing = KMeans.kMeans(dataMat, k)
KMeans.showCluster(dataMat, k, centroids, clustAssing)
'''

'''
dataMat = numpy.mat(KMeans.loadDataSet("E:/TestDatas/MachineLearningInAction/Ch10/testSet2.txt"))
k = 3
centroids, clustAssing = KMeans.bitKmeans(dataMat, k)
KMeans.showCluster(dataMat, k, centroids, clustAssing)
'''

KMeans.clusterClubs("E:/TestDatas/MachineLearningInAction/Ch10/places.txt",
                    "E:/TestDatas/MachineLearningInAction/Ch10/Portland.png")
