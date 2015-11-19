import numpy as np
from numpy import linalg as la
import SVDRec

'''
data = SVDRec.loadExData()
U, Sigma, VT = numpy.linalg.svd(data)

Sig3 = numpy.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
rebuild = U[:, :3] * Sig3 * VT[:3, :]

print rebuild
'''

'''
myMat = np.mat(SVDRec.loadExData())
# print SVDRec.ecludSim(myMat[:, 0], myMat[:, 4])
# print SVDRec.ecludSim(myMat[:, 0], myMat[:, 0])

# print SVDRec.cosSim(myMat[:, 0], myMat[:, 4])
# print SVDRec.cosSim(myMat[:, 0], myMat[:, 0])

# print SVDRec.pearsSim(myMat[:, 0], myMat[:, 4])
# print SVDRec.pearsSim(myMat[:, 0], myMat[:, 0])
'''

'''
myMat = np.mat([[4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0]])

reco = SVDRec.recommend(myMat, 2)

print reco
'''

'''
U, Sigma, VT = la.svd(np.mat(SVDRec.loadExData2()))
Sig2 = Sigma ** 2
SigTotal = np.sum(Sig2)
print np.sum(Sig2[:3]) / SigTotal
'''

'''
myMat = np.mat(SVDRec.loadExData2())
reco = SVDRec.recommend(myMat, 1, estMethod=SVDRec.svdEst)
print reco
'''

SVDRec.imgCompress("E:/TestDatas/MachineLearningInAction/Ch14/0_5.txt")


