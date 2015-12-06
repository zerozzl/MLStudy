import Pretreatment
import KNN

'''
# Generate Image
trainDatas, trainLabels = Pretreatment.loadTrainData('/home/hadoop/workdatas/kaggle/DigitRecognizer/train_sort.csv');
trainLabels = trainLabels[0];
Pretreatment.generateImage('/home/hadoop/workdatas/kaggle/DigitRecognizer/imgs/', trainDatas, trainLabels);
'''


'''
# KNN Test
import numpy
group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']

tar = [1.0, 1.2];

result = KNN.classify(tar, group, labels, 3);

print result;
'''

# KNN Clasify
trainDatas, trainLabels = Pretreatment.loadTrainData('/home/hadoop/workdatas/kaggle/DigitRecognizer/train.csv');
trainLabels = trainLabels[0];
testDatas = Pretreatment.loadTestData('/home/hadoop/workdatas/kaggle/DigitRecognizer/test.csv');
result = KNN.process(testDatas, trainDatas, trainLabels);
Pretreatment.generateResultFile('/home/hadoop/workdatas/kaggle/DigitRecognizer/result_knn_10.csv', result);

print 'success'
