# coding: UTF-8
import numpy as np
import MNIST

def main():
    imgData = "E:/TestDatas/MLStudy/UFLDL/train-images.idx3-ubyte";
    labData = "E:/TestDatas/MLStudy/UFLDL/train-labels.idx1-ubyte";
    
    inputSize = 28 * 28;
    numLabels = 5;
    hiddenSize = 200;
    sparsityParam = 0.1;
    lamb = 3e-3;
    beta = 3;
    
    images = MNIST.loadImages(imgData);
    labels = MNIST.loadLabels(labData);
    
    labelSet = [];
    unlabeledSet = [];
    
    for i in range(len(labels)):
        if(labels[i] >= 5):
            unlabeledSet.append(i);
        else:
            labelSet.append(i);
    
    numTrain = len(labelSet) / 2;
    trainSet = labelSet[:numTrain];
    testSet = labelSet[numTrain:];
    
    unlabeledData = np.mat(np.zeros((inputSize, len(unlabeledSet))));
    for i in range(len(unlabeledSet)):
        unlabeledData[:, i] = images[:, unlabeledSet[i]];
         
    
# trainData   = mnistData(:, trainSet);
# trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5
# 
# testData   = mnistData(:, testSet);
# testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5
# 
# % Output Some Statistics
# fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
# fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
# fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

main();
    
