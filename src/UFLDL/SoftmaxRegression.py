#coding: UTF-8
import struct
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# 读取图片数据
def loadImages(filepath):
    binfile = open(filepath , 'rb')
    buf = binfile.read()
    datas = [];
    
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    
    for i in range(numImages):
        im = struct.unpack_from('>784B' ,buf, index)
        index += struct.calcsize('>784B')
        datas.append(im);
    
    datas = np.mat(datas).T;
    return datas;

# 读取标签数据
def loadLabels(filepath):
    binfile = open(filepath , 'rb')
    buf = binfile.read()
    labels = [];
    
    index = 0
    magic, numLabels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    
    for i in range(numLabels):
        la = struct.unpack_from('>1B' ,buf, index)
        index += struct.calcsize('>1B')
        labels.append(int(la[0]));
    
    return labels;

# 读取参数
def loadThetas(filepath):
    fr = open(filepath);
    thetas = [];
    for line in fr:
        thetas.append(float(line.strip()));
    return thetas;

# 代价函数
def softmaxCost(thetas, numClasses, inputSize, lamb, data, labels):
    thetas = thetas.reshape(numClasses, inputSize);
    numCases = np.shape(data)[1];
    
    groundTruth = np.zeros((numCases, 10));
    for i in range(numCases):
        groundTruth[i][labels[i]] = 1;
    
    M = thetas * data;
    M = M - np.max(M, 0);
    M = np.exp(M);
    M = M / np.sum(M, 0);
    
    cost = (np.sum(groundTruth.reshape(1, numCases * 10)
                  * np.log(M.T).reshape(numCases * 10, 1))
                  / (-numCases)) + (lamb / 2) * np.sum(np.square(thetas));
    
    thetasGrad = ((groundTruth.T - M) * data.T / (-numCases)) + lamb * thetas;
    thetasGrad = thetasGrad.reshape(1, numClasses * inputSize);
    
    thetasGrad = np.array(thetasGrad)[0];
    
    return cost, thetasGrad;

# 梯度计算
def computeNumericalGradient(J, thetas, numClasses, inputSize, lamb, data, labels):
    n = np.shape(thetas)[0];
    numgrad = np.zeros(n);
    epsilon = 1e-4;
    E = np.eye(n);
    
    for i in range(min(n, 50)):
        delta = E[:, i] * epsilon;
        numgrad[i] = (J((thetas + delta), numClasses, inputSize, lamb, data, labels)[0]
                      - J((thetas - delta), numClasses, inputSize, lamb, data, labels)[0]) / (epsilon * 2.0);
    
    return numgrad;

# 保存结果
def saveResult(filepath, dataList):
    fileHandler = open(filepath, "w");
    for data in dataList:
        fileHandler.write(str(data) + "\n");
    fileHandler.close();

# 训练参数
def train(imgData, labData, resultFolder):
    inputSize = 28 * 28;
    numClasses = 10;
    lamb = 1e-4;
    checkGrad = False;
    
    images = loadImages(imgData);
    labels = loadLabels(labData);
    
    thetas = 0.005 * np.random.random(inputSize * numClasses);
    
    # 是否进行梯度检验
    if checkGrad:
        grads = softmaxCost(thetas, numClasses, inputSize, lamb, images, labels)[1];
        numgrad = computeNumericalGradient(softmaxCost,
                        thetas,numClasses, inputSize, lamb, images, labels);
        for i in range(50):
            print grads[i], '    ', numgrad[i];
    
    # 使用LBFGS进行优化
    optimThetas = fmin_l_bfgs_b(func=softmaxCost, x0=thetas,
                  args=(numClasses, inputSize, lamb, images, labels),
                  maxiter=100)[0];
    
    # 保存数据
    saveResult(resultFolder + "thetas.txt", optimThetas);

# 预测数据
def predict(imgData, labData, resultFolder):
    inputSize = 28 * 28;
    numClasses = 10;
    
    images = loadImages(imgData);
    labels = loadLabels(labData);
    thetas = np.array(loadThetas(resultFolder + "thetas.txt"));
    
    thetas = thetas.reshape(numClasses, inputSize);
    
    pred = np.argmax(thetas * images, 0);
    pred = np.array(pred)[0];
    
    error = 0;
    m = np.shape(labels)[0];
    
    for i in range(m):
        if pred[i] != labels[i]:
            error += 1;
    
    print "Accuracy: ", (m - error * 1.0) / m;

# 主函数 
def main():
    trainImgData = "E:/TestDatas/MLStudy/UFLDL/train-images.idx3-ubyte";
    trainLabData = "E:/TestDatas/MLStudy/UFLDL/train-labels.idx1-ubyte";
    testImgData = "E:/TestDatas/MLStudy/UFLDL/t10k-images.idx3-ubyte";
    testLabData = "E:/TestDatas/MLStudy/UFLDL/t10k-labels.idx1-ubyte";
    resultFolder = "E:/TestDatas/MLStudy/UFLDL/result/SoftmaxRegression/";
    
#     train(trainImgData, trainLabData, resultFolder);
    predict(testImgData, testLabData, resultFolder);

main();

