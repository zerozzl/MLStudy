import numpy as np;
from sklearn.cross_validation import train_test_split;
import Plot;

def loadDataSet(folder, filename):
    datas = [];
    labels = [];
    fr = open(folder + filename);
    for line in fr.readlines():
        if filename == "data1.txt":
            arr = line.strip().split(",");
            datas.append([float(1), float(arr[0]), float(arr[1])]);
            labels.append(int(arr[2]));
        elif filename == "data2.txt":
            arr = line.strip().split();
            datas.append([float(1), float(arr[0]), float(arr[1])]);
            labels.append(int(arr[2]));
        elif filename == "data3.txt":
            arr = line.strip().split(",");
            datas.append([float(arr[0]), float(arr[1])]);
            labels.append(int(arr[2]));
    return datas, labels;

def splitData(datas, labels, type=0):
    trainDatas, testDatas, trainLabels, testLabels = train_test_split(datas, labels, test_size = 0.2);
    if type == 0:
        return trainDatas, trainLabels, testDatas, testLabels;
    elif type == 1:
        trainDatas, cvDatas, trainLabels, cvLabels = train_test_split(trainDatas, trainLabels, test_size = 0.25);
        return trainDatas, trainLabels, cvDatas, cvLabels, testDatas, testLabels;
    else:
        return datas, labels;

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

def classfy(theta, x):
    h = sigmoid(x * theta.T);
    if h >= 0.5:
        return 1;
    else:
        return 0;

def batchGradDecent(X, Y, alpha, lamb, iterNum):
    X = np.mat(X);
    Y = np.mat(Y);
    m, n = np.shape(X);
    thetas = np.zeros((1, n));
    costs = [];
    for i in range(iterNum):
        thetas_re = [0];
        thetas_re.extend(np.array(thetas)[0][1:]);
        thetas_re = np.mat(thetas_re);
        cost = (-1.0 / m) * np.sum(Y * np.log(sigmoid(X * thetas.T))
                               + (1 - Y) * np.log(1 - sigmoid(X * thetas.T))) + (lamb / (2 * m)) * np.sum(thetas_re * thetas_re.T);
        grads = (1.0 / m) * (X.T * (sigmoid(X * thetas.T) - Y.T)) + (lamb / m) * thetas_re.T;
        thetas = thetas - alpha * grads.T
        costs.append([i+1, cost]);
        print "Iterater time: %d, cost: %f" % ((i+1), cost);
    Plot.plotCost(costs);
    return np.array(thetas)[0];

def stocGradDecent(X, Y, alpha, lamb, iterNum):
    X = np.mat(X);
    Ymat = np.mat(Y);
    m, n = np.shape(X);
    thetas = np.zeros((1, n));
    costs = [];
    for i in range(iterNum):
        dataIndex = range(m);
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01;
            randIndex = int(np.random.uniform(0, len(dataIndex)));
            
            thetas_re = [0];
            thetas_re.extend(thetas[0][1:]);
            thetas_re = np.mat(thetas_re);
            
            grads = (X[randIndex,:] * np.sum(sigmoid(X[randIndex,:] * thetas.T) - Y[randIndex])) + lamb * thetas_re;
            thetas = thetas - alpha * grads
            
            del(dataIndex[randIndex])
        
        thetas_re = [0];
        thetas_re.extend(thetas[0][1:]);
        thetas_re = np.mat(thetas_re);
        cost = (-1.0 / m) * np.sum(Ymat * np.log(sigmoid(X * thetas.T))
                               + (1 - Ymat) * np.log(1 - sigmoid(X * thetas.T))) + (lamb / (2 * m)) * np.sum(thetas_re * thetas_re.T);
        costs.append([i+1, cost]);
        print "Iterater time: %d, cost: %f" % ((i+1), cost);
    
    Plot.plotCost(costs);
    return np.array(thetas)[0];

