import Plot;
import LogisticRegression as LR;
import numpy as np;
import matplotlib.pyplot as plt;

def trainData1(folder, gradDecent):
    datas, labels = LR.loadDataSet(folder, "data1.txt");
    alpha = 0.001;
    lamb = 0;
#     itemNum = 200000; # batch
    itemNum = 10000;
    
    thetas = gradDecent(datas, labels, alpha, lamb, itemNum);
    print "thetas: ", thetas;
    
    thetas = np.mat(thetas);
    m = np.shape(datas)[0];
    error = 0;
    for i in range(m):
        pred = LR.classfy(thetas, np.mat(datas[i]));
        if pred != labels[i]:
            error = error + 1;
    errorRate = float(error) * 100 / m;
    print "The error rate of the test is %f percents" % errorRate
    
    thetas = np.array(thetas)[0];
    Plot.plot(folder, "data1.txt", thetas);

def trainData2(folder, gradDecent):
    datas, labels = LR.loadDataSet(folder, "data2.txt");
    alpha = 0.1;
    lamb = 1;
#     itemNum = 500; # batch
    itemNum = 100;
    thetas = gradDecent(datas, labels, alpha, lamb, itemNum);
    print "thetas: ", thetas;
    
    thetas = np.mat(thetas);
    m = np.shape(datas)[0];
    error = 0;
    for i in range(m):
        pred = LR.classfy(thetas, np.mat(datas[i]));
        if pred != labels[i]:
            error = error + 1;
    errorRate = float(error) * 100 / m;
    print "The error rate of the test is %f percents" % errorRate
    
    thetas = np.array(thetas)[0];
    Plot.plot(folder, "data2.txt", thetas);
    
def trainData3(folder, gradDecent):
    datas, labels = LR.loadDataSet(folder, "data3.txt");
    
    datasMap = [];
    degree = 6;
    m = np.shape(datas)[0];
    for k in range(m):
        tmp = [1];
        for i in range(degree):
            for j in range(i + 2):
                tmp.append(pow(datas[k][0], i + 1 - j) * pow(datas[k][1], j));
        datasMap.append(tmp);
    
    datas = np.mat(datasMap);
    
    alpha = 1;
    lamb = 1;
    
#     itemNum = 100; # batch
    itemNum = 500;
    thetas = gradDecent(datas, labels, alpha, lamb, itemNum);
    print "thetas: ", thetas;
    
    thetas = np.mat(thetas);
    m = np.shape(datas)[0];
    error = 0;
    for i in range(m):
        pred = LR.classfy(thetas, np.mat(datas[i]));
        if pred != labels[i]:
            error = error + 1;
    errorRate = float(error) * 100 / m;
    print "The error rate of the test is %f percents" % errorRate
      
    thetas = np.array(thetas)[0];
    Plot.plot(folder, "data3.txt", thetas);


folder = "E:/TestDatas/MLStudy/LogisticRegression/";
# # Plot.plot(folder, "data3.txt");
trainData3(folder, LR.batchGradDecent);

