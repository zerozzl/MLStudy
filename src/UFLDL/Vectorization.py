# coding: UTF-8
import numpy as np
from scipy.optimize import fmin_l_bfgs_b;
import MNIST
import matplotlib.pyplot as plt

# 画出源数据
def plotDatas(folder, datas):
    fig = plt.figure();
    for i in range(100):
        fig.add_subplot(10, 10, i + 1);
        img = np.array(datas[:, i].T)[0];
        img = img.reshape(28, 28);
        plt.imshow(img , cmap='gray');
    plt.savefig(folder + "datas.png", format="png");

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

def sigmoidDeri(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)));

def saveResult(filepath, dataList):
    fileHandler = open(filepath, "w");
    for data in dataList:
        fileHandler.write(str(data) + "\n");
    fileHandler.close();

# 参数初始化
def initializeParameters(hiddenSize, visibleSize):
    r = np.sqrt(6) / np.sqrt(hiddenSize + visibleSize + 1);
    W1 = np.random.uniform(0, 1, [hiddenSize, visibleSize]) * 2 * r - r;
    W2 = np.random.uniform(0, 1, [visibleSize, hiddenSize]) * 2 * r - r;
    b1 = np.zeros([hiddenSize, 1]);
    b2 = np.zeros([visibleSize, 1]);
    
    thetas = [];
    tmp = np.array(np.reshape(W1, (1, hiddenSize * visibleSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(W2, (1, visibleSize * hiddenSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(b1, (1, hiddenSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(b2, (1, visibleSize)));
    thetas.extend(tmp[0]);
    
    return np.array(thetas);

# 代价、梯度计算函数
def sparseAutoencoderCost(thetas, visibleSize, hiddenSize,
                          lamb, sparsityParam, beta, patches):
    
    W1 = np.reshape(thetas[0:hiddenSize * visibleSize], (hiddenSize, visibleSize));
    W2 = np.reshape(thetas[hiddenSize * visibleSize : 2 * hiddenSize * visibleSize], (visibleSize, hiddenSize));
    b1 = np.reshape(thetas[2 * hiddenSize * visibleSize : 2 * hiddenSize * visibleSize + hiddenSize], (hiddenSize, 1));
    b2 = np.reshape(thetas[2 * hiddenSize * visibleSize + hiddenSize :], (visibleSize, 1));
    
    m = np.shape(patches)[1];  # 64 10000
    
    z2 = W1 * patches + b1;  # 25, 10000
    a2 = sigmoid(z2);
    z3 = W2 * a2 + b2;  # 64, 10000
    a3 = sigmoid(z3);
    
    Jcost = np.sum(np.square(a3 - patches)) / (2 * m);
    Jweight = (np.sum(np.square(W1)) + np.sum(np.square(W2))) / 2;
    
    rho = np.sum(a2, 1) / m;
    Jsparse = sum(np.multiply(sparsityParam, np.log(sparsityParam / rho))
                  + np.multiply((1 - sparsityParam), np.log((1 - sparsityParam) / (1 - rho))));
    
    cost = Jcost + lamb * Jweight + beta * Jsparse;
    
    d3 = np.multiply(-(patches - a3), sigmoidDeri(z3));  # 64, 10000
    sterm = beta * (-sparsityParam / rho + (1 - sparsityParam) / (1 - rho));
    
    d2 = np.multiply((W2.T * d3 + sterm), sigmoidDeri(z2));  # 25, 10000
    
    W1grad = d2 * patches.T / m + lamb * W1;
    W2grad = d3 * a2.T / m + lamb * W2;
    b1grad = np.sum(d2, 1) / m;
    b2grad = np.sum(d3, 1) / m;
    
    grads = [];
    tmp = np.array(np.reshape(W1grad, (1, hiddenSize * visibleSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(W2grad, (1, visibleSize * hiddenSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(b1grad, (1, hiddenSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(b2grad, (1, visibleSize)));
    grads.extend(tmp[0]);
    grads = np.array(grads);
    
    return cost, grads;

# 画出计算结果
def displayNetwork(patches, imgRow, imgCol, pRow, pCol, filepath=None):
    patches = np.mat(patches);
    fig = plt.figure();
    m = np.shape(patches)[1];
    for i in range(m):
        fig.add_subplot(imgRow, imgCol, i + 1);
        img = patches[:, i];
        img = img.reshape(pRow, pCol).T;
        plt.imshow(img , cmap='gray');
    
    if filepath is not None:
        plt.savefig(filepath, format='png');
    else:
        plt.show();

# 主函数
def main():
    dataFile = "E:/TestDatas/MLStudy/UFLDL/train-images.idx3-ubyte";
    resultFolder = "E:/TestDatas/MLStudy/UFLDL/result/Vectorization/";
    
    visibleSize = 28 * 28;
    hiddenSize = 196;
    sparsityParam = 0.1;
    lamb = 3e-3;
    beta = 3;
    
    images = MNIST.loadImages(dataFile, 10000);
    patches = images / 255.0;
    
    # 画图元数据
    plotDatas(resultFolder, patches);
       
    # 初始化数据集
    thetas = initializeParameters(hiddenSize, visibleSize);
        
    # 使用LBFGS进行优化
    optimThetas = fmin_l_bfgs_b(func=sparseAutoencoderCost, x0=thetas,
                  args=(visibleSize, hiddenSize, lamb, sparsityParam, beta, patches),
                  maxiter=500)[0];
        
    # 保存数据
    saveResult(resultFolder + "thetas.txt", optimThetas);
        
    # 可视化结果
    W1 = np.reshape(optimThetas[0 : hiddenSize * visibleSize], (hiddenSize, visibleSize));
    displayNetwork(W1.T, 14, 14, 28, 28, resultFolder + "results.png");


main();
