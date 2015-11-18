# coding: UTF-8
import numpy as np
from scipy.optimize import fmin_l_bfgs_b;
import matplotlib.pyplot as plt

# 读取图片数据
def loadDataSet(filepath):
    fr = open(filepath);
    datas = [];
    for line in fr.readlines():
        arr = line.strip().split();
        datas.append([float(x) for x in arr]);
    return datas;

# 画出原图像
def plotDatas(folder, imgs):
    m = np.shape(imgs)[0];
    for i in range(m):
        img = imgs[i];
        img = np.mat(img);
        fig = plt.figure();
        fig.add_subplot(111);
        img = img.reshape(512, 512).T;
        plt.imshow(img , cmap="gray");
        
        filename = folder + "img_" + str(i) + ".png";
        plt.savefig(filename, format="png");

# 生成测试数据
def sampleIMAGES(datas):
    patchsize = 8;
    numpatches = 10000;
    patches = np.zeros((patchsize * patchsize, numpatches));

    m = np.shape(datas)[0];
    for imageNum in range(m):  # 在每张图片中随机选取1000个patch，共10000个patch
        img = datas[imageNum];
        img = img.reshape(512, 512).T;
        rowNum, colNum = np.shape(img);
        for patchNum in range(1000):  # 实现每张图片选取1000个patch
            xPos = int(np.random.uniform(0, rowNum - patchsize));
            yPos = int(np.random.uniform(0, colNum - patchsize));
            patches[:, imageNum * 1000 + patchNum] = img[xPos : xPos + 8, yPos : yPos + 8].reshape(64);
    
    patches = normalizeData(patches);
    return patches;

# 均值归一化
def normalizeData(patches):
    patches = patches - np.mean(patches, axis=0);
    pstd = np.mat(3 * np.std(patches, axis=0));
    patches = patches / pstd;
    patches = (patches + 1) * 0.4 + 0.1;
    return patches;

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

def computeNumericalGradient(J, thetas, visibleSize, hiddenSize,
                          lamb, sparsityParam, beta, patches):
    n = np.shape(thetas)[0];
    numgrad = np.zeros(n);
    epsilon = 1e-4;
    E = np.eye(n);
    
    for i in range(min(n, 100)):
        delta = E[:, i] * epsilon;
        numgrad[i] = (J((thetas + delta), visibleSize, hiddenSize,
                          lamb, sparsityParam, beta, patches)[0]
                      - J((thetas - delta), visibleSize, hiddenSize,
                          lamb, sparsityParam, beta, patches)[0]) / (epsilon * 2.0);
    
    return numgrad;

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

def sigmoidDeri(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)));

def saveResult(filepath, dataList):
    fileHandler = open(filepath, "w");
    for data in dataList:
        fileHandler.write(str(data) + "\n");
    fileHandler.close();

#################################### 梯度检查 ####################################

def simpleQuadraticFunction(x):
    value = x[0] * x[0] + 3 * x[0] * x[1];
    grad = np.zeros(2);
    grad[0] = 2 * x[0] + 3 * x[1];
    grad[1] = 3 * x[0];
    return value, grad;

def checkNumericalGradient():
    x = [4, 10];
    grad = simpleQuadraticFunction(x)[1];  # value=126, grad[0]=38, grad[1]=12;
    
    numgrad = computeNumericalGradient(simpleQuadraticFunction, x);
    print numgrad;
    print grad;
    print('The above two rows you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');
    
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad);
    print diff;
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');

#################################### 梯度检查 ####################################


# 主函数
def main():
    dataFile = "E:/TestDatas/MLStudy/UFLDL/IMAGES.dat";
    resultFolder = "E:/TestDatas/MLStudy/UFLDL/result/SparseAutoencoder/";
    
    visibleSize = 8 * 8;
    hiddenSize = 25;
    sparsityParam = 0.01;
    lamb = 0.0001;
    beta = 3;
    checkGrad = False;
    
    datas = loadDataSet(dataFile);
    datas = np.mat(datas);
    
    # 画出原图
    plotDatas(resultFolder, datas);
    
    patches = sampleIMAGES(datas);
     
    # 画图数据集
    displayNetwork(patches[:,[int(x) for x in np.random.uniform(0, np.shape(patches)[1], 225)]], 15, 15, 8, 8, resultFolder + "patches.png");
     
    # 初始化数据集
    thetas = initializeParameters(hiddenSize, visibleSize);
     
    # 是否进行梯度检验
    if checkGrad:
        grads = sparseAutoencoderCost(thetas, visibleSize, hiddenSize,
                              lamb, sparsityParam, beta, patches)[1];
        numgrad = computeNumericalGradient(sparseAutoencoderCost, thetas, visibleSize, hiddenSize,
                              lamb, sparsityParam, beta, patches);
        for i in range(100):
            print grads[i], '    ', numgrad[i]; 
     
    # 使用LBFGS进行优化
    optimThetas = fmin_l_bfgs_b(func=sparseAutoencoderCost, x0=thetas,
                  args=(visibleSize, hiddenSize, lamb, sparsityParam, beta, patches),
                  maxiter=500)[0];
     
    # 保存数据
    saveResult(resultFolder + "thetas.txt", optimThetas);
     
    # 可视化结果
    W1 = np.reshape(optimThetas[0 : hiddenSize * visibleSize], (hiddenSize, visibleSize));
    displayNetwork(W1.T, 5, 5, 8, 8, resultFolder + "results.png");
    

# checkNumericalGradient();

main();

