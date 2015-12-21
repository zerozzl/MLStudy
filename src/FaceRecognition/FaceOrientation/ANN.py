# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

# 读取PGM图片
def read_pgm(pgm):
    fr = open(pgm);
    header = fr.readline();
    magic = header.split()[0];
    maxsample = 1 if (magic == 'P4') else 0;
    while (len(header.split()) < 3 + (1, 0)[maxsample]):
        s = fr.readline();
        header += s if (len(s) and s[0] != '#') else '';
    width, height = [int(item) for item in header.split()[1:3]];
    samples = 3 if (magic == 'P6') else 1;
    if(maxsample == 0):
        maxsample = int(header.split()[3]);
    pixels = np.fromfile(fr, count=width * height * samples, dtype='u1' if maxsample < 256 else '>u2');
#     pixels = pixels.reshape(height, width) if samples == 1 else pixels.reshape(height, width, samples);
    return pixels, height, width;

# 画出PGM图片
def plot_pgm(pgm):
    img = np.mat(pgm);
    fig = plt.figure();
    fig.add_subplot(111);
    plt.imshow(img , cmap="gray");
    plt.show();

# 读取图片数据及
def loadDataSet(folder, filename):
    datas = [];
    labels = [];  # 0: left, 1: straight, 2: right, 3: up
    fr = open(folder + filename);
    for line in fr.readlines():
        orien = line.strip().split('_')[1];
        lab_vec = [0, 0, 0, 0];
        if orien == 'left':
            lab_vec[0] = 1;
        elif orien == 'straight':
            lab_vec[1] = 1;
        elif orien == 'right':
            lab_vec[2] = 1;
        elif orien == 'up':
            lab_vec[3] = 1;
        else:
            raise NameError('orientation not recognize!');
        labels.append(lab_vec);
        datas.append(read_pgm(folder + line.strip())[0]);
    datas = np.mat(datas) / 255.0;
    labels = np.mat(labels);
    return datas, labels;

# 初始化参数
def initThetas(inputSize, hiddenSize, outputSize):
    r = np.sqrt(6) / np.sqrt(hiddenSize + inputSize + 1);
    w1 = np.random.uniform(0, 1, [hiddenSize, inputSize]) * 2 * r - r;
    w2 = np.random.uniform(0, 1, [outputSize, hiddenSize]) * 2 * r - r;
    b1 = np.zeros([hiddenSize, 1]);
    b2 = np.zeros([outputSize, 1]);
    
    thetas = [];
    tmp = np.array(np.reshape(w1, (1, hiddenSize * inputSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(w2, (1, outputSize * hiddenSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(b1, (1, hiddenSize)));
    thetas.extend(tmp[0]);
    tmp = np.array(np.reshape(b2, (1, outputSize)));
    thetas.extend(tmp[0]);
    
    return np.array(thetas);

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x));

def sigmoid_deri(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)));

# 梯度计算
def costAndGrad(thetas, inputSize, hiddenSize, outputSize, lamb, datas, labels):
    w1 = np.reshape(thetas[0:hiddenSize * inputSize], (hiddenSize, inputSize));  # 3, 960
    w2 = np.reshape(thetas[hiddenSize * inputSize : hiddenSize * inputSize + outputSize * hiddenSize], (outputSize, hiddenSize));  # 4, 3
    b1 = np.reshape(thetas[hiddenSize * inputSize + outputSize * hiddenSize : hiddenSize * inputSize + outputSize * hiddenSize + hiddenSize], (hiddenSize, 1));  # 3, 1
    b2 = np.reshape(thetas[hiddenSize * inputSize + outputSize * hiddenSize + hiddenSize :], (outputSize, 1));  # 4, 1
    
    m = np.shape(datas)[0];
    
    z2 = w1 * datas.T + b1;  # 3, 277
    a2 = sigmoid(z2);  # 3, 277
    z3 = w2 * a2 + b2;  # 4, 277
    a3 = sigmoid(z3);  # 4, 277
    
    Jcost = np.sum(np.square(labels.T - a3)) / (2 * m);
    Jweight = (np.sum(np.square(w1)) + np.sum(np.square(w2))) / 2;
    
    cost = Jcost + lamb * Jweight
    
    d3 = np.multiply(-(labels.T - a3), sigmoid_deri(z3));  # 4, 277
    d2 = np.multiply(w2.T * d3, sigmoid_deri(z2));  # 3, 277
    
    w1grad = d2 * datas / m + lamb * w1;
    w2grad = d3 * a2.T / m + lamb * w2;
    b1grad = np.sum(d2, 1) / m;
    b2grad = np.sum(d3, 1) / m;
    
    grads = [];
    tmp = np.array(np.reshape(w1grad, (1, hiddenSize * inputSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(w2grad, (1, outputSize * hiddenSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(b1grad, (1, hiddenSize)));
    grads.extend(tmp[0]);
    tmp = np.array(np.reshape(b2grad, (1, outputSize)));
    grads.extend(tmp[0]);
    grads = np.array(grads);
    
    return cost, grads;
    
# 梯度检验
def computeNumericalGradient(J, thetas, inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels):
    n = np.shape(thetas)[0];
    numgrad = np.zeros(n);
    epsilon = 1e-4;
    E = np.eye(n);
    
    for i in range(min(n, 100)):
        delta = E[:, i] * epsilon;
        numgrad[i] = (J((thetas + delta), inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels)[0]
                      - J((thetas - delta), inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels)[0]) / (epsilon * 2.0);
    
    return numgrad;

# 保存结果
def saveResult(filepath, dataList):
    fileHandler = open(filepath, "w");
    for data in dataList:
        fileHandler.write(str(data) + "\n");
    fileHandler.close();
 
# 主函数
def main(src_folder, trainFile, resultFile, inputSize, hiddenSize, outputSize):
    trainDatas, trainLabels = loadDataSet(src_folder, trainFile);
    lamb = 0.0001;
    
    checkGrad = False;
    
    thetas = initThetas(inputSize, hiddenSize, outputSize);
    
    # 是否进行梯度检验
    if checkGrad:
        grads = costAndGrad(thetas, inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels)[1];
        numgrad = computeNumericalGradient(costAndGrad, thetas, inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels);
        for i in range(100):
            print grads[i], '    ', numgrad[i];
    
    # 使用LBFGS进行优化
    optimThetas = fmin_l_bfgs_b(func=costAndGrad, x0=thetas,
                  args=(inputSize, hiddenSize, outputSize, lamb, trainDatas, trainLabels),
                  maxiter=300)[0];
        
    # 保存数据
    saveResult(src_folder + resultFile, optimThetas);

# 加载训练结果
def load_thetas(filepath, inputSize, hiddenSize, outputSize):
    thetas = [];
    fr = open(filepath);
    for line in fr.readlines():
        thetas.append(float(line.strip()));
        
    w1 = np.reshape(thetas[0:hiddenSize * inputSize], (hiddenSize, inputSize));  # 3, 960
    w2 = np.reshape(thetas[hiddenSize * inputSize : hiddenSize * inputSize + outputSize * hiddenSize], (outputSize, hiddenSize));  # 4, 3
    b1 = np.reshape(thetas[hiddenSize * inputSize + outputSize * hiddenSize : hiddenSize * inputSize + outputSize * hiddenSize + hiddenSize], (hiddenSize, 1));  # 3, 1
    b2 = np.reshape(thetas[hiddenSize * inputSize + outputSize * hiddenSize + hiddenSize :], (outputSize, 1));  # 4, 1
    
    return w1, w2, b1, b2;

# 画出图像
def display_network(thetas, rowsize, colsize):
    fig = plt.figure();
    m = np.shape(thetas)[0];
    fig_row =  int(np.sqrt(m)) + 1;
    for i in range(m):
        fig.add_subplot(fig_row, fig_row, i + 1);
        img = np.array(thetas[i, :]);
        img = np.reshape(img, (rowsize, colsize));
        plt.imshow(img , cmap='gray');
    plt.show();

# 可视化结果
def visualization_results(thetas_file, inputSize, hiddenSize, outputSize):
    w1 = load_thetas(thetas_file, inputSize, hiddenSize, outputSize)[0];
    display_network(w1, 30, 32);

# 测试正确率
def test(root, test_file, thetas_file, inputSize, hiddenSize, outputSize):
    trainDatas, trainLabels = loadDataSet(root, test_file);
    w1, w2, b1, b2 = load_thetas(thetas_file, inputSize, hiddenSize, outputSize);
    
    pred = sigmoid(w2 * sigmoid(w1 * trainDatas.T + b1) + b2);
    
    m = np.shape(trainLabels)[0];
    error = 0;
    
    for i in range(m):
        tLabel = list(np.array(trainLabels[i])[0]);
        t = tLabel.index(max(tLabel));
        
        oLabel = list(np.array(pred[:, i].T)[0]);
        o = oLabel.index(max(oLabel));
        
        if t != o:
            error += 1;
    
    print "正确率: %.1f" % ((m - error) * 100.0 / m);


root = "/home/hadoop/ProgramDatas/MLStudy/FaceRecognition/";
trainFile = 'all_train.txt';
resultFile = 'thetas.txt';
testFile = 'all_test1.txt';
inputSize = 960;
hiddenSize = 10;
outputSize = 4;

# main(root, trainFile, resultFile, inputSize, hiddenSize, outputSize);
# visualization_results(root + resultFile, inputSize, hiddenSize, outputSize);
test(root, testFile, root + resultFile, inputSize, hiddenSize, outputSize);
