# coding: UTF-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d

# 卷积神经网络结构
class CNNNet:
    
    def __init__(self):
        self.layers = [];
        self.layers.append({'type':'i'});
        self.layers.append({'type':'c', 'outputmaps':6, 'kernelsize':5});
        self.layers.append({'type':'s', 'scale':2});
        self.layers.append({'type':'c', 'outputmaps':12, 'kernelsize':5});
        self.layers.append({'type':'s', 'scale':2});
        
        self.ffw = None;
        self.ffb = None;
        self.fv = None;
        self.o = None;
        self.e = None;
        self.L = None;
        self.od = None;
        self.fvd = None;
        self.rL = None;

# 读取图片数据
def loadImages(filepath, num=-1):
    binfile = open(filepath , 'rb')
    buf = binfile.read()
    datas = [];
    
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    
    takeNum = numImages;
    if num > 0:
        takeNum = num;
    
    for i in range(takeNum):
        im = struct.unpack_from('>784B' , buf, index)
        index += struct.calcsize('>784B')
        im = np.reshape(im, (numRows, numColumns)) / 255.0;
        datas.append(im);
    
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
        num = struct.unpack_from('>1B' , buf, index)
        index += struct.calcsize('>1B')
        la = np.zeros(10);
        la[int(num[0])] = 1;
        labels.append(la);
    
    return labels;

# 画出源数据
def plotDatas(datas):
    fig = plt.figure()
    fig.add_subplot(111);
    plt.imshow(datas)
    plt.show()

# 初始化网络参数
def cnninit(net, x, y):
    inputmaps = 1;
    mapsize = np.array(np.shape(x[0]));
    
    for l in range(len(net.layers)):
        if net.layers[l]['type'] == 'c':
            mapsize = mapsize - net.layers[l]['kernelsize'] + 1;
            fan_out = net.layers[l]['outputmaps'] * np.square(net.layers[l]['kernelsize']);
            fan_in = inputmaps * np.square(net.layers[l]['kernelsize']);
            net.layers[l]['k'] = (np.random.uniform(0, 1, (inputmaps, net.layers[l]['outputmaps'], net.layers[l]['kernelsize'], net.layers[l]['kernelsize'])) - 0.5) * 2 * np.sqrt(6.0 / (fan_in + fan_out));
            net.layers[l]['b'] = np.zeros(net.layers[l]['outputmaps']);
            inputmaps = net.layers[l]['outputmaps'];
        if net.layers[l]['type'] == 's':
            mapsize = mapsize / net.layers[l]['scale'];
            net.layers[l]['b'] = np.zeros(inputmaps);
    
    fvnum = np.prod(mapsize) * inputmaps;
    onum = np.shape(y)[1];
    
    net.ffw = (np.random.uniform(0, 1, (onum, fvnum)) - 0.5) * 2 * np.sqrt(6 / (onum + fvnum));
    net.ffb = np.zeros((onum, 1));
    
    return net;

# 前向传播
def cnnff(net ,x):
    n = len(net.layers);
    net.layers[0]['a'] = [x];
    inputmaps = 1;
    
    for l in range(1, n):
        if net.layers[l]['type'] == 'c':
            net.layers[l]['a'] = [];
            for j in range(net.layers[l]['outputmaps']):
                tm, tr, tc = np.shape(net.layers[l - 1]['a'][0]);
                z = np.zeros((tm, tr - net.layers[l]['kernelsize'] + 1, tc - net.layers[l]['kernelsize'] + 1));
                for i in range(inputmaps):
                    for dl in range(tm):
                        z[dl] += convolve2d(net.layers[l - 1]['a'][i][dl], net.layers[l]['k'][i][j], mode='valid');
                net.layers[l]['a'].append(sigmoid(z + net.layers[l]['b'][j]));
            inputmaps = net.layers[l]['outputmaps'];
        elif net.layers[l]['type'] == 's':
            net.layers[l]['a'] = [];
            for j in range(inputmaps):
                tm, tr, tc = np.shape(net.layers[l-1]['a'][j]);
                z = [];
                for dl in range(tm):
                    z.append(convolve2d(net.layers[l - 1]['a'][j][dl], np.ones((net.layers[l]['scale'], net.layers[l]['scale'])) / np.square(net.layers[l]['scale']), mode='valid'));
                z = np.array(z);
                net.layers[l]['a'].append(z[:, ::net.layers[l]['scale'], ::net.layers[l]['scale']]);
    
    net.fv = [];
    for j in range(len(net.layers[n - 1]['a'])):
        tm, tr, tc = np.shape(net.layers[n - 1]['a'][j]);
        if j == 0:
            net.fv = np.reshape(net.layers[n - 1]['a'][j], (tm, tr * tc));
        else:
            net.fv = np.append(net.fv, np.reshape(net.layers[n - 1]['a'][j], (tm, tr * tc)), axis = 1);
    
    net.o = sigmoid(np.mat(net.fv) * np.mat(net.ffw).T + np.mat(net.ffb).T);
    
    return net;

# sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x));

# 反向传播误差
def cnnbp(net, y):
    n = len(net.layers);
    
    net.e = net.o - y; # error
    net.L = 0.5 * np.sum(np.square(net.e)) / np.shape(net.e)[1];
    
    net.od = np.multiply(net.e, np.multiply(net.o, 1 - net.o));
    net.fvd = np.mat(net.od) * np.mat(net.ffw); # (50, 192)

    if net.layers[n-1]['type'] == 'c':
        net.fvd = np.multiply(net.fvd, np.multiply(net.fv, (1 - net.fv)));
    
    tm, tr, tc = np.shape(net.layers[n - 1]['a'][0]);
    fvnum = tr * tc;
    
    net.layers[n - 1]['d'] = [];
    for j in range(len(net.layers[n - 1]['a'])):
        net.layers[n - 1]['d'].append(np.array(net.fvd[:, j * fvnum : (j + 1) * fvnum]).reshape(tm, tr, tc));
    
    

# 训练网络
def cnntrain(net, x, y, alpha, batchsize, numepochs):
    m = np.shape(x)[0];
    numbatches = m / batchsize;
    net.rL = [];
    for i in range(numepochs):
        print '正在执行迭代: ', i, '/', numepochs;
        start = time.clock();
        dataIndex = range(m);
        np.random.shuffle(dataIndex);
        for b in range(numbatches):
            batch_x = [];
            batch_y = [];
            for index in range(b * batchsize, (b + 1) * batchsize):
                batch_x.append(x[dataIndex[index]]);
                batch_y.append(y[dataIndex[index]]);
            
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
        
        finish = time.clock();
        print '本次执行时间: ', (finish - start), '秒';

def main(folder):
    train_x = loadImages(root + 'train-images.idx3-ubyte');  # (60000, 28, 28)
    train_y = loadLabels(root + 'train-labels.idx1-ubyte');  # (60000)
    
    alpha = 1;  # 学习率
    batchsize = 50;  # 每次挑出一个batchsize的batch来训练
    numepochs = 1;  # 迭代次数
    
    cnn = CNNNet();
    cnn = cnninit(cnn, train_x, train_y);
    cnn = cnntrain(cnn, train_x, train_y, alpha, batchsize, numepochs);
    

root = '/home/hadoop/ProgramDatas/MNISTDataset/';
main(root);
