# coding: UTF-8
import os
from pyspark import SparkConf
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from scipy.signal import convolve2d

# 卷积神经网络结构
class CNNNet:
    
    class Layer:
        
        def __init__(self, ty, opm=None, ks=None, sc=None):
            self.type = ty;
            self.outputmaps = opm;
            self.kernelsize = ks;
            self.scale = sc;
            self.kernels = None;
            self.w = None;
            self.b = None;
            self.a = None;
            self.d = None;
    
    def __init__(self, inputsize, outputsize):
        self.layers = [];
        self.layers.append(self.Layer(ty='i'));
        self.layers.append(self.Layer(ty='c', opm=6, ks=5));
        self.layers.append(self.Layer(ty='s', sc=2));
        self.layers.append(self.Layer(ty='c', opm=12, ks=5));
        self.layers.append(self.Layer(ty='s', sc=2));
        self.layers.append(self.Layer(ty='r'));
        self.layers.append(self.Layer(ty='o'));
        
        self.inputsize = inputsize;
        self.outputsize = outputsize;
        
        self.init_params();
    
    # 初始化网络参数
    def init_params(self):
        inputmaps = self.inputsize[0];
        mapsize = self.inputsize[1:3];
        for l in range(len(self.layers)):
            if self.layers[l].type == 'c':
                mapsize = [mapsize[0] - self.layers[l].kernelsize + 1, mapsize[1] - self.layers[l].kernelsize + 1];
                fan_out = self.layers[l].outputmaps * self.layers[l].kernelsize * self.layers[l].kernelsize;
                fan_in = inputmaps * self.layers[l].kernelsize * self.layers[l].kernelsize;
                self.layers[l].kernels = (np.random.uniform(0, 1, (inputmaps, self.layers[l].outputmaps, self.layers[l].kernelsize, self.layers[l].kernelsize)) - 0.5) * 2 * np.sqrt(6.0 / (fan_in + fan_out));
                self.layers[l].b = np.zeros(self.layers[l].outputmaps);
                inputmaps = self.layers[l].outputmaps;
            elif self.layers[l].type == 's':
                mapsize = [mapsize[0] / self.layers[l].scale, mapsize[1] / self.layers[l].scale];
                self.layers[l].b = np.zeros(inputmaps);
            elif self.layers[l].type == 'r':
                fvnum = mapsize[0] * mapsize[1] * inputmaps;
                self.layers[l].w = (np.random.uniform(0, 1, (self.outputsize, fvnum)) - 0.5) * 2 * np.sqrt(6.0 / (self.outputsize + fvnum));
                self.layers[l].b = np.zeros((self.outputsize, 1));
    
    # 导出模型
    def exportModel(self, filepath):
        datas = [];
        n = len(self.layers);
        for l in range(n):
            info = 'layer|l:' + str(l) + '|type:' + self.layers[l].type;
            if self.layers[l].type == 'c':
                info += '|outputmaps:' + str(self.layers[l].outputmaps) + '|kernelsize:' + str(self.layers[l].kernelsize);
                si, sj, skr, skc = np.shape(self.layers[l].kernels);
                info += '|kshape:' + str(si) + '/' + str(sj) + '/' + str(skr) + '/' + str(skc) + '|k:';
                for i in range(si):
                    for j in range(sj):
                        info += str(i) + '/' + str(j) + '/' + self.mat2str(self.layers[l].kernels[i][j]) + ',';
                info = info[:len(info)-1] + '|b:' + self.mat2str(self.layers[l].b);
            elif self.layers[l].type == 's':
                info += '|scale:' + str(self.layers[l].scale) + '|b:' + self.mat2str(self.layers[l].b);
            elif self.layers[l].type == 'r':
                si, sj = np.shape(self.layers[l].w);
                info += '|wshape:' + str(si) + '/' + str(sj) + '|w:' + self.mat2str(self.layers[l].w);
                si, sj = np.shape(self.layers[l].b);
                info += '|bshape:' + str(si) + '/' + str(sj) + '|b:' + self.mat2str(self.layers[l].b);
            datas.append(info);
        
        with open(filepath, 'w') as myfile:
            for line in datas:
                myfile.write(line);
                myfile.write('\n');
    
    # 矩阵数据转换成字符串保存
    def mat2str(self, mat):
        s = '[';
        mat = np.mat(mat);
        m, n = np.shape(mat);
        for i in range(m):
            for j in range(n):
                s += str(mat[i, j]) + '!';
        s = s[:len(s)-1] + ']';
        return s;

# 训练网络
def cnntrain(net, datas, alpha, numepochs):
#     m = datas.count();
#     numbatches = m / batchsize;
#     net.rL = [];
    for i in range(numepochs):
        start = clock();
        
        x, y = datas.take(1)[0];
        net, y = feedforward(net, (x, y));
        net, lost = backpropagation(net, y);
        
#             pred = datas.map(lambda x : self.feedforward(x));
#             pred = pred.collect();
#             print np.shape(pred);
#             print pred[0];


#             dataIndex = range(m);
#             np.random.shuffle(dataIndex);
#             for b in range(numbatches):
#                 batch_x = [];
#                 batch_y = [];
#                 for index in range(b * batchsize, (b + 1) * batchsize):
#                     batch_x.append(x[dataIndex[index]]);
#                     batch_y.append(y[dataIndex[index]]);
          
#             net = cnnff(net, batch_x);
#             net = cnnbp(net, batch_y);
#             net = cnnapplygrads(net, alpha);
          
#                 if len(net.rL) == 0:
#                     net.rL.append(net.L);
#               
#                 net.rL.append(0.99 * net.rL[-1] + 0.01 * net.L);
#                 print '正在执行迭代: ', i + 1, '/', numepochs, ', 内循环: ', b + 1, '/', numbatches, ', 误差: ', 0.99 * net.rL[-1] + 0.01 * net.L;
      
        finish = clock();
        print '本次执行时间: ', (finish - start), '秒';
    
# 前向传播
def feedforward(net, data):
    n = len(net.layers);
    net.layers[0].a = [np.reshape(data[0], (net.inputsize[1], net.inputsize[2]))];
    inputmaps = net.inputsize[0];
    
    for l in range(1, n):
        net.layers[l].a = [];
        if net.layers[l].type == 'c':
            for j in range(net.layers[l].outputmaps):
                rs, cs = np.shape(net.layers[l - 1].a[0]);
                z = np.zeros((rs - net.layers[l].kernelsize + 1, cs - net.layers[l].kernelsize + 1));
                for i in range(inputmaps):
                    z += convolve2d(net.layers[l - 1].a[i], net.layers[l].kernels[i][j], mode='valid');
                net.layers[l].a.append(sigmoid(z + net.layers[l].b[j]));
            inputmaps = net.layers[l].outputmaps;
        elif net.layers[l].type == 's':
            for j in range(inputmaps):
                z = convolve2d(net.layers[l - 1].a[j], np.ones((net.layers[l].scale, net.layers[l].scale)) / np.square(net.layers[l].scale), mode='valid');
                net.layers[l].a.append(z[::net.layers[l].scale, ::net.layers[l].scale]);
        elif net.layers[l].type == 'r':
            fv = [];
            for j in range(len(net.layers[l - 1].a)):
                rs, cs = np.shape(net.layers[l - 1].a[j]);
                if j == 0:
                    fv = np.reshape(net.layers[l - 1].a[j], (rs * cs));
                else:
                    fv = np.append(fv, np.reshape(net.layers[l - 1].a[j], (rs * cs)), axis = 1);
            net.layers[l].a = fv;
        elif net.layers[l].type == 'o':
            net.layers[l].a = sigmoid(np.mat(net.layers[l - 1].a) * np.mat(net.layers[l - 1].w).T + np.mat(net.layers[l - 1].b).T);
    return (net, data[1]);
    
# 反向传播误差
def backpropagation(net, y):
    n = len(net.layers);
    out = np.zeros(net.outputsize);
    out[y] = 1;
    err = net.layers[n-1].a - out;
    lost = 0.5 * np.sum(np.square(err));
# 
#     net.od = np.multiply(net.e, np.multiply(net.o, 1 - net.o));
#     net.fvd = np.mat(net.od) * np.mat(net.ffw); # (50, 192)
# 
#     if net.layers[n-1]['type'] == 'c':
#         net.fvd = np.multiply(net.fvd, np.multiply(net.fv, (1 - net.fv)));
# 
#     tm, tr, tc = np.shape(net.layers[n - 1]['a'][0]);
#     fvnum = tr * tc;
# 
#     net.layers[n - 1]['d'] = [];
#     for j in range(len(net.layers[n - 1]['a'])):
#         net.layers[n - 1]['d'].append(np.array(net.fvd[:, j * fvnum : (j + 1) * fvnum]).reshape(tm, tr, tc));
# 
#     for l in range(n - 2, 0, -1):
#         if net.layers[l]['type'] == 'c':
#             net.layers[l]['d'] = [];
#             for j in range(len(net.layers[l]['a'])):
#                 dtmp = [];
#                 for dl in range(len(net.layers[l]['a'][j])):
#                     dtmp.append(np.multiply(np.multiply(net.layers[l]['a'][j][dl], 1 - net.layers[l]['a'][j][dl]),
#                             (kronecker(net.layers[l + 1]['d'][j][dl], np.ones((net.layers[l + 1]['scale'],net.layers[l + 1]['scale'])) / np.square(float(net.layers[l + 1]['scale']))))));
#                 net.layers[l]['d'].append(dtmp);
#         elif net.layers[l]['type'] == 's':
#             net.layers[l]['d'] = [];
#             for i in range(len(net.layers[l]['a'])):
#                 tm, tr, tc = np.shape(net.layers[l]['a'][0]);
#                 z = np.zeros((tm, tr, tc));
#                 for j in range(len(net.layers[l + 1]['a'])):
#                     ztmp = [];
#                     for dl in range(len(net.layers[l + 1]['d'][j])):
#                         ztmp.append(convolve2d(net.layers[l + 1]['d'][j][dl], np.rot90(net.layers[l + 1]['k'][i][j], 2), mode='full'));
#                     z += ztmp;
#                 net.layers[l]['d'].append(z);
# 
#     for l in range(1, n):
#         if net.layers[l]['type'] == 'c':
#             dk = [];
#             for i in range(len(net.layers[l - 1]['a'])):
#                 dkj = [];
#                 for j in range(len(net.layers[l]['a'])):
#                     tdk = [];
#                     for dl in range(len(net.layers[l - 1]['a'][i])):
#                         tdk.append(convolve2d(np.rot90(net.layers[l - 1]['a'][i][dl], 2), net.layers[l]['d'][j][dl], mode='valid'));
#                     dkj.append(np.sum(tdk, 0) / np.shape(tdk)[0]);
#                 dk.append(dkj);
#             net.layers[l]['dk'] = dk;
#         
#             net.layers[l]['db'] = [];
#             for j in range(len(net.layers[l]['a'])):
#                 net.layers[l]['db'].append(np.sum(net.layers[l]['d'][j]) / np.shape(net.layers[l]['d'][j])[0]);
# 
#     net.dffw = np.mat(net.od).T * np.mat(net.fv) / np.shape(net.od)[0];
#     net.dffb = np.mean(net.od, 0).T;
#     return net;

def loadSparkData(sc, filepath):
    datas = sc.textFile(filepath, 60).map(
                    lambda line : line.split(",")).map(
                    lambda line : [int(x) for x in line]).map(
                    lambda line : (line[:-1], line[-1]));
    return datas;

# sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x));

# 克罗内克积
def kronecker(A, B):
    m, n = np.shape(A);
    p, q = np.shape(B);
    K = np.zeros((m * p, n * q));
    for i in range(m):
        tmpR = [];
        for j in range(n):
            tmp = A[i, j] * B;
            if j == 0:
                tmpR = tmp;
            else:
                tmpR = np.column_stack((tmpR, tmp));
        if i == 0:
            K = tmpR;
        else:
            K = np.row_stack((K, tmpR));
    return K;

# 画出源数据
def plotDatas(datas):
    fig = plt.figure();
    fig.add_subplot(111);
    plt.imshow(datas);
    plt.show();

def main():
    inputsize = [1, 28, 28];
    outputsize = 10;
    alpha = 1;
    numepochs = 1;
    
    os.environ['SPARK_HOME'] = '/apps/spark/spark-1.4.1-bin-hadoop2.6';
    conf = SparkConf().setAppName('MNIST CNN').setMaster('spark://localhost:7077');
    sc = SparkContext(conf=conf);

    trainData = loadSparkData(sc, 'hdfs://localhost:9000/datas/mnist/test-datas.csv');
    trainData.cache();
    
    cnn = CNNNet(inputsize, outputsize);
    cnntrain(cnn, trainData, alpha, numepochs);
    
    sc.stop();

main();