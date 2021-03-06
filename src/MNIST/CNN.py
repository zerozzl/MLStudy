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
        self.dffw = None;
        self.dffb = None;
        self.rL = None;

    # 导出模型
    def exportModel(self, filepath):
        datas = [];
        n = len(self.layers);
        for l in range(n):
            type = self.layers[l]['type'];
            info = 'layer|l:' + str(l) + '|type:' + str(type);
            if type == 'c':
                info += '|outputmaps:' + str(self.layers[l]['outputmaps']) + '|kernelsize:' + str(self.layers[l]['kernelsize']);
                k = self.layers[l]['k'];
                si, sj, skr, skc = np.shape(k);
                info += '|kshape:' + str(si) + '/' + str(sj) + '/' + str(skr) + '/' + str(skc) + '|k:';
                for i in range(si):
                    for j in range(sj):
                        info += str(i) + '/' + str(j) + '/' + self.mat2str(k[i][j]) + ',';
                info = info[:len(info)-1] + '|b:' + self.mat2str(self.layers[l]['b']);
            elif type == 's':
                info += '|scale:' + str(self.layers[l]['scale']) + '|b:' + self.mat2str(self.layers[l]['b']);
            datas.append(info);
        
        si, sj = np.shape(self.ffw);
        datas.append('ffw|shape:' + str(si) + '/' + str(sj) + '|w:' + self.mat2str(self.ffw));
        si, sj = np.shape(self.ffb);
        datas.append('ffb|shape:' + str(si) + '/' + str(sj) + '|b:' + self.mat2str(self.ffb));
        
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
    
    # 导入模型
    def importModel(self, filepath):
        self.layers = [];
        self.ffw = None;
        self.ffb = None;
        with open(filepath, 'r') as myfile:
            for line in myfile.readlines():
                arr = line.strip().split('|');
                if arr[0] == 'layer':
                    size = len(arr);
                    map = {};
                    for i in range(1, size):
                        tmp = arr[i].split(':');
                        map[tmp[0]] = tmp[1];
                    if map['type'] == 'i':
                        self.layers.append({'type':'i'});
                    elif map['type'] == 'c':
                        tmp = {};
                        tmp['type'] = 'c';
                        tmp['outputmaps'] = int(map['outputmaps']);
                        tmp['kernelsize'] = int(map['kernelsize']);
                        
                        b = map['b'][map['b'].index('[')+1:map['b'].index(']')]
                        b = [float(x) for x in b.split('!')];
                        tmp['b'] = np.array(b);
                        
                        kshape = [int(x) for x in map['kshape'].strip().split('/')];
                        kernels = np.zeros((kshape[0], kshape[1], kshape[2], kshape[3]));
                        
                        klist = map['k'].strip().split(',');
                        for kdata in klist:
                            kdata = kdata.strip().split('/');
                            kvalue = kdata[-1][kdata[-1].index('[')+1:kdata[-1].index(']')];
                            kvalue = [float(x) for x in kvalue.split('!')];
                            kvalue = np.reshape(np.array(kvalue), (kshape[2], kshape[3]));
                            kernels[int(kdata[0])][int(kdata[1])] = kvalue;
                        
                        tmp['k'] = np.array(kernels);
                        self.layers.append(tmp);
                        
                    elif map['type'] == 's':
                        tmp = {};
                        tmp['type'] = 's';
                        tmp['scale'] = int(map['scale']);
                        b = map['b'][map['b'].index('[')+1:map['b'].index(']')]
                        b = [float(x) for x in b.split('!')];
                        tmp['b'] = np.array(b);
                        self.layers.append(tmp);
                elif arr[0] == 'ffw':
                    size = len(arr);
                    map = {};
                    for i in range(1, size):
                        tmp = arr[i].split(':');
                        map[tmp[0]] = tmp[1];
                    wshape = [int(x) for x in map['shape'].strip().split('/')];
                    wdata = map['w'][map['w'].index('[')+1:map['w'].index(']')];
                    wdata = [float(x) for x in wdata.split('!')];
                    wdata = np.reshape(np.array(wdata), (wshape[0], wshape[1]));
                    self.ffw = np.array(wdata);
                elif arr[0] == 'ffb':
                    size = len(arr);
                    map = {};
                    for i in range(1, size):
                        tmp = arr[i].split(':');
                        map[tmp[0]] = tmp[1];
                    bshape = [int(x) for x in map['shape'].strip().split('/')];
                    bdata = map['b'][map['b'].index('[')+1:map['b'].index(']')];
                    bdata = [float(x) for x in bdata.split('!')];
                    bdata = np.reshape(np.array(bdata), (bshape[0], bshape[1]));
                    self.ffb = np.array(bdata);
    
    # 画出隐含层
    def exportHiddenLayerGraph(self, filepath):
        for l in range(len(self.layers)):
            lay = self.layers[l];
            if lay['type'] == 'c':
                kernels = lay['k'];
                si, sj, sr, sc =  np.shape(kernels);
                fig = plt.figure();
                pos = 1;
                for i in range(si):
                    for j in range(sj):
                        fig.add_subplot(si, sj, pos);
                        plt.imshow(kernels[i][j] , cmap='gray');
                        pos += 1;
                plt.savefig(filepath + 'layer_' + str(l) + '.png', format='png');

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

# 画出误差图像
def plotError(datas, filepath=None):
    fig = plt.figure();
    ax = fig.add_subplot(111);
    x = range(len(datas));
    ax.plot(x, datas);
    
    if filepath is not None:
        plt.savefig(filepath, format='png');
    else:
        plt.show();

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
    
    net.ffw = (np.random.uniform(0, 1, (onum, fvnum)) - 0.5) * 2 * np.sqrt(6.0 / (onum + fvnum));
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
    
    for l in range(n - 2, 0, -1):
        if net.layers[l]['type'] == 'c':
            net.layers[l]['d'] = [];
            for j in range(len(net.layers[l]['a'])):
                dtmp = [];
                for dl in range(len(net.layers[l]['a'][j])):
                    dtmp.append(np.multiply(np.multiply(net.layers[l]['a'][j][dl], 1 - net.layers[l]['a'][j][dl]),
                                (kronecker(net.layers[l + 1]['d'][j][dl], np.ones((net.layers[l + 1]['scale'],net.layers[l + 1]['scale'])) / np.square(float(net.layers[l + 1]['scale']))))));
                net.layers[l]['d'].append(dtmp);
        elif net.layers[l]['type'] == 's':
            net.layers[l]['d'] = [];
            for i in range(len(net.layers[l]['a'])):
                tm, tr, tc = np.shape(net.layers[l]['a'][0]);
                z = np.zeros((tm, tr, tc));
                for j in range(len(net.layers[l + 1]['a'])):
                    ztmp = [];
                    for dl in range(len(net.layers[l + 1]['d'][j])):
                        ztmp.append(convolve2d(net.layers[l + 1]['d'][j][dl], np.rot90(net.layers[l + 1]['k'][i][j], 2), mode='full'));
                    z += ztmp;
                net.layers[l]['d'].append(z);
    
    for l in range(1, n):
        if net.layers[l]['type'] == 'c':
            dk = [];
            for i in range(len(net.layers[l - 1]['a'])):
                dkj = [];
                for j in range(len(net.layers[l]['a'])):
                    tdk = [];
                    for dl in range(len(net.layers[l - 1]['a'][i])):
                        tdk.append(convolve2d(np.rot90(net.layers[l - 1]['a'][i][dl], 2), net.layers[l]['d'][j][dl], mode='valid'));
                    dkj.append(np.sum(tdk, 0) / np.shape(tdk)[0]);
                dk.append(dkj);
            net.layers[l]['dk'] = dk;
            
            net.layers[l]['db'] = [];
            for j in range(len(net.layers[l]['a'])):
                net.layers[l]['db'].append(np.sum(net.layers[l]['d'][j]) / np.shape(net.layers[l]['d'][j])[0]);
    
    net.dffw = np.mat(net.od).T * np.mat(net.fv) / np.shape(net.od)[0];
    net.dffb = np.mean(net.od, 0).T;
    return net;

# 更新梯度信息
def cnnapplygrads(net, alpha):
    for l in range(1, len(net.layers)):
        if net.layers[l]['type'] == 'c':
            for j in range(len(net.layers[l]['a'])):
                for i in range(len(net.layers[l - 1]['a'])):
                    net.layers[l]['k'][i][j] = net.layers[l]['k'][i][j] - alpha * net.layers[l]['dk'][i][j];
                net.layers[l]['b'][j] = net.layers[l]['b'][j] - alpha * net.layers[l]['db'][j];
    
    net.ffw = net.ffw - alpha * net.dffw;
    net.ffb = net.ffb - alpha * net.dffb;
    
    return net;

# 训练网络
def cnntrain(net, x, y, alpha, batchsize, numepochs):
    m = np.shape(x)[0];
    numbatches = m / batchsize;
    net.rL = [];
    for i in range(numepochs):
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
            net = cnnapplygrads(net, alpha);
            
            if len(net.rL) == 0:
                net.rL.append(net.L);
            
            net.rL.append(0.99 * net.rL[-1] + 0.01 * net.L);
            print '正在执行迭代: ', i + 1, '/', numepochs, ', 内循环: ', b + 1, '/', numbatches, ', 误差: ', 0.99 * net.rL[-1] + 0.01 * net.L;
        
        finish = time.clock();
        print '本次执行时间: ', (finish - start), '秒';
    
    return net;

# 测试结果
def cnntest(net, x, y):
    net = cnnff(net, x);
    m = np.shape(y)[0];
    
    pred = np.argmax(net.o, 1);
    pred = np.array(pred.reshape(1, pred.shape[0]))[0];
    
    y = np.argmax(y, 1);
    
    bad = [];
    
    for i in range(m):
        if pred[i] != y[i]:
            bad.append(i);
    
    err = float(len(bad)) / m;
    return err, bad;

# 生成模型
def model_create(dataFolder, resultFolder, modelFile, errpngFile=None):
    train_x = loadImages(dataFolder + 'train-images.idx3-ubyte');  # (60000, 28, 28)
    train_y = loadLabels(dataFolder + 'train-labels.idx1-ubyte');  # (60000)
    
    alpha = 1;  # 学习率
    batchsize = 50;  # 每次挑出一个batchsize的batch来训练
    numepochs = 1;  # 迭代次数
    
    cnn = CNNNet();
    cnn = cnninit(cnn, train_x, train_y);
    cnn = cnntrain(cnn, train_x, train_y, alpha, batchsize, numepochs);
    
    cnn.exportModel(modelFile);
    cnn.exportHiddenLayerGraph(resultFolder);
    plotError(cnn.rL, errpngFile);

# 测试模型  
def model_test(dataFolder, modelfile):
    test_x = loadImages(dataFolder + 't10k-images.idx3-ubyte');  # (10000, 28, 28)
    test_y = loadLabels(dataFolder + 't10k-labels.idx1-ubyte');  # (10000)
    
    cnn = CNNNet();
    cnn.importModel(modelfile);
    
    err, bad = cnntest(cnn, test_x, test_y);
    print '正确率: ', (1 - err) * 100, '%';

def main():
    dataFolder = '/home/hadoop/ProgramDatas/MNISTDataset/';
    resultFolder = '/home/hadoop/ProgramDatas/MLStudy/MNIST/';
#     dataFolder = 'E:/TestDatas/MNISTDataset/';
#     resultFolder = 'E:/TestDatas/MNIST/';
    modelFile = resultFolder + 'model.txt';
    errpngFile = resultFolder + 'error.png';
    model_create(dataFolder, resultFolder, modelFile, errpngFile);
    model_test(dataFolder, modelFile);

main();
