# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from time import clock
from scipy.signal import convolve2d

# 卷积神经网络结构
class CNNNet:
    
    # 卷积层
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
            self.dk = None;
            self.db = None;
            self.dw = None;
    
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
        self.x = None;
        self.y = None;
        self.lost = None;
        
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
    
    # 导入数据
    def import_data(self, x, y):
        self.x = x;
        self.y = y;
    
    # 前向传播
    def feedforward(self, x=None):
        n = len(self.layers);
        if x is None:
            x = self.x;
        self.layers[0].a = [np.reshape(np.array(x) / 255.0, (self.inputsize[1], self.inputsize[2]))];
        inputmaps = self.inputsize[0];
        
        for l in range(1, n):
            self.layers[l].a = [];
            if self.layers[l].type == 'c':
                for j in range(self.layers[l].outputmaps):
                    rs, cs = np.shape(self.layers[l - 1].a[0]);
                    z = np.zeros((rs - self.layers[l].kernelsize + 1, cs - self.layers[l].kernelsize + 1));
                    for i in range(inputmaps):
                        z += convolve2d(self.layers[l - 1].a[i], self.layers[l].kernels[i][j], mode='valid');
                    self.layers[l].a.append(sigmoid(z + self.layers[l].b[j]));
                inputmaps = self.layers[l].outputmaps;
            elif self.layers[l].type == 's':
                for j in range(inputmaps):
                    z = convolve2d(self.layers[l - 1].a[j], np.ones((self.layers[l].scale, self.layers[l].scale)) / np.square(self.layers[l].scale), mode='valid');
                    self.layers[l].a.append(z[::self.layers[l].scale, ::self.layers[l].scale]);
            elif self.layers[l].type == 'r':
                fv = [];
                for j in range(len(self.layers[l - 1].a)):
                    rs, cs = np.shape(self.layers[l - 1].a[j]);
                    if j == 0:
                        fv = np.reshape(self.layers[l - 1].a[j], (rs * cs));
                    else:
                        fv = np.append(fv, np.reshape(self.layers[l - 1].a[j], (rs * cs)), axis = 1);
                self.layers[l].a = fv;
            elif self.layers[l].type == 'o':
                self.layers[l].a = sigmoid(np.mat(self.layers[l - 1].a) * np.mat(self.layers[l - 1].w).T + np.mat(self.layers[l - 1].b).T);
        
        return self.layers[n - 1].a;
    
    # 反向传播误差
    def backpropagation(self, out=None, ynum=None, exportGradAndCost=False):
        n = len(self.layers);
        y = np.zeros(self.outputsize);
        
        print ynum;
        
        if ynum is not None:
            y[ynum] = 1;
        else:
            y[self.y] = 1;
        if out is None:
            out = self.layers[n-1].a;
        err = out - y;
        self.lost = 0.5 * np.sum(np.square(err));
        
        for l in range(n-1, 0, -1):
            if self.layers[l].type == 'o':
                self.layers[l].d = np.multiply(err, np.multiply(out, 1 - out));
            elif self.layers[l].type == 'r':
                self.layers[l].d = np.mat(self.layers[l + 1].d) * np.mat(self.layers[l].w);
                if self.layers[l].type == 'c':
                    self.layers[l].d = np.multiply(self.layers[l].d, np.multiply(self.layers[l].a, 1 - self.layers[l].a));
                
                rs, cs = np.shape(self.layers[l-1].a[0]);
                fvnum = rs * cs;
                self.layers[l - 1].d = [];
                for j in range(len(self.layers[l - 1].a)):
                    self.layers[l - 1].d.append(np.array(self.layers[l].d[:,j * fvnum : (j + 1) * fvnum]).reshape(rs, cs));
            elif self.layers[l].type == 's':
                if self.layers[l + 1].kernels is not None:
                    self.layers[l].d = [];
                    for i in range(len(self.layers[l].a)):
                        rs, cs = np.shape(self.layers[l].a[0]);
                        z = np.zeros((rs, cs));
                        for j in range(len(self.layers[l + 1].a)):
                            z += convolve2d(self.layers[l + 1].d[j], np.rot90(self.layers[l + 1].kernels[i][j], 2), mode='full');
                        self.layers[l].d.append(z);
            elif self.layers[l].type == 'c':
                self.layers[l].d = [];
                for j in range(len(self.layers[l].a)):
                    self.layers[l].d.append(np.multiply(np.multiply(self.layers[l].a[j], 1 - self.layers[l].a[j]),
                            (kronecker(self.layers[l + 1].d[j], np.ones((self.layers[l + 1].scale, self.layers[l + 1].scale)) / np.square(float(self.layers[l + 1].scale))))));
     
        for l in range(1, n):
            if self.layers[l].type == 'c':
                self.layers[l].dk = [];
                for i in range(len(self.layers[l - 1].a)):
                    dkj = [];
                    for j in range(len(self.layers[l].a)):
                        dkj.append(convolve2d(np.rot90(self.layers[l - 1].a[i], 2), self.layers[l].d[j], mode='valid'));
                    self.layers[l].dk.append(dkj);
                    
                self.layers[l].db = [];
                for j in range(len(self.layers[l].a)):
                    self.layers[l].db.append(np.sum(self.layers[l].d[j]));
            elif self.layers[l].type == 'r':
                self.layers[l].dw = np.mat(self.layers[l + 1].d).T * np.mat(self.layers[l].a);
                self.layers[l].db = self.layers[l + 1].d.T;
        
        if exportGradAndCost:
            grads = [];
            for l in range(1, n):
                if self.layers[l].type == 'c':
                    dmap = {};
                    dmap['layer'] = l;
                    dmap['type'] = 'c';
                    dmap['dk'] = self.layers[l].dk;
                    dmap['db'] = self.layers[l].db;
                    grads.append(dmap);
                elif self.layers[l].type == 'r':
                    dmap = {};
                    dmap['layer'] = l;
                    dmap['type'] = 'r';
                    dmap['dw'] = self.layers[l].dw;
                    dmap['db'] = self.layers[l].db;
                    grads.append(dmap);
            return grads, self.lost;
    
    # 累加梯度
    def accumulationGrads(self, gradAndCost1, gradAndCost2):
        grads = [];
        grad1Arr = gradAndCost1[0];
        grad2Arr = gradAndCost2[0];
        for i in range(len(grad1Arr)):
            grad1 = grad1Arr[i];
            grad2 = grad2Arr[i];
            dmap = {};
            if grad1['layer'] == grad2['layer']:
                dmap['layer'] = grad1['layer'];
                if grad1['type'] == 'c':
                    dmap['type'] = 'c';
                    dmap['dk'] = np.array(grad1['dk']) + np.array(grad2['dk']);
                    dmap['db'] = np.array(grad1['db']) + np.array(grad2['db']);
                elif grad1['type'] == 'r':
                    dmap['type'] = 'r';
                    dmap['dw'] = np.array(grad1['dw']) + np.array(grad2['dw']);
                    dmap['db'] = np.array(grad1['db']) + np.array(grad2['db']);
                grads.append(dmap);
            else:
                raise Exception('Layer is not match');
        cost = gradAndCost1[1] + gradAndCost2[1];
        return (grads, cost);
    
    # 更新批量梯度信息
    def applyBatchGrads(self, grads, batch, alpha):
        for i in range(len(grads)):
            l = grads[i]['layer']
            if self.layers[l].type == grads[i]['type']:
                if grads[i]['type'] == 'c':
                    self.layers[l].kernels = np.array(self.layers[l].kernels) - alpha * (np.array(grads[i]['dk']) / batch);
                    self.layers[l].b = self.layers[l].b - alpha * (np.array(grads[i]['db']) / batch);
                elif grads[i]['type'] == 'r':
                    self.layers[l].w = self.layers[l].w - alpha * (np.array(grads[i]['dw']) / batch);
                    self.layers[l].b = self.layers[l].b - alpha * (np.array(grads[i]['db']) / batch);
            else:
                raise Exception('Layer is not match');
    
    # 更新梯度信息
    def applyGrads(self, alpha):
        for l in range(0, len(self.layers)):
            if self.layers[l].type == 'c':
                si, sj, sr, sc = np.shape(self.layers[l].dk);
                for i in range(si):
                    for j in range(sj):
                        self.layers[l].kernels[i][j] = self.layers[l].kernels[i][j] - alpha * self.layers[l].dk[i][j];
                    self.layers[l].b[j] = self.layers[l].b[j] - alpha * self.layers[l].db[j];
            elif self.layers[l].type == 'r':
                self.layers[l].w = self.layers[l].w - alpha * self.layers[l].dw;
                self.layers[l].b = self.layers[l].b - alpha * self.layers[l].db;
        
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
    
    # 导入模型
    def importModel(self, filepath):
        self.layers = [];
        with open(filepath, 'r') as myfile:
            for line in myfile.readlines():
                arr = line.strip().split('|');
                size = len(arr);
                map = {};
                for i in range(1, size):
                    tmp = arr[i].split(':');
                    map[tmp[0]] = tmp[1];
                if map['type'] == 'i':
                    self.layers.append(self.Layer(ty='i'));
                elif map['type'] == 'o':
                    self.layers.append(self.Layer(ty='o'));
                elif map['type'] == 'c':
                    layer = self.Layer(ty='c', opm=int(map['outputmaps']), ks=int(map['kernelsize']));
                    layer.b = np.array([float(x) for x in map['b'][map['b'].index('[')+1:map['b'].index(']')].split('!')]);
                    kshape = [int(x) for x in map['kshape'].strip().split('/')];
                    kernels = np.zeros((kshape[0], kshape[1], kshape[2], kshape[3]));
                    klist = map['k'].strip().split(',');
                    for kdata in klist:
                        kdata = kdata.strip().split('/');
                        kvalue = kdata[-1][kdata[-1].index('[')+1:kdata[-1].index(']')];
                        kvalue = [float(x) for x in kvalue.split('!')];
                        kvalue = np.reshape(np.array(kvalue), (kshape[2], kshape[3]));
                        kernels[int(kdata[0])][int(kdata[1])] = kvalue;
                    layer.kernels = np.array(kernels);
                    self.layers.append(layer);
                elif map['type'] == 's':
                    layer = self.Layer(ty='s', sc=int(map['scale']));
                    layer.b = np.array([float(x) for x in map['b'][map['b'].index('[')+1:map['b'].index(']')].split('!')]);
                    self.layers.append(layer);
                elif map['type'] == 'r':
                    layer = self.Layer(ty='r');
                    wshape = [int(x) for x in map['wshape'].strip().split('/')];
                    wdata = [float(x) for x in map['w'][map['w'].index('[')+1:map['w'].index(']')].split('!')];
                    layer.w = np.array(np.reshape(np.array(wdata), (wshape[0], wshape[1])));
                    
                    bshape = [int(x) for x in map['bshape'].strip().split('/')];
                    bdata = [float(x) for x in map['b'][map['b'].index('[')+1:map['b'].index(']')].split('!')];
                    layer.b = np.array(np.reshape(np.array(bdata), (bshape[0], bshape[1])));
                    self.layers.append(layer);

# stochastic
def cnntrain_stoc(datas, inputsize, outputsize, alpha, numepochs):
    rl = [];
    m = datas.count();
    for i in range(numepochs):
        start = clock();
        
        cnn = CNNNet(inputsize, outputsize);
        datas = datas.collect();
        
        for j in range(m):
            cnn.import_data(datas[j][0], datas[j][1]);
            cnn.feedforward();
            cnn.backpropagation();
            cnn.applyGrads(alpha);
        
            if len(rl) == 0:
                rl.append(cnn.lost);
            rl.append(0.99 * rl[-1] + 0.01 * cnn.lost);
            print '正在执行迭代: ', i + 1, '/', numepochs, ', 内循环: ', j + 1, '/', m, ', 误差: ', 0.99 * rl[-1] + 0.01 * cnn.lost;
        
        finish = clock();
        print '本次执行时间: ', (finish - start), '秒';
    plotError(rl);
    return cnn, rl;

# mini batch
def cnntrain_minibatch(datas, inputsize, outputsize, alpha, numepochs, batchsize):
    rl = [];
    m = datas.count();
    numbatches = m / batchsize;
    cnn = CNNNet(inputsize, outputsize);
    for i in range(numepochs):
        start = clock();
        dataSplits = datas.randomSplit(np.ones(numbatches), i);
        for j in range(len(dataSplits)):
            dsp = dataSplits[j];
            bm = dsp.count();
            gradsAndCost = dsp.map(lambda line : (cnn.feedforward(line[0]), line[1])).map(
                                    lambda line : cnn.backpropagation(line[0], line[1], True)).reduce(
                                    lambda gAndCa, gAndCb : cnn.accumulationGrads(gAndCa, gAndCb));
            
            cnn.applyBatchGrads(gradsAndCost[0], bm, alpha);
            
            if len(rl) == 0:
                rl.append(gradsAndCost[1] / bm);
            rl.append(0.99 * rl[-1] + 0.01 * gradsAndCost[1] / bm);
            print '正在执行迭代: ', i + 1, '/', numepochs, ', 内循环: ', j + 1, '/',len(dataSplits), ', 误差: ', 0.99 * rl[-1] + 0.01 * gradsAndCost[1] / bm;
        finish = clock();
        print '本次执行时间: ', (finish - start), '秒';
    return cnn, rl;

# 测试结果
def cnntest(cnn, testData):
    m = testData.count();
    err = testData.map(lambda line : (cnn.feedforward(line[0]), line[1])).map(
                            lambda line : (np.sum(np.argmax(line[0], 1)), line[1])).map(
                            lambda line : 0 if line[0] == line[1] else 1).filter(
                            lambda line : line == 1).count();
    err = float(err) / m;
    return err;

# 读取spark数据
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

# 读取图片数据及
def loadDataSet(folder, filename):
    datas = [];
    fr = open(folder + filename);
    for line in fr.readlines():
        orien = line.strip().split('_')[1];
        if orien == 'left':
            labv = 0;
        elif orien == 'straight':
            labv = 1;
        elif orien == 'right':
            labv = 2;
        elif orien == 'up':
            labv = 3;
        else:
            raise Exception('orientation not recognize!');
        
        print np.shape(read_pgm(folder + line.strip())[0]);
#         datas.append(read_pgm(folder + line.strip())[0]);
#     datas = np.mat(datas) / 255.0;
#     return datas;

# 生成模型
def model_create(srcFolder, trainFile, inputsize, outputsize, alpha, numepochs):
    trainData = loadDataSet(srcFolder, trainFile);
#     cnn, rl = cnntrain_stoc(trainData, inputsize, outputsize, alpha, numepochs);
#     cnn, rl = cnntrain_minibatch(trainData, inputsize, outputsize, alpha, numepochs, batchsize);
#     cnn.exportModel(ModelFile);
#     plotError(rl, ErrPngFile);
    

# # 测试模型  
# def model_test(testData, ModelFile, inputsize, outputsize):
#     cnn = CNNNet(inputsize, outputsize);
#     cnn.importModel(ModelFile);
#     err = cnntest(cnn, testData);
#     print '正确率: ', (1 - err) * 100, '%';

def main():
    root = "E:/TestDatas/FaceOrientation/";
    trainFile = 'all_train.txt';
    testFile = 'all_test1.txt';
#     resultFile = 'thetas.txt';
     
    inputsize = [1, 30, 32];
    outputsize = 4;
    alpha = 1;
    numepochs = 1;
#     batchsize = 100;
#     
    model_create(root, trainFile, inputsize, outputsize, alpha, numepochs);
#     model_test(SparkMaster, TestData, ModelFile, inputsize, outputsize);
 
main();


