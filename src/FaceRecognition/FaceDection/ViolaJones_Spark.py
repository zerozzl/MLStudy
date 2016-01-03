# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import operator
from pyspark import SparkConf
from pyspark import SparkContext

# 特征模板
class FeaTemplate:
    
    def __init__(self, t, ang, w, h):
        self.type = t;
        self.ang = ang;
        self.w = w;
        self.h = h;

# 积分图
class IntegralImage:
    
    def __init__(self, image, label, w):
        self.orig = np.array(image);
        self.sat, self.rsat = self.computeIntegrogram(self.orig);
        self.label = label;
        self.weight = w;

    # 计算积分图
    def computeIntegrogram(self, img):
        m, n = np.shape(img);
        sat = np.zeros((m + 1, n + 1));
        rsat = np.zeros((m + 1, n + 2));
        for y in range(m):
            for x in range(n):
                sat[y + 1, x + 1] = sat[y, x + 1] + sat[y + 1, x] - sat[y, x] + img[y, x];
                if y == 0:
                    rsat[y + 1, x + 1] = img[y, x];
                else:
                    rsat[y + 1, x + 1] = rsat[y, x] + rsat[y, x + 2] - rsat[y - 1, x + 1] + img[y, x] + img[y - 1, x];
        return sat, rsat;
    
    # 获取举行图像积分和
    def getAreaSum(self, x, y, w, h, angle=0):
        if angle == 0:
            return self.sat[y, x] + self.sat[y + h, x + w] - self.sat[y, x + w] - self.sat[y + h, x];
        elif angle == 45:
            return self.rsat[y, x + 1] + self.rsat[y + w + h, x - h + w + 1] - self.rsat[y + h, x - h + 1] - self.rsat[y + w, x + w + 1];

# Haar-Like特征
class HaarLikeFeature:
    
    def __init__(self, fea_type, pos, w, h):
        self.type = fea_type;
        self.pos = pos;
        self.w = w;
        self.h = h;
        self.theta = 0.0;
        self.p = 1.0;
        self.err = 0.0;
    
    # 投票
    def getVote(self, intImg):
        score = self.getEigenvalue(intImg);
        if self.p * score <= self.p * self.theta:
            return 1;
        else:
            return 0;
    
    def toString(self):
        return 'type:' + self.type + '|pos:' + str(self.pos) + '|w:' + str(self.w) + '|h:' + str(self.h) + '|theta:' + str(self.theta) + '|p:' + str(self.p);

def getEigenvalue(fType, fPos, w, h, intImg):
    eigenvalue = 0;
    if fType == '1a':
        part = w / 2;
        negative = intImg.getAreaSum(fPos[1], fPos[0], part, h);
        positive = intImg.getAreaSum(fPos[1] + part, fPos[0], part, h);
        eigenvalue = positive - negative;
    elif fType == '1b':
        part = h / 2;
        negative = intImg.getAreaSum(fPos[1], fPos[0], w, part);
        positive = intImg.getAreaSum(fPos[1], fPos[0] + part, w, part);
        eigenvalue = positive - negative;
    elif fType == '1c':
        part = w / 2;
        negative = intImg.getAreaSum(fPos[1] + h, fPos[0], part, h, 45);
        positive = intImg.getAreaSum(fPos[1] + h + part, fPos[0] + part, part, h, 45);
        eigenvalue = positive - negative;
    elif fType == '1d':
        part = h / 2;
        negative = intImg.getAreaSum(fPos[1] + h, fPos[0], w, part, 45);
        positive = intImg.getAreaSum(fPos[1] + part, fPos[0] + part, w, part, 45);
        eigenvalue = positive - negative;
    elif fType == '2a':
        part = w / 3;
        negative1 = intImg.getAreaSum(fPos[1], fPos[0], part, h);
        positive = intImg.getAreaSum(fPos[1] + part, fPos[0], part, h);
        negative2 = intImg.getAreaSum(fPos[1] + 2 * part, fPos[0], part, h);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2b':
        part = w / 4;
        negative1 = intImg.getAreaSum(fPos[1], fPos[0], part, h);
        positive = intImg.getAreaSum(fPos[1] + part, fPos[0], 2 * part, h);
        negative2 = intImg.getAreaSum(fPos[1] + 3 * part, fPos[0], part, h);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2c':
        part = h / 3;
        negative1 = intImg.getAreaSum(fPos[1], fPos[0], w, part);
        positive = intImg.getAreaSum(fPos[1], fPos[0] + part, w, part);
        negative2 = intImg.getAreaSum(fPos[1], fPos[0] + 2 * part, w, part);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2d':
        part = h / 4;
        negative1 = intImg.getAreaSum(fPos[1], fPos[0], w, part);
        positive = intImg.getAreaSum(fPos[1], fPos[0] + part, w, 2 * part);
        negative2 = intImg.getAreaSum(fPos[1], fPos[0] + 3 * part, w, part);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2e':
        part = w / 3;
        negative1 = intImg.getAreaSum(fPos[1] + h, fPos[0], part, h, 45);
        positive = intImg.getAreaSum(fPos[1] + h + part, fPos[0] + part, part, h, 45);
        negative2 = intImg.getAreaSum(fPos[1] + h + 2 * part, fPos[0] + 2 * part, part, h, 45);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2f':
        part = w / 4;
        negative1 = intImg.getAreaSum(fPos[1] + h, fPos[0], part, h, 45);
        positive = intImg.getAreaSum(fPos[1] + h + part, fPos[0] + part, 2 * part, h, 45);
        negative2 = intImg.getAreaSum(fPos[1] + h + 3 * part, fPos[0] + 3 * part, part, h, 45);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2g':
        part = h / 3;
        negative1 = intImg.getAreaSum(fPos[1] + h, fPos[0], w, part, 45);
        positive = intImg.getAreaSum(fPos[1] + 2 * part, fPos[0] + part, w, part, 45);
        negative2 = intImg.getAreaSum(fPos[1] + part, fPos[0] + 2 * part, w, part, 45);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '2h':
        part = h / 4;
        negative1 = intImg.getAreaSum(fPos[1] + h, fPos[0], w, part, 45);
        positive = intImg.getAreaSum(fPos[1] + 3 * part, fPos[0] + part, w, 2 * part, 45);
        negative2 = intImg.getAreaSum(fPos[1] + part, fPos[0] + 3 * part, w, part, 45);
        eigenvalue = positive - negative1 - negative2;
    elif fType == '3a':
        partw = w / 3;
        parth = h / 3;
        whole = intImg.getAreaSum(fPos[1], fPos[0], w, h);
        positive = intImg.getAreaSum(fPos[1] + partw, fPos[0] + parth, partw, parth);
        eigenvalue = 2 * positive - whole;
    elif fType == '3b':
        partw = w / 3;
        parth = h / 3;
        whole = intImg.getAreaSum(fPos[1] + h, fPos[0], w, h, 45);
        positive = intImg.getAreaSum(fPos[1] + partw + 2 * parth, fPos[0] + partw + parth, partw, parth, 45);
        eigenvalue = 2 * positive - whole;
    
    return eigenvalue;
    
# 初始化特征模板
def initFeaTemplates():
    # 具体特征模板形状，请参考论文: An Extended Set of Haar-like Features for Rapid Object Detection
    templates = [];
    templates.append(FeaTemplate('1a', 0, 2, 1));
    templates.append(FeaTemplate('1b', 0, 1, 2));
    templates.append(FeaTemplate('1c', 45, 2, 1));
    templates.append(FeaTemplate('1d', 45, 1, 2));
    templates.append(FeaTemplate('2a', 0, 3, 1));
    templates.append(FeaTemplate('2b', 0, 4, 1));
    templates.append(FeaTemplate('2c', 0, 1, 3));
    templates.append(FeaTemplate('2d', 0, 1, 4));
    templates.append(FeaTemplate('2e', 45, 3, 1));
    templates.append(FeaTemplate('2f', 45, 4, 1));
    templates.append(FeaTemplate('2g', 45, 1, 3));
    templates.append(FeaTemplate('2h', 45, 1, 4));
    templates.append(FeaTemplate('3a', 0, 3, 3));
    templates.append(FeaTemplate('3b', 45, 3, 3));
    return templates;

# 初始化特征
def initFeatures(W, H, templates):
    print 'initing features...'
    features = [];
    for feaTemp in templates:
        if feaTemp.ang == 0:
            wblock = int(W / feaTemp.w);
            hblock = int(H / feaTemp.h);
            for i in range(wblock):
                for j in range(hblock):
                    for y in range(H - (j + 1) * feaTemp.h + 1):
                        for x in range(W - (i + 1) * feaTemp.w + 1):
                            features.append(HaarLikeFeature(feaTemp.type, (y, x), (i + 1) * feaTemp.w, (j + 1) * feaTemp.h));
    
        elif feaTemp.ang == 45:
            for i in range(W):
                for j in range(H):
                    edge = (i + 1) * feaTemp.w + (j + 1) * feaTemp.h;
                    if edge > H or edge > W:
                        break;
                    for y in range(H - edge + 1):
                        for x in range(W - edge + 1):
                            features.append(HaarLikeFeature(feaTemp.type, (y, x), (i + 1) * feaTemp.w, (j + 1) * feaTemp.h));
    
    return features;

# 读取图片
def readImage(filepath, imgSize):
    im = Image.open(filepath);
    im = im.resize((imgSize, imgSize)).convert("L");
    return np.array(im);

# 读取训练数据
def loadFaceDataSet(mitSrc, imgSize, DEBUG):
    images = [];
    # 读取MIT数据
    faceFolder = mitSrc + 'faces/';
    nonfaceFolder = mitSrc + 'nonfaces/';
    
    faces = os.listdir(faceFolder);
    nonfaces = os.listdir(nonfaceFolder);
    
    if DEBUG:
        faces = faces[:5];
        nonfaces = nonfaces[:5];
    
    pos_weight = 1.0 / (2 * len(faces));
    neg_weight = 1.0 / (2 * len(nonfaces));
    
    print 'loading face images...';
    for filename in faces:
        im = readImage(faceFolder + filename, imgSize);
        images.append(IntegralImage(im, 1, pos_weight));
    
    print 'loading nonface images...';
    for filename in nonfaces:
        im = readImage(nonfaceFolder + filename, imgSize);
        images.append(IntegralImage(im, 0, neg_weight));
    
    np.random.shuffle(images);
    return images;

# 画出源数据
def plotDatas(datas):
    fig = plt.figure();
    fig.add_subplot(111);
    plt.imshow(datas, cmap="gray");
    plt.show();

# AdaBoost 算法
class AdaBoost:
    
    def __init__(self, datas, features):
        self.datas = datas;
        self.features = features;
    
    def train(self, SparkMaster):
        conf = SparkConf().setAppName('MNIST CNN Test').setMaster(SparkMaster);
        sc = SparkContext(conf=conf);
    
        T = 1;
        choose = {};
        for i in range(T):
            print 'training classifer ', i, ' / ', T;
            feas = [];
            for fea in self.features:
                feas.append([fea.type, fea.pos, fea.w, fea.h]);
            feas = sc.parallelize(feas, 3);
            feas = feas.map(lambda line : trainWeakClassifier(line[0], line[1], line[2], line[3], self.datas));
            
            items = feas.collect();
            print len(items);
            print items[0];
        
            self.trainWeakClassifier();
            pickFea = self.pickWeakClassifier();
            alpha = 0.5 * np.log((1.0 - (pickFea.err + 0.0001)) / (pickFea.err + 0.0001));
            choose[pickFea] = alpha;
            self.updateSamplesWeight(pickFea, alpha);
        
        sc.stop();
        return choose;
    
    # 选择弱分类器
    def pickWeakClassifier(self):
        bestFea = None;
        bestErr = 1.0;
        for fea in self.features:
            if fea.err < bestErr:
                bestFea = fea;
                bestErr = fea.err;
        self.features.remove(bestFea);
        return bestFea;
    
    # 更新样本权重
    def updateSamplesWeight(self, pickFea, alpha):
        z = 0.0;
        for data in self.datas:
            if pickFea.getVote(data) == data.label:
                z += data.weight * np.exp(-1.0 * alpha);
            else:
                z += data.weight * np.exp(alpha);
        
        for data in self.datas:
            if pickFea.getVote(data) == data.label:
                data.weight = data.weight * np.exp(-1.0 * alpha) / z;
            else:
                data.weight = data.weight * np.exp(alpha) / z;

# 训练弱分类器
def trainWeakClassifier(fType, fPos, w, h, datas):
    stable = [];
    wpos = 0;
    wneg = 0;
    for data in datas:
        stable.append([getEigenvalue(fType, fPos, w, h, data), data.label, data.weight]);
        if data.label == 1:
            wpos += data.weight;
        else:
            wneg += data.weight;
     
    sortedScore = sorted(stable, key=operator.itemgetter(0));
    spos = 0;
    sneg = 0;
    bestSplit = 0;
    bestErr = 1;
    polarity = 1;
     
    for item in sortedScore:
        err = min((spos + wneg - sneg), (sneg + wpos - spos));
        if err < bestErr:
            bestErr = err;
            bestSplit = item[0];
            if (spos + wneg - sneg) < (sneg + wpos - spos):
                polarity = -1;
            else:
                polarity = 1;
         
        if item[1] == 1:
            spos += item[2];
        else:
            sneg += item[2];
     
    return bestSplit, bestErr, polarity;

# 导出模型
def exportClassifier(filepath, classifiers):
    fileHandler = open(filepath, "w");
    for fea in classifiers:
        fileHandler.write(fea.toString() + "\n");
    fileHandler.close();
   
# 主函数
def main():
    DEBUG = False;
    imgSize = 24;
    os.environ['SPARK_HOME'] = '/apps/spark/spark-1.4.1-bin-hadoop2.6';
    SparkMaster = 'spark://localhost:7077';
    
    mitSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/MIT/';
    modelFile = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/model.txt';
    
    templates = initFeaTemplates();
    features = initFeatures(imgSize, imgSize, templates);
    
    trainDatas = loadFaceDataSet(mitSrc, imgSize, DEBUG);
       
    adaBoost = AdaBoost(trainDatas, features);
    classifiers = adaBoost.train(SparkMaster);
    
    exportClassifier(modelFile, classifiers);

main();
