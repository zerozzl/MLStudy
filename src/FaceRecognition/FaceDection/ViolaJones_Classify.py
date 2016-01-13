# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# 读取图片
def readImage(filepath, imgSize=None):
    im = Image.open(filepath);
    im = im.convert("L");
    if imgSize is not None:
        im = im.resize((imgSize, imgSize));
    
    im = np.array(im);
    im = (im - np.mean(im)) / 255;
    return im;

# 画出源数据
def plotDatas(datas):
    fig = plt.figure();
    fig.add_subplot(111);
    plt.imshow(datas, cmap="gray");
    plt.show();

# 积分图
class IntegralImage:
    
    def __init__(self, image, label):
        self.orig = np.array(image);
        self.sat, self.rsat = self.computeIntegrogram(self.orig);
        self.label = label;

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
    
    def __init__(self, fea_type, pos, w, h, th, p, weight):
        self.type = fea_type;
        self.pos = pos;
        self.w = w;
        self.h = h;
        self.theta = th;
        self.p = p;
        self.weight = weight;
    
    def getEigenvalue(self, intImg):
        eigenvalue = 0;
        if self.type == '1a':
            part = self.w / 2;
            negative = intImg.getAreaSum(self.pos[1], self.pos[0], part, self.h);
            positive = intImg.getAreaSum(self.pos[1] + part, self.pos[0], part, self.h);
            eigenvalue = positive - negative;
        elif self.type == '1b':
            part = self.h / 2;
            negative = intImg.getAreaSum(self.pos[1], self.pos[0], self.w, part);
            positive = intImg.getAreaSum(self.pos[1], self.pos[0] + part, self.w, part);
            eigenvalue = positive - negative;
        elif self.type == '1c':
            part = self.w / 2;
            negative = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.pos[1] + self.h + part, self.pos[0] + part, part, self.h, 45);
            eigenvalue = positive - negative;
        elif self.type == '1d':
            part = self.h / 2;
            negative = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.pos[1] + part, self.pos[0] + part, self.w, part, 45);
            eigenvalue = positive - negative;
        elif self.type == '2a':
            part = self.w / 3;
            negative1 = intImg.getAreaSum(self.pos[1], self.pos[0], part, self.h);
            positive = intImg.getAreaSum(self.pos[1] + part, self.pos[0], part, self.h);
            negative2 = intImg.getAreaSum(self.pos[1] + 2 * part, self.pos[0], part, self.h);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2b':
            part = self.w / 4;
            negative1 = intImg.getAreaSum(self.pos[1], self.pos[0], part, self.h);
            positive = intImg.getAreaSum(self.pos[1] + part, self.pos[0], 2 * part, self.h);
            negative2 = intImg.getAreaSum(self.pos[1] + 3 * part, self.pos[0], part, self.h);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2c':
            part = self.h / 3;
            negative1 = intImg.getAreaSum(self.pos[1], self.pos[0], self.w, part);
            positive = intImg.getAreaSum(self.pos[1], self.pos[0] + part, self.w, part);
            negative2 = intImg.getAreaSum(self.pos[1], self.pos[0] + 2 * part, self.w, part);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2d':
            part = self.h / 4;
            negative1 = intImg.getAreaSum(self.pos[1], self.pos[0], self.w, part);
            positive = intImg.getAreaSum(self.pos[1], self.pos[0] + part, self.w, 2 * part);
            negative2 = intImg.getAreaSum(self.pos[1], self.pos[0] + 3 * part, self.w, part);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2e':
            part = self.w / 3;
            negative1 = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.pos[1] + self.h + part, self.pos[0] + part, part, self.h, 45);
            negative2 = intImg.getAreaSum(self.pos[1] + self.h + 2 * part, self.pos[0] + 2 * part, part, self.h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2f':
            part = self.w / 4;
            negative1 = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.pos[1] + self.h + part, self.pos[0] + part, 2 * part, self.h, 45);
            negative2 = intImg.getAreaSum(self.pos[1] + self.h + 3 * part, self.pos[0] + 3 * part, part, self.h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2g':
            part = self.h / 3;
            negative1 = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.pos[1] + 2 * part, self.pos[0] + part, self.w, part, 45);
            negative2 = intImg.getAreaSum(self.pos[1] + part, self.pos[0] + 2 * part, self.w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2h':
            part = self.h / 4;
            negative1 = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.pos[1] + 3 * part, self.pos[0] + part, self.w, 2 * part, 45);
            negative2 = intImg.getAreaSum(self.pos[1] + part, self.pos[0] + 3 * part, self.w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '3a':
            partw = self.w / 3;
            parth = self.h / 3;
            whole = intImg.getAreaSum(self.pos[1], self.pos[0], self.w, self.h);
            positive = intImg.getAreaSum(self.pos[1] + partw, self.pos[0] + parth, partw, parth);
            eigenvalue = 2 * positive - whole;
        elif self.type == '3b':
            partw = self.w / 3;
            parth = self.h / 3;
            whole = intImg.getAreaSum(self.pos[1] + self.h, self.pos[0], self.w, self.h, 45);
            positive = intImg.getAreaSum(self.pos[1] + partw + 2 * parth, self.pos[0] + partw + parth, partw, parth, 45);
            eigenvalue = 2 * positive - whole;
        
        return eigenvalue;
    
    # 投票
    def getVote(self, intImg):
        score = self.getEigenvalue(intImg);
        if self.p * score <= self.p * self.theta:
            return 1;
        else:
            return 0;
    
    def toString(self):
        return 'type:' + self.type + '|pos:' + str(self.pos[0]) + ',' + str(self.pos[1]) + '|w:' + str(self.w) + '|h:' + str(self.h) + '|theta:' + str(self.theta) + '|p:' + str(self.p) + '|weight:' + str(self.weight);

def importAdaboostModel(filepath):
    fr = open(filepath);
    model = [];
    for line in fr.readlines():
        items = line.strip().split('|');
        ftype = None;
        pos = None;
        w = None;
        h = None;
        theta = None;
        p = None;
        weight = None;
        for item in items:
            key, val = item.split(':');
            if key == 'type':
                ftype = val;
            elif key == 'pos':
                tmp = val.strip().split(',');
                pos = (int(tmp[0]), int(tmp[1]));
            elif key == 'w':
                w = int(val);
            elif key == 'h':
                h = int(val);
            elif key == 'theta':
                theta = float(val);
            elif key == 'p':
                p = int(val);
            elif key == 'weight':
                weight = float(val);
        model.append(HaarLikeFeature(ftype, pos, w, h, theta, p, weight));
    return model;

# 读取MIT数据
def loadMIT(dateSrc, imgSize, DEBUG):
    images = [];
    # 读取MIT数据
    faceFolder = dateSrc + 'faces/';
    nonfaceFolder = dateSrc + 'nonfaces/';
    
    faces = os.listdir(faceFolder);
    nonfaces = os.listdir(nonfaceFolder);
    
    if DEBUG:
        faces = faces[:5];
        nonfaces = nonfaces[:5];
    
    for filename in faces:
        im = readImage(faceFolder + filename, imgSize);
        images.append(IntegralImage(im, 1));
    
    for filename in nonfaces:
        im = readImage(nonfaceFolder + filename, imgSize);
        images.append(IntegralImage(im, 0));
    
    return images;

# 读取Yale数据
def loadYale(dateSrc, imgSize, DEBUG):
    images = [];
    
    faces = os.listdir(dateSrc);
    if DEBUG:
        faces = faces[:10];
    
    for filename in faces:
        im = readImage(dateSrc + filename, imgSize);
        images.append(IntegralImage(im, 1));
     
    return images;

# 读取ORL数据
def loadORL(dateSrc, imgSize, DEBUG):
    images = [];
    
    folders = os.listdir(dateSrc);
    
    if DEBUG:
        folders = folders[:1];
    
    for fol in folders:
        faces = os.listdir(dateSrc + fol);
        for filename in faces:
            im = readImage(dateSrc + fol + '/' + filename, imgSize);
            images.append(IntegralImage(im, 1));
     
    return images;

# 读取FERET数据
def loadFERET(dateSrc, imgSize, DEBUG):
    images = [];
    
    folders = os.listdir(dateSrc);
    
    if DEBUG:
        folders = folders[:1];
    
    for fol in folders:
        faces = os.listdir(dateSrc + fol);
        for filename in faces:
            im = readImage(dateSrc + fol + '/' + filename, imgSize);
            images.append(IntegralImage(im, 1));
     
    return images;

def predict(images, model):
    zw = 0.0;
    for item in model:
        zw += item.weight;
    
    err = 0;
    m = len(images);
    for i in range(m):
        print 'predicting img: ' + str(i + 1) + '/' + str(m);
        score = 0.0;
        pred = None;
        for item in model:
            score += item.getVote(images[i]) * item.weight;
        
        if score >= 0.6 * zw:
            pred = 1;
        else:
            pred = 0;
        
        if pred != images[i].label:
            err += 1;
    
    print 'Predict Accuracy: ' + str(float(m - err) * 100 / m) + '%';

def findFace(image, model):
    pass;

def main():
    DEBUG = False;
    modelfile = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/adaboost_model.txt';
    model = importAdaboostModel(modelfile);
    
#     # check accuracy
#     mitSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/MIT/';
#     yaleSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/Yale/';
#     orlSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/ORL/';
#     feretSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/FERET/FERET_80_80/';
#      
#     mitDatas = loadMIT(mitSrc, 24, DEBUG);
#     yaleDatas = loadYale(yaleSrc, 24, DEBUG);
#     orlDatas = loadORL(orlSrc, 24, DEBUG);
#     feretDatas = loadFERET(feretSrc, 24, DEBUG);
#     predict(mitDatas, model);

    picFile = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/LFW/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg';
    image = IntegralImage(readImage(picFile), -1);
    findFace(image, model);

main();
