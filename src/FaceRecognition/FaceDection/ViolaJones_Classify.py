# -*- coding:utf-8 -*-
from PIL import Image
import ImageDraw
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
#     im = (im - np.mean(im)) / 255;
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
    
    def getEigenvalue(self, ftype, pos, w, h, intImg):
        eigenvalue = 0;
        if ftype == '1a':
            part = w / 2;
            negative = intImg.getAreaSum(pos[1], pos[0], part, h);
            positive = intImg.getAreaSum(pos[1] + part, pos[0], part, h);
            eigenvalue = positive - negative;
        elif ftype == '1b':
            part = h / 2;
            negative = intImg.getAreaSum(pos[1], pos[0], w, part);
            positive = intImg.getAreaSum(pos[1], pos[0] + part, w, part);
            eigenvalue = positive - negative;
        elif ftype == '1c':
            part = w / 2;
            negative = intImg.getAreaSum(pos[1] + h, pos[0], part, h, 45);
            positive = intImg.getAreaSum(pos[1] + h + part, pos[0] + part, part, h, 45);
            eigenvalue = positive - negative;
        elif ftype == '1d':
            part = h / 2;
            negative = intImg.getAreaSum(pos[1] + h, pos[0], w, part, 45);
            positive = intImg.getAreaSum(pos[1] + part, pos[0] + part, w, part, 45);
            eigenvalue = positive - negative;
        elif ftype == '2a':
            part = w / 3;
            negative1 = intImg.getAreaSum(pos[1], pos[0], part, h);
            positive = intImg.getAreaSum(pos[1] + part, pos[0], part, h);
            negative2 = intImg.getAreaSum(pos[1] + 2 * part, pos[0], part, h);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2b':
            part = w / 4;
            negative1 = intImg.getAreaSum(pos[1], pos[0], part, h);
            positive = intImg.getAreaSum(pos[1] + part, pos[0], 2 * part, h);
            negative2 = intImg.getAreaSum(pos[1] + 3 * part, pos[0], part, h);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2c':
            part = h / 3;
            negative1 = intImg.getAreaSum(pos[1], pos[0], w, part);
            positive = intImg.getAreaSum(pos[1], pos[0] + part, w, part);
            negative2 = intImg.getAreaSum(pos[1], pos[0] + 2 * part, w, part);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2d':
            part = h / 4;
            negative1 = intImg.getAreaSum(pos[1], pos[0], w, part);
            positive = intImg.getAreaSum(pos[1], pos[0] + part, w, 2 * part);
            negative2 = intImg.getAreaSum(pos[1], pos[0] + 3 * part, w, part);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2e':
            part = w / 3;
            negative1 = intImg.getAreaSum(pos[1] + h, pos[0], part, h, 45);
            positive = intImg.getAreaSum(pos[1] + h + part, pos[0] + part, part, h, 45);
            negative2 = intImg.getAreaSum(pos[1] + h + 2 * part, pos[0] + 2 * part, part, h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2f':
            part = w / 4;
            negative1 = intImg.getAreaSum(pos[1] + h, pos[0], part, h, 45);
            positive = intImg.getAreaSum(pos[1] + h + part, pos[0] + part, 2 * part, h, 45);
            negative2 = intImg.getAreaSum(pos[1] + h + 3 * part, pos[0] + 3 * part, part, h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2g':
            part = h / 3;
            negative1 = intImg.getAreaSum(pos[1] + h, pos[0], w, part, 45);
            positive = intImg.getAreaSum(pos[1] + 2 * part, pos[0] + part, w, part, 45);
            negative2 = intImg.getAreaSum(pos[1] + part, pos[0] + 2 * part, w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '2h':
            part = h / 4;
            negative1 = intImg.getAreaSum(pos[1] + h, pos[0], w, part, 45);
            positive = intImg.getAreaSum(pos[1] + 3 * part, pos[0] + part, w, 2 * part, 45);
            negative2 = intImg.getAreaSum(pos[1] + part, pos[0] + 3 * part, w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif ftype == '3a':
            partw = w / 3;
            parth = h / 3;
            whole = intImg.getAreaSum(pos[1], pos[0], w, h);
            positive = intImg.getAreaSum(pos[1] + partw, pos[0] + parth, partw, parth);
            eigenvalue = 2 * positive - whole;
        elif ftype == '3b':
            partw = w / 3;
            parth = h / 3;
            whole = intImg.getAreaSum(pos[1] + h, pos[0], w, h, 45);
            positive = intImg.getAreaSum(pos[1] + partw + 2 * parth, pos[0] + partw + parth, partw, parth, 45);
            eigenvalue = 2 * positive - whole;
        
        return eigenvalue / (w * h);
    
    # 投票
    def getVote(self, wstep, hstep, multiple, intImg):
        pos = (self.pos[0] + hstep, self.pos[1] + wstep);
        w = self.w * multiple;
        h = self.h * multiple;
        score = self.getEigenvalue(self.type, pos, w, h, intImg);
        score = score * multiple;
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
    
    posNum = 0;
    negNum = 0;
    tpr = 0;
    fpr = 0;
    err = 0;
    
    m = len(images);
    for i in range(m):
        print 'predicting img: ' + str(i + 1) + '/' + str(m);
        score = 0.0;
        pred = None;
        for item in model:
            score += item.getVote(0, 0, 1, images[i]) * item.weight;
        
        if score >= 0.5 * zw:
            pred = 1;
        else:
            pred = 0;
        
        if images[i].label == 1:
            posNum += 1;
            if pred == 1:
                tpr += 1;
            else:
                err += 1;
        else :
            negNum += 1;
            if pred == 1:
                fpr += 1;
                err += 1;
    
    print tpr, fpr;
    tpr = float(tpr) * 100 / posNum;
    fpr = float(fpr) * 100 / negNum;
    print 'Accuracy: ' + str(float(m - err) * 100 / m) + '%, TPR: ' + str(tpr) + '%, FPR: ' + str(fpr) + '%';

# 查找人脸
def findFace(image, model, detSize):
    faces = {};
    zw = 0.0;
    for item in model:
        zw += item.weight;
    ih, iw = np.shape(image.orig);
    
    multiple = 1;
    step = 6;
    
    while(detSize < iw or detSize < ih):
        xstep = iw - (detSize + step) + 1;
        ystep = ih - (detSize + step) + 1;
        total = ((xstep / step) + 1) * ((ystep / step) + 1);
        current = 0;
        for i in range(0, ystep, step):
            for j in range(0, xstep, step):
                current += 1;
                print 'detSize: ' + str(detSize) + ', inner: ' + str(current) + '/' + str(total);
                score = 0.0;
                for item in model:
                    score += item.getVote(j, i, multiple, image) * item.weight;
                
                if score >= 0.5 * zw:
                    if faces.has_key(multiple):
                        tmp = faces.get(multiple);
                        tmp.append([i, j]);
                        faces[multiple] = tmp;
                    else:
                        faces[multiple] = [[i, j]];
                    
        multiple = multiple * 2;
        detSize = detSize * 2;
    
    return faces;

# 合并查找人脸的结果
def combineFaces(faces, detSize):
    T = 5;
    combineResult = {};
    for key in faces:
        all = faces.get(key);
        result = [];
        if len(all) < T:
            continue;
        
        R = (key * detSize) * (key * detSize);
        while(len(all) > 0):
            point = all[0];
            stack = [];
            all.remove(point);
            for item in all:
                if (np.square(point[0]-item[0]) + np.square(point[1]-item[1])) < R:
                    stack.append(item);
            if len(stack) > T:
                size = len(stack) + 1;
                target = point;
                for item in stack:
                    target = [target[0] + item[0], target[1] + item[1]];
                    all.remove(item);
                target = [target[0] / size, target[1] / size];
                result.append(target);
        
        combineResult[key] = result;
    
    return combineResult;
    
def main():
    DEBUG = False;
    detSize = 24;
#     modelfile = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/adaboost_model.txt';
    modelfile = 'E:/TestDatas/MLStudy/FaceDection/adaboost_model.txt';
    model = importAdaboostModel(modelfile);
    
#     # check accuracy
#     mitSrc = 'E:/TestDatas/MLStudy/FaceDection/MIT/';
#     yaleSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/Yale/';
#     orlSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/ORL/';
#     feretSrc = '/home/hadoop/ProgramDatas/MLStudy/FaceDection/FERET/FERET_80_80/';
#      
#     mitDatas = loadMIT(mitSrc, 24, DEBUG);
#     yaleDatas = loadYale(yaleSrc, 24, DEBUG);
#     orlDatas = loadORL(orlSrc, 24, DEBUG);
#     feretDatas = loadFERET(feretSrc, 24, DEBUG);
#     predict(mitDatas, model);

    picFile = 'z:/ViolaJones/lfw1.jpg';
    image = IntegralImage(readImage(picFile), -1);
#     plotDatas(image.orig);
    faces = findFace(image, model, detSize);

    faces = combineFaces(faces, detSize);
    
    img = Image.open(picFile);
    img_d = ImageDraw.Draw(img);
    
    for mul in faces:
        targets = faces[mul];
        for face in targets:
            distance = mul * detSize;
            img_d.line(((face[0], face[1]), (face[0], face[1] + distance),
                     (face[0] + distance, face[1] + distance),
                    (face[0] + distance, face[1]), (face[0], face[1])), fill=150);
        img.save('z:/ViolaJones/result.png');

main();
