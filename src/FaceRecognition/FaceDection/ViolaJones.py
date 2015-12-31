# -*- coding:utf-8 -*-
import numpy as np

# 特征模板
class FeaTemplate:
    
    def __init__(self, t, ang, w, h):
        self.type = t;
        self.ang = ang;
        self.w = w;
        self.h = h;

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
    
    def __init__(self, fea_type, pos, w, h):
        self.type = fea_type;
        self.top_left = pos;
        self.w = w;
        self.h = h;
    
    def getEigenvalue(self, intImg):
        eigenvalue = 0;
        if self.type == '1a':
            part = self.w / 2;
            negative = intImg.getAreaSum(self.top_left[1], self.top_left[0], part, self.h);
            positive = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0], part, self.h);
            eigenvalue = positive - negative;
        elif self.type == '1b':
            part = self.h / 2;
            negative = intImg.getAreaSum(self.top_left[1], self.top_left[0], self.w, part);
            positive = intImg.getAreaSum(self.top_left[1], self.top_left[0] + part, self.w, part);
            eigenvalue = positive - negative;
        elif self.type == '1c':
            part = self.w / 2;
            negative = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.top_left[1] + self.h + part, self.top_left[0] + part, part, self.h, 45);
            eigenvalue = positive - negative;
        elif self.type == '1d':
            part = self.h / 2;
            negative = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0] + part, self.w, part, 45);
            eigenvalue = positive - negative;
        elif self.type == '2a':
            part = self.w / 3;
            negative1 = intImg.getAreaSum(self.top_left[1], self.top_left[0], part, self.h);
            positive = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0], part, self.h);
            negative2 = intImg.getAreaSum(self.top_left[1] + 2 * part, self.top_left[0], part, self.h);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2b':
            part = self.w / 4;
            negative1 = intImg.getAreaSum(self.top_left[1], self.top_left[0], part, self.h);
            positive = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0], 2 * part, self.h);
            negative2 = intImg.getAreaSum(self.top_left[1] + 3 * part, self.top_left[0], part, self.h);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2c':
            part = self.h / 3;
            negative1 = intImg.getAreaSum(self.top_left[1], self.top_left[0], self.w, part);
            positive = intImg.getAreaSum(self.top_left[1], self.top_left[0] + part, self.w, part);
            negative2 = intImg.getAreaSum(self.top_left[1], self.top_left[0] + 2 * part, self.w, part);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2d':
            part = self.h / 4;
            negative1 = intImg.getAreaSum(self.top_left[1], self.top_left[0], self.w, part);
            positive = intImg.getAreaSum(self.top_left[1], self.top_left[0] + part, self.w, 2 * part);
            negative2 = intImg.getAreaSum(self.top_left[1], self.top_left[0] + 3 * part, self.w, part);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2e':
            part = self.w / 3;
            negative1 = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.top_left[1] + self.h + part, self.top_left[0] + part, part, self.h, 45);
            negative2 = intImg.getAreaSum(self.top_left[1] + self.h + 2 * part, self.top_left[0] + 2 * part, part, self.h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2f':
            part = self.w / 4;
            negative1 = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], part, self.h, 45);
            positive = intImg.getAreaSum(self.top_left[1] + self.h + part, self.top_left[0] + part, 2 * part, self.h, 45);
            negative2 = intImg.getAreaSum(self.top_left[1] + self.h + 3 * part, self.top_left[0] + 3 * part, part, self.h, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2g':
            part = self.h / 3;
            negative1 = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.top_left[1] + 2 * part, self.top_left[0] + part, self.w, part, 45);
            negative2 = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0] + 2 * part, self.w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '2h':
            part = self.h / 4;
            negative1 = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], self.w, part, 45);
            positive = intImg.getAreaSum(self.top_left[1] + 3 * part, self.top_left[0] + part, self.w, 2 * part, 45);
            negative2 = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0] + 3 * part, self.w, part, 45);
            eigenvalue = positive - negative1 - negative2;
        elif self.type == '3a':
            part = self.w / 3;
            whole = intImg.getAreaSum(self.top_left[1], self.top_left[0], self.w, self.h);
            positive = intImg.getAreaSum(self.top_left[1] + part, self.top_left[0] + part, part, part);
            eigenvalue = 2 * positive - whole;
        elif self.type == '3b':
            part = self.w / 3;
            whole = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0], self.w, self.h, 45);
            positive = intImg.getAreaSum(self.top_left[1] + self.h, self.top_left[0] + 2 * part, part, part, 45);
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
    features = [];
    for feaTemp in templates:
        if feaTemp.ang == 0:
            w = feaTemp.w;
            h = feaTemp.h;
        elif feaTemp.ang == 45:
            w = h = feaTemp.w + feaTemp.h;
        
        X = int(W / w);
        Y = int(H / h);
        
        for i in range(X):
            for j in range(Y):
                for y in range(H - (j + 1) * h + 1):
                    for x in range(W - (i + 1) * w + 1):
                        features.append(HaarLikeFeature(feaTemp.type, (y, x), (i + 1) * feaTemp.w, (j + 1) * feaTemp.h));

    return features;

# # 获取Haar特征数量
# def getHaarCount(W, H, w, h, angle=0):
#     if angle == 45:
#         w = h = w + h;
#     X = int(W / w);
#     Y = int(H / h);
#     return  X * Y * (W + 1 - w * (X + 1) / 2.0) * (H + 1 - h * (Y + 1) / 2.0);

# 主函数
def main():
    templates = initFeaTemplates();
    features = initFeatures(24, 24, templates);
    print len(features);
    for i in range(100):
        print features[i].type, features[i].top_left, features[i].w, features[i].h;


# im = IntegralImage([[1, 2, 3, 4, 5, 6, 7],
#                     [8, 9, 10, 11, 12, 13, 14],
#                     [15, 16, 17, 18, 19, 20, 21],
#                     [22, 23, 24, 25, 26, 27, 28],
#                     [29, 30, 31, 32, 33, 34, 35],
#                     [36, 37, 38, 39, 40, 41, 42],
#                     [43, 44, 45, 46, 47, 48, 49]], 1);
# print im.orig;
# # print im.sat;
# # print im.rsat;
# print im.getAreaSum(3, 0, 3, 3, 45);
# haarlike = HaarLikeFeature('3a', (0, 0), 3, 3);
# eigenvalue = haarlike.getEigenvalue(im);
# print eigenvalue;

# ws = set();
# hs = set();
# 
# print len(feas);
# for i in range(len(feas)):
#     fe = feas[i];
#     ws.add(fe.w);
#     hs.add(fe.h);
#     if i < 1000:
#         print 'pos: ', fe.top_left, ', width: ', fe.w, 'height: ', fe.h;
# 
# print ws;
# print hs;

# temps = initFeaTemplates();
# sum = 0;
# for temp in temps:
#     print getHaarCount(24, 24, temp.w, temp.h, temp.ang);
#     sum += getHaarCount(24, 24, temp.w, temp.h, temp.ang);
# print sum;
