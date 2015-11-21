#coding: UTF-8
import numpy as np
import struct
import matplotlib.pyplot as plt

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
        im = struct.unpack_from('>784B' ,buf, index)
        index += struct.calcsize('>784B')
        datas.append(im);
    
    datas = np.mat(datas).T;
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
        la = struct.unpack_from('>1B' ,buf, index)
        index += struct.calcsize('>1B')
        labels.append(int(la[0]));
    
    return labels;

# 画出源数据
def plotDatas(datas, numRows, numColumns):
    fig = plt.figure()
    for i in range(100):
        fig.add_subplot(10, 10, i+1);
        img = np.array(datas[i])
        img = img.reshape(numRows, numColumns)
        plt.imshow(img , cmap='gray')
    plt.show()

