import csv;
import numpy as np;
from PIL import Image;

def loadTrainData(filepath):
    print 'loading training data...'
    datas = [];
    with open(filepath, 'rb') as dataFile:
        lines = csv.reader(dataFile);
        for line in lines:
            datas.append(line);
    datas.remove(datas[0]);
    datas = np.array(datas);
    labels = datas[:, 0];
    features = datas[:, 1:];
    return stringToInt(features), stringToInt(labels);

def loadTestData(filepath):
    print 'loading test data...'
    datas = [];
    with open(filepath, 'rb') as dataFile:
        lines = csv.reader(dataFile);
        for line in lines:
            datas.append(line);
    datas.remove(datas[0]);
    datas = np.array(datas);
    return stringToInt(datas);

def stringToInt(source):
    source = np.mat(source);
    m, n = np.shape(source);
    target = np.zeros((m, n));
    for i in range(m):
        for j in range(n):
            target[i, j] = int(source[i, j]);
    return target;

def generateImage(folder, features, labels):
    print 'generating images...'
    m, n = np.shape(features);
    mm = int(np.sqrt(n));
    for k in range(m):
        digit = features[k];
        c = Image.new("RGB",(mm, mm));
        for i in range (0, mm):
            for j in range (0, mm):
                c.putpixel([i,j], int(digit[i * mm + j]));
        filename = folder + str(k + 1) + '_' + str(int(labels[k])) + '.png';
        c.save(filename);

def generateResultFile(filepath, data):
    with open(filepath, 'wb') as file:
        mywriter = csv.writer(file);
        for i in data:
            row = [i];
            mywriter.writerow(row);
