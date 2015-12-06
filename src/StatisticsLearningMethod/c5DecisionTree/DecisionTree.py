from math import log;
import operator

def calcShannonEnt(data):
    m = len(data);
    labelMat = {};
    for x in data:
        label = x[-1];
        labelMat[label] = labelMat.get(label, 0) + 1;
    
    ent = 0;
    for key in labelMat:
        prob = float(labelMat[key]) / m;
        ent += -1 * prob * log(prob, 2);
    
    return ent;

def splitData(data, axis, value):
    split = [];
    for x in data:
        if x[axis] == value:
            nx = x[:axis];
            nx.extend(x[axis+1:]);
            split.append(nx);
    
    return split;

def chooseFeatureToSplit(data, type=0): # type=0 : c3,    type=1 : c4.5
    num = len(data[0]) - 1;
    baseEnt = calcShannonEnt(data);
    bestInfoGain = 0;
    bestFeature = -1;
    for i in range(num):
        feaSet = set([x[i] for x in data]);
        newEnt = 0;
        Ha = 0;
        for fea in feaSet:
            subData = splitData(data, i, fea);
            prob = float(len(subData)) / len(data);
            newEnt += prob * calcShannonEnt(subData);
            if type == 1:
                Ha += -1 * prob * log(prob, 2);
        
        infoGain = baseEnt - newEnt;
        if type == 1:
            infoGain = infoGain / Ha;
        
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain;
            bestFeature = i;
    
    return bestFeature;

def majorityClass(classList):
    labelMat = {};
    for item in classList:
        labelMat[item] = labelMat.get(item, 0) + 1;
    sort = sorted(labelMat.iteritems(), key = operator.itemgetter(1), reverse = True);
    return sort[0][0];

def createTree(data, label):
    classList = [x[-1] for x in data];
    if classList.count(classList[0]) == len(classList):
        return classList[0];
    
    if len(data[0]) == 1:
        return majorityClass(classList);
    
    bestFea = chooseFeatureToSplit(data, 1);
    bestFeaLabel = label[bestFea];
    
    tree = {bestFeaLabel:{}};
    del(label[bestFea]);
    
    feaSet = set([x[bestFea] for x in data]);
    for fea in feaSet:
        subLabel = label[:];
        tree[bestFeaLabel][fea] = createTree(splitData(data, bestFea, fea), subLabel);
    
    return tree;

def main():
    data = [[1, 0, 0, 3, 0], 
            [1, 0, 0, 2, 0], 
            [1, 1, 0, 2, 1], 
            [1, 1, 1, 3, 1], 
            [1, 0, 0, 3, 0], 
            [2, 0, 0, 3, 0], 
            [2, 0, 0, 2, 0], 
            [2, 1, 1, 2, 1], 
            [2, 0, 1, 1, 1], 
            [2, 0, 1, 1, 1], 
            [3, 0, 1, 1, 1], 
            [3, 0, 1, 2, 1], 
            [3, 1, 0, 2, 1], 
            [3, 1, 0, 1, 1], 
            [3, 0, 0, 3, 0]];
    label = ['agee', 'have work', 'have house', 'credit pos'];
    
    tree = createTree(data, label);
    print tree;
    
    
main();
