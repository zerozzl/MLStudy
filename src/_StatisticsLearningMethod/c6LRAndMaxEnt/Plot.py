import numpy as np;
import matplotlib.pyplot as plt;

def plot(folder, filename, weights=None):
    data, label = loadData(folder, filename);
    __plot__(data, label, weights);

def loadData(folder, filename):
    data = [];
    label = [];
    fr = open(folder + filename);
    for line in fr.readlines():
        if filename == "data1.txt":
            arr = line.strip().split(",");
            data.append([float(arr[0]), float(arr[1])]);
            label.append(int(arr[2]));
        elif filename == "data2.txt":
            arr = line.strip().split();
            data.append([float(arr[0]), float(arr[1])]);
            label.append(int(arr[2]));
        elif filename == "data3.txt":
            arr = line.strip().split(",");
            data.append([float(arr[0]), float(arr[1])]);
            label.append(int(arr[2]));
            
    return data, label;

def __plot__(data, label, weights=None):
    dataArr = np.array(data);
    m = np.shape(dataArr)[0];
    x_1 = [];
    y_1 = [];
    x_2 = [];
    y_2 = [];
    for i in range(m):
        if int(label[i]) == 1:
            x_1.append(dataArr[i, 0]);
            y_1.append(dataArr[i, 1]);
        else:
            x_2.append(dataArr[i, 0]);
            y_2.append(dataArr[i, 1]);
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(x_1, y_1, s=30, c='red', marker='s');
    ax.scatter(x_2, y_2, s=30, c='blue');
    
    if weights is not None:
        if len(weights) <= 3:
            xbegin = np.min(x_1);
            xend = np.max(x_1);
            dist = int(xend - xbegin);
            deno = 1;
            while dist != 0:
                dist = dist / 10;
                deno += 1;
            step = 1.0 / pow(10, deno - 1);
            x = np.arange(xbegin - 5 * step, xend + 5 * step, step);
            y = (-weights[0] - weights[1] * x) / weights[2];
            ax.plot(x, y);
        else:
            degree = 6;
            x = np.arange(-1, 1.5, 0.025)
            y = np.arange(-1, 1.5, 0.025)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros((len(x), len(y)));
            for i in range(len(x)):
                for j in range(len(y)):
                    tmp = [1];
                    for ii in range(degree):
                        for jj in range(ii + 2):
                            tmp.append(pow(x[i], ii + 1 - jj) * pow(y[j], jj));
                    Z[i, j] = np.sum(np.mat(tmp) * np.mat(weights).T);
            Z = Z.T;
            plt.contour(X, Y, Z, [0, 0]);
    
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show();

def plotCost(cost):
    iterNum = [x[0] for x in cost]
    value = [x[1] for x in cost]
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(iterNum, value, s=1, c='red');
    plt.xlabel('Iter Times');
    plt.ylabel('cost');
    plt.show();

