import numpy as np
import matplotlib.pyplot as plt; 

# original problem
def original(X, y, alpha):
    m, n = np.shape(X);
    w = np.zeros((n, 1));
    b = 0;
    stop = False;
    while(not stop):
        stop = True;
        for i in range(m):
            if np.sum(-y[i] * (X[i] * w + b)) >= 0:
                w = w + alpha * y[i] * X[i].T;
                b = b + alpha * y[i];
                # print "error point: x%d, w:%s, b:%d" % ((i + 1), w.T, b);
                stop = False;
                break;
    
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(X[0, 0], X[0, 1], s=30, c='blue', marker='s');
    ax.scatter(X[1, 0], X[1, 1], s=30, c='blue', marker='s');
    ax.scatter(X[2, 0], X[2, 1], s=30, c='red', marker='x');
    
    xl = np.arange(0, 3, 0.1);
    yl = np.array((-w[0] * xl - b) / w[1]).T;
    
    ax.plot(xl, yl);
    
    plt.title("original problem");
    plt.ylim(0, 5);
    plt.xlim(0, 5);
    plt.show();

# dual problem
def dual(X, y, alpha):
    m = np.shape(X)[0];
    gram = np.zeros((m, m));
    for i in range(m):
        for j in range(m):
            gram[i][j] = X[i] * X[j].T;
    gram = np.mat(gram);
    
    a = np.zeros((m, 1));
    b = 0;
    stop = False;
    
    while(not stop):
        stop = True;
        for i in range(m):
            if y[i] * (np.sum(np.multiply(a.T, y) * gram[:, i]) + b) <= 0:
                a[i] = a[i] + alpha;
                b = b + alpha * y[i];
                # print "error point: x%d, a:%s, b:%d" % ((i + 1), a.T, b);
                stop = False;
                break;
    
    w = np.multiply(a.T, y) * X;
    w = np.array(w)[0];
    
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(X[0, 0], X[0, 1], s=30, c='blue', marker='s');
    ax.scatter(X[1, 0], X[1, 1], s=30, c='blue', marker='s');
    ax.scatter(X[2, 0], X[2, 1], s=30, c='red', marker='x');
    
    xl = np.arange(0, 3, 0.1);
    yl = np.array((-w[0] * xl - b) / w[1]).T;
    
    ax.plot(xl, yl);
    
    plt.title("dual problem");
    plt.ylim(0, 5);
    plt.xlim(0, 5);
    plt.show();


def main():
    X = np.mat([[3, 3], [4, 3], [1, 1]]);
    y = np.array([1, 1, -1]);
    alpha = 1;
    
#     original(X, y, alpha);
    dual(X, y, alpha);


main();