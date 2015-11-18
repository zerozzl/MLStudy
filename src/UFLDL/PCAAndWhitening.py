# coding: UTF-8
import numpy as np;
from numpy import linalg as la
import matplotlib.pyplot as plt;

def loadPCA_2DDataSet(filepath):
    datas = [];
    fr = open(filepath);
    for line in fr.readlines():
        arr = line.strip().split();
        datas.append([float(x) for x in arr]);
    datas = np.mat(datas).T;
    return datas;

def loadPCA_IMGDataSet(filepath):
    datas = [];
    fr = open(filepath);
    for line in fr.readlines():
        arr = line.strip().split();
        datas.append([float(x) for x in arr]);
    return datas;

def sampleImages(imgs, patchSize, numPatches):
    m = np.shape(imgs)[0];
    numsamples = numPatches / m;
    patches = np.zeros((patchSize * patchSize, numPatches));
    for i in range(m):
        img = np.mat(imgs[i]);
        img = img.reshape(512, 512).T;
        for j in range(numsamples):
            y = int(np.random.uniform(0, 512 - patchSize));
            x = int(np.random.uniform(0, 512 - patchSize));
            patches[:, i * numsamples + j] = img[y:y + patchSize, x:x + patchSize].reshape(patchSize * patchSize);
    return np.mat(patches);
    
# 画出原图像
def plotImages(folder, imgs):
    m = np.shape(imgs)[0];
    for i in range(m):
        img = imgs[i];
        img = np.mat(img);
        fig = plt.figure();
        fig.add_subplot(111);
        img = img.reshape(512, 512).T;
        plt.imshow(img , cmap="gray");
        
        filename = folder + "img_" + str(i) + ".png";
        plt.savefig(filename, format="png");

# 画出计算结果
def displayNetwork(patches, imgRow, imgCol, pRow, pCol, filepath=None):
    patches = np.mat(patches);
    fig = plt.figure();
    for i in range(imgRow * imgCol):
        fig.add_subplot(imgRow, imgCol, i + 1);
        img = patches[:, i];
        img = img.reshape(pRow, pCol).T;
        plt.imshow(img , cmap='gray');
    
    if filepath is not None:
        plt.savefig(filepath, format='png');
    else:
        plt.show();

def plotCovariance(covat, filepath=None):
    fig = plt.figure();
    fig.add_subplot(1,1,1);
    plt.imshow(covat);
    if filepath is not None:
        plt.savefig(filepath);
    else:
        plt.show();

def PCA_2D():
    datafile = "E:/TestDatas/MLStudy/UFLDL/pcaData.txt";
    resultFolder = "E:/TestDatas/MLStudy/UFLDL/result/PCAAndWhitening/";
    
    datas = loadPCA_2DDataSet(datafile);
    
    # 画出原数据
    fig1 = plt.figure();
    ax1 = fig1.add_subplot(111);
    ax1.scatter(datas[:, 0], datas[:, 1], s=30, c='blue');
    plt.savefig(resultFolder + "pca2d_1.png", format="png");

    # 画出方差方向
    m = np.shape(datas)[0];
    sigma = datas.T * datas / m;
    u, s, v = la.svd(sigma);
    
    ua = np.array(u)
    fig2 = plt.figure();
    ax2 = fig2.add_subplot(111);
    ax2.scatter(datas[:, 0], datas[:, 1], s=30, c='b');
    x = np.arange(0, 0.5, 0.01);
    y1 = ua[0][0] * x / ua[1][0];
    y2 = ua[0][1] * x / ua[1][1];
    ax2.plot(x, y1, 'r-');
    ax2.plot(x, y2, 'r-');
    plt.savefig(resultFolder + "pca2d_2.png", format="png");

    # 画出旋转后的数据
    xRot = u.T * datas.T;
    fig3 = plt.figure();
    ax3 = fig3.add_subplot(111);
    ax3.scatter(xRot[0, :], xRot[1, :], s=30, c='blue');
    plt.savefig(resultFolder + "pca2d_3.png", format="png");

    # 画出降维后的数据
    uh = np.zeros(np.shape(u));
    uh[:, 0:1] = u[:, 0:1];
    xHat = u * (uh.T * datas.T);
    fig4 = plt.figure();
    ax4 = fig4.add_subplot(111);
    ax4.scatter(xHat[0, :], xHat[1, :], s=30, c='blue');
    plt.savefig(resultFolder + "pca2d_4.png", format="png");

    epsilon = 1e-5;
    
    # PCA白化
    xPCAWhite = np.diag(1 / np.sqrt(s + epsilon)) * u.T * datas.T;
    fig5 = plt.figure();
    ax5 = fig5.add_subplot(111);
    ax5.scatter(xPCAWhite[0, :], xPCAWhite[1, :], s=30, c='blue');
    plt.savefig(resultFolder + "pca2d_5.png", format="png");

    # ZCA白化
    xZCAWhite = u * np.diag(1 / np.sqrt(s + epsilon)) * u.T * datas.T;
    fig6 = plt.figure();
    ax6 = fig6.add_subplot(111);
    ax6.scatter(xZCAWhite[0, :], xZCAWhite[1, :], s=30, c='blue');
    plt.savefig(resultFolder + "pca2d_6.png", format="png");

def PCA_IMG(datafile, resultFolder):
    datafile = "E:/TestDatas/MLStudy/UFLDL/IMAGES_PCA.dat";
    resultFolder = "E:/TestDatas/MLStudy/UFLDL/result/PCAAndWhitening/";
           
    patchSize = 12;
    numPatches = 10000;
    datas = loadPCA_IMGDataSet(datafile);
        
    # 画出原图像
    plotImages(resultFolder, datas);

    patches = sampleImages(datas, patchSize, numPatches);
    
    # 画出patches
    displayNetwork(patches, 14, 14, patchSize, patchSize, resultFolder + "patches.png");

    patches = patches - np.mean(patches, 0);
    
    m = np.shape(patches)[1];
    sigma = patches * patches.T / m;
    U, S, V = la.svd(sigma);
    
    xRot = U.T * patches;
    
    covat = xRot * xRot.T / m;
    plotCovariance(covat, resultFolder + "covar_pca.png");
    
    k = 0;
    var = 0;
    for i in range(len(S)):
        var = var + S[i];
        if (var / sum(S)) > 0.99:
            k = i + 1;
            break;
    
    uHat = np.zeros(np.shape(U));
    uHat[:, 0:k] = U[:, 0:k];
    xHat = U * (uHat.T * patches);
    displayNetwork(xHat, 14, 14, patchSize, patchSize, resultFolder + "patches_pca.png");

    epsilon = 0.1;
    xPCAWhite = np.diag(1 / np.sqrt(S + epsilon)) * U.T * patches;
    displayNetwork(xPCAWhite, 14, 14, patchSize, patchSize, resultFolder + "patches_pca_whitened.png");
    
    covarwh = xPCAWhite * xPCAWhite.T / m;
    plotCovariance(covarwh, resultFolder + "covar_pca_whitened.png");
    
    xZCAWhite = U * xPCAWhite;
    displayNetwork(xZCAWhite, 14, 14, patchSize, patchSize, resultFolder + "patches_zca_whitened.png");

def main():
    PCA_2D();
    PCA_IMG();


main();

