# -*- coding:utf-8 -*-
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm  
from matplotlib.ticker import LinearLocator, FormatStrFormatter  
import matplotlib.pyplot as plt  

def h_1(x, y):
    return np.sin(np.sqrt(np.square(x) + np.square(y))) / np.sqrt(np.square(x) + np.square(y)) + np.exp((np.cos(2 * math.pi * x) + np.cos(2 * math.pi * y)) / 2) - 2.71289;

def h_2(x, y):
    return 0.5 - (np.square((np.sin(np.sqrt(np.square(x) + np.square(y))))) - 0.5) / np.square((1 + 0.001 * (np.square(x) + np.square(y))));

def h_3(x, y):
    return np.square(x) - 10 * np.cos(2 * math.pi * x) + 10 + np.square(y) - 10 * np.cos(2 * math.pi * y) + 10;

def h_4(x, y):
    return np.square(x) - x * y + 6 * x + np.square(y);

def plot(code):
    h = None;
    if code == 1:
        h = h_1;
    elif code == 2:
        h = h_2;
    elif code == 3:
        h = h_3;
    elif code == 4:
        h = h_4;
        
    fig = plt.figure();
    ax = fig.gca(projection='3d');
    x = np.arange(-4, 4, 0.1);
    y = np.arange(-4, 4, 0.1);
    x, y = np.meshgrid(x, y);
    z = h(x, y);
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False);
    ax.set_zlim(-1.01, 1.01);
       
    ax.zaxis.set_major_locator(LinearLocator(10));
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'));
    
    fig.colorbar(surf, shrink=0.5, aspect=5);

    plt.show();

plot(1);