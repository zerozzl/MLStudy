# -*- coding:utf-8 -*-
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm  
from matplotlib.ticker import LinearLocator, FormatStrFormatter  
import matplotlib.pyplot as plt  

def chooseFun(code):
    if code == 1:
        return

def h_1(x, y):
    return np.sin(np.sqrt(np.square(x) + np.square(y))) / np.sqrt(np.square(x) + np.square(y)) + np.exp((np.cos(2 * math.pi * x) + np.cos(2 * math.pi * y)) / 2) - 2.71289;

fig = plt.figure()  
ax = fig.gca(projection='3d')  
X = np.arange(-4, 4, 0.1)  
Y = np.arange(-4, 4, 0.1)
X, Y = np.meshgrid(X, Y)  
Z = h_1(X, Y);  
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)  
ax.set_zlim(-1.01, 1.01)  
   
ax.zaxis.set_major_locator(LinearLocator(10))  
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  
   
fig.colorbar(surf, shrink=0.5, aspect=5)  
   
plt.show() 

'''
function f = fun(X)
x = X(1);
y = X(2);

choose = 1;

switch choose
    case 2
        f_schaffer = 0.5-((sin(sqrt(x.^2+y.^2))).^2-0.5)./(1+0.001*(x.^2+y.^2)).^2; # x[-4, 4]
        f = f_schaffer;
    case 3
        f_rastrigin = x.^2-10*cos(2*pi*x)+10 + y.^2-10*cos(2*pi*y)+10; # x[-4, 4]
        f = -f_rastrigin; 
end

y = x1^2 - x1*x2 + 6 * x1 + x2^2;
'''
