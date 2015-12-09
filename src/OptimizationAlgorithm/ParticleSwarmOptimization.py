# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 适应度函数
def fitness_fun(x):
    return x * np.sin(10 * np.pi * x) + 2.0;

# 粒子
class particle:
    
    def __init__(self, fitness, x):
        self.pos = x; # 位置
        self.v = [0.0] * len(x); # 速度
        self.fitness = fitness(x);
        self.selfBest = 0.0; # 自己找到的最佳位置
        self.globalBest = 0.0; # 全局或局部的最佳位置

# 粒子群算法主体
class PSO:
    
    '''
    w: 原来速度的系数，所以叫做惯性权重
    c1: 粒子跟踪自己历史最优值的权重系数，它表示粒子自身的认识，所以叫"认知"。
    c2: 粒子跟踪群体最优值的权重系数，它表示粒子对整个群体知识的认识，所以叫做"社会知识"
    r: 对位置更新的时候，在速度前面加的一个系数，这个系数我们叫做约束因子
    particle_param-[begin, end, scale]: 最小值，最大值，规模
    '''
    def __init__(self, fitness_fun, w, c1, c2, r, particle_param):
        self.fitness = fitness_fun; # 适应度函数
        self.scopeBegin = particle_param[0];
        self.scopeEnd = particle_param[1];
        self.scale = particle_param[2]; # 粒子数量
        self.particles = self.initParticles();
        self.bestPos = None; # 最佳位置
        self.bestFitness = 0.0; # 最佳适应度
        self.tryTimes = 100; # 达到最大值后尝试的次数，如果达到最大值后再尝试tryTimes次，最大值仍然没有改变，则停止
        self.iterTimes = 0; # 总迭代次数
    
    def initParticles(self):
        particles = {};
        for i in range(self.scale):
            pos = np.random.uniform(self.scopeBegin, self.scopeEnd, 1);
            particles[i] = particle(self.fitness, pos);
        return particles;
    
    def global_version(self):
        pass;
    
    def start(self):
#         self.plot(self.fitness, self.scopeBegin, self.scopeEnd, self.particles);
        self.global_version();
        return;
        
    # 画出图像
    def plot(self, fitness, begin, end, parMap=None):
        fig = plt.figure();
        ax = fig.add_subplot(111);
        x = np.arange(begin, end, 0.01);
        y = fitness(x);
        
        if parMap is not None:
            x_l = [];
            y_l = [];
            for par in parMap.values():
                x_l.append(par.pos);
                y_l.append(par.fitness);
            ax.scatter(x_l, y_l, s=10, c='red');
        
        ax.plot(x, y);
        plt.xlim(-1, 2);
        plt.ylim(0, 4);
        plt.show();

pso = PSO(fitness_fun, 2, 2, 2, 1, [-1, 2, 50]);
pso.start();
