# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 适应度函数
def fitness_fun(x):
    return x * np.sin(10 * np.pi * x) + 2.0;

# 粒子
class particle:
    
    def __init__(self, fitness, x, num):
        self.pos = x; # 位置
        self.fitness = fitness(x);
        self.v = [0.0] * len(x); # 速度
        self.selfBestPos = x; # 自己找到的最佳位置
        self.selfBestFit = fitness(x); # 自己找到的最优值
        self.localBestPos = x; # 局部的最佳位置
        self.localBestFit = fitness(x); # 局部的最优值
        self.neighbor = [num]; # 保存局部版本时的邻居
    
    # 全局版本更新位置
    def updatePos_Global(self, fitness, w, c1, c2, r, bestPos, posL, posR):
        self.v = np.dot(w, self.v) + c1 * np.random.rand() * (self.selfBestPos - self.pos) + c2 * np.random.rand() * (bestPos - self.pos);
        self.pos = self.pos + r * self.v;
        
        if self.pos < posL:
            self.pos = posL;
        elif self.pos > posR:
            self.pos = posR;
        
        self.fitness = fitness(self.pos);
        
        if self.fitness > self.selfBestFit:
            self.selfBestPos = self.pos;
            self.selfBestFit = self.fitness;
    
    # 局部版本更新位置
    def updatePos_Local(self, fitness, w, c1, c2, r, posL, posR, particles):
        # 判断是否需要增加邻居
        if len(self.neighbor) < len(particles):
            keys = particles.keys();
            keys = [key for key in keys if key not in self.neighbor];
            pick = keys[np.random.randint(len(keys))];
            self.neighbor.append(pick);
        
        # 确定局部最大值
        for num in self.neighbor:
            par = particles[num];
            if par.fitness > self.localBestFit:
                self.localBestPos = par.pos;
                self.localBestFit = par.fitness;
        
        # 更新速度，位置，自己找到的最优值等信息
        self.v = np.dot(w, self.v) + c1 * np.random.rand() * (self.selfBestPos - self.pos) + c2 * np.random.rand() * (self.localBestPos - self.pos);
        self.pos = self.pos + r * self.v;
        
        if self.pos < posL:
            self.pos = posL;
        elif self.pos > posR:
            self.pos = posR;
        
        self.fitness = fitness(self.pos);
        
        if self.fitness > self.selfBestFit:
            self.selfBestPos = self.pos;
            self.selfBestFit = self.fitness;

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
        self.w = w;
        self.c1 = c1;
        self.c2 = c2;
        self.r = r;
        self.scopeBegin = particle_param[0];
        self.scopeEnd = particle_param[1];
        self.scale = particle_param[2]; # 粒子数量
        self.particles = self.initParticles();
        self.bestPos = np.random.uniform(self.scopeBegin, self.scopeEnd, 1); # 最佳位置
        self.bestFitness = 0.0; # 最佳适应度
        self.tryTimes = 100; # 达到最大值后尝试的次数，如果达到最大值后再尝试tryTimes次，最大值仍然没有改变，则停止
        self.iterTimes = 0; # 总迭代次数
    
    #　初始化粒子群
    def initParticles(self):
        particles = {};
        for i in range(self.scale):
            pos = np.random.uniform(self.scopeBegin, self.scopeEnd, 1);
            particles[i] = particle(self.fitness, pos, i);
        return particles;
    
    # 开始寻找最优解
    def optimal(self, ver):
        best = self.bestFitness;
        ttimes = self.tryTimes;
        self.iterTimes = 0;
        
        while ttimes > 0:
            self.iterTimes += 1;
            
            # 更新每个粒子的位置
            for par in self.particles.values():
                if ver == 1:
                    par.updatePos_Global(self.fitness, self.w, self.c1, self.c2,
                            self.r, self.bestPos, self.scopeBegin, self.scopeEnd);
                elif ver == 2:
                    par.updatePos_Local(self.fitness, self.w, self.c1, self.c2,
                            self.r, self.scopeBegin, self.scopeEnd, self.particles);
                
            # 找出全局最优解
            for par in self.particles.values():
                if par.fitness > self.bestFitness:
                    self.bestPos = par.pos;
                    self.bestFitness = par.fitness;
                
            # 判断是否结束循环
            if self.bestFitness > best:
                best = self.bestFitness;
                ttimes = self.tryTimes;
            else:
                ttimes -= 1;
    
    def start(self, ver):
        # 画出初始图像
        self.plot(self.fitness, self.scopeBegin, self.scopeEnd, self.particles);
        
        self.optimal(ver);
        
        print 'Iterater Times: ', self.iterTimes;
        print 'best pos: ', self.bestPos;
        print 'best fitness: ', self.bestFitness;
        
        # 画出最终图像
        self.plot(self.fitness, self.scopeBegin, self.scopeEnd, self.particles);
        
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

pso = PSO(fitness_fun, 0.5, 2, 2, 1, [-1, 2, 50]);
pso.start(2);

