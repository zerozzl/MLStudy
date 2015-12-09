# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 适应度函数
def fitness_fun(x):
    return x * np.sin(10 * np.pi * x) + 2.0;
    
# 染色体
class Genome:
    
    '''
    fitness: 适应度函数
    pos: 位置（此处是横坐标）
    scope: 数据的跨度
    codeLen: 染色体二进制编码长度
    '''
    def __init__(self, fitness, scope, codeLen, pos=None, code=None):
        if pos is not None:
            self.pos = pos;
            self.code = self.float2code(pos, scope, codeLen);
        elif code is not None:
            self.code = code;
            self.pos = self.code2float(self.code, scope, codeLen);
        self.fitness = fitness(self.pos);
        
    # 浮点数转换二进制字符串
    def float2code(self, num, scope, codeLen):
        num = (num + 1.0) * (pow(2.0, codeLen) - 1) / scope;
        code = '';
        while num > 1:
            code = str(int(num % 2)) + code;
            num /= 2;
        if len(code) < codeLen:
            code = '0' * (codeLen - len(code)) + code;
        return code;
    
    # 二进制字符串转换浮点数
    def code2float(self, binstr, scope, codeLen):
        i = 0;
        num = 1.0;
        while i < len(binstr):
            num += int(binstr[i]) * pow(2, len(binstr) - i - 1);
            i += 1;
        num = -1.0 + num * scope / (pow(2.0, codeLen) - 1);
        return num;

# 遗传算法主体
class GenAlg:
    
    '''
    fitness: 适应度函数
    gen_param-[begin, end, len]: 最小值，最大值，二进制编码长度
    scale: 种群规模（数量）
    cross_rate: 交叉生成的后代的比例
    mutate_rate: 突变率
    '''
    def __init__(self, fitness_fun, gen_param, scale, cross_rate, mutate_rate):
        self.fitness = fitness_fun;
        self.genScope = gen_param[1] - gen_param[0]; # 染色体取值范围
        self.genScopeBegin = gen_param[0];
        self.genScopeEnd = gen_param[1];
        self.genCodeLen = gen_param[2] # 染色体二进制编码长度
        self.scale = scale; # 种群规模（数量）
        self.cross_rate = cross_rate; # 交叉率
        self.mutate_rate = mutate_rate; # 突变率
        self.genList = self.initGenomes(); # 初始化染色体
        self.bestFit = 0; # 最优值
        self.bestGen = None; # 最优值对应的染色体
        self.tryTimes = 100; # 达到最大值后尝试的次数，如果达到最大值后再尝试tryTimes次，最大值仍然没有改变，则停止
        self.iterTimes = 0; # 总迭代次数
    
    # 初始化染色体
    def initGenomes(self):
        genList = [];
        datas = np.random.uniform(self.genScopeBegin, self.genScopeEnd, self.scale);
        for data in datas:
            gen = Genome(self.fitness, self.genScope, self.genCodeLen, pos=data);
            genList.append(gen);
        return genList;
    
    # 轮盘赌选取下一代染色体
    def getChromoRoulette(self, genList):
        totalFitness = 0;
        for gen in genList:
            totalFitness += gen.fitness;
        
        scope = np.random.rand() * totalFitness;
        distance = 0.0;
        
        for gen in genList:
            distance += gen.fitness;
            if distance >= scope:
                return gen;

    # 均匀交叉
    def uniformCrossover(self, code1, code2):
        scope = len(code1) - 1;
        cross = np.random.randint(0, scope, np.random.randint(0, scope / 2));
        n_code1 = '';
        n_code2 = '';
        for i in range(len(code1)):
            if i in cross:
                n_code1 += code2[i];
                n_code2 += code1[i];
            else:
                n_code1 += code1[i];
                n_code2 += code2[i];
        return n_code1, n_code2;

    # 染色体变异
    def mutate(self, code):
        pos = np.random.randint(0, len(code) - 1);
        code = list(code)
        if code[pos] == '1':
            code[pos] = '0';
        else:
            code[pos] = '1';
        code = ''.join(code)
        return code;
    
    # 生成下一代染色体
    def competition(self):
        genList_n = []; # 新的染色体
        m = int((1 - self.cross_rate) * self.scale); # 从原来的染色体中选取留下来的数量
        cr = self.scale - m; # 新生成染色体的数量
        mr = int(self.mutate_rate * self.scale) # 变异的数量
        
        # 从原来染色体中挑选留下来的
        for i in range(m):
            gen = self.getChromoRoulette(self.genList);
            self.genList.remove(gen);
            genList_n.append(gen);
            
        # 交叉生成新的染色体
        cr_index = list(np.random.randint(0, len(genList_n), cr));
        while len(cr_index) > 0:
            if len(cr_index) > 1:
                i_1 = cr_index[0];
                i_2 = cr_index[1];
                cr_index.remove(i_1);
                cr_index.remove(i_2);
                code_1, code_2 = self.uniformCrossover(
                                        genList_n[i_1].code, genList_n[i_2].code);
                gen_1 = Genome(self.fitness, self.genScope, self.genCodeLen, code=code_1);
                gen_2 = Genome(self.fitness, self.genScope, self.genCodeLen, code=code_2);
                genList_n.append(gen_1);
                genList_n.append(gen_2);
            else:
                i_1 = cr_index[0];
                cr_index.remove(i_1);
                genList_n.append(genList_n[i_1]);
        
        # 变异
        mr_index = list(np.random.randint(0, len(genList_n), mr));
        for ind in mr_index:
            gen_m = Genome(self.fitness, self.genScope, self.genCodeLen, 
                           code=self.mutate(genList_n[ind].code));
            genList_n[ind] = gen_m;
        
        self.getBestFitness(genList_n);
        self.genList = genList_n;
    
    # 找出当前适应度最大的染色体
    def getBestFitness(self, genList):
        for gen in genList:
            if gen.fitness > self.bestFit:
                self.bestFit = gen.fitness;
                self.bestGen = gen;
    
    #　开始运行
    def start(self):
        # 画出初始图像
        self.plot(self.fitness, self.genScopeBegin, self.genScopeEnd, self.genList);
        
        best = self.bestFit;
        ttimes = self.tryTimes;
        self.iterTimes = 0;
        
        while ttimes > 0:
            self.iterTimes += 1;
            self.competition();
            if self.bestFit > best:
                best = self.bestFit;
                ttimes = self.tryTimes;
            else:
                ttimes -= 1;
        
        print 'Iterater Times: ', self.iterTimes;
        print 'best pos: ', self.bestGen.pos;
        print 'best fitness: ', self.bestFit;
        # 画出最终图像
        self.plot(self.fitness, self.genScopeBegin, self.genScopeEnd, self.genList);
    
    # 画出图像
    def plot(self, fitness, begin, end, genList=None):
        fig = plt.figure();
        ax = fig.add_subplot(111);
        x = np.arange(begin, end, 0.01);
        y = fitness(x);
        
        if genList is not None:
            x_l = [];
            y_l = [];
            for gen in genList:
                x_l.append(gen.pos);
                y_l.append(gen.fitness);
            ax.scatter(x_l, y_l, s=10, c='red');
        
        ax.plot(x, y);
        plt.xlim(-1, 2);
        plt.ylim(0, 4);
        plt.show();

alg = GenAlg(fitness_fun, [-1, 2, 22], 50, 0.2, 0.05);
alg.start();

