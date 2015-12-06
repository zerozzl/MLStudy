import numpy as np
import matplotlib.pyplot as plt
from __builtin__ import str

def fitness_fun(x):
    return x * np.sin(10 * np.pi * x) + 2.0;

def float2code(num, begin, end):
    num = (num + 1.0) * (pow(2.0, 22) - 1) / (end - begin);
    code = '';
    while num > 1:
        code = str(int(num % 2)) + code;
        num /= 2;
    if len(code) < 22:
        code = '0' * (22 - len(code)) + code;
    return code;
 
def code2float(binstr, begin, end):
    i = 0;
    num = 1.0;
    while i < len(binstr):
        num += int(binstr[i]) * pow(2, len(binstr) - i - 1);
        i += 1;
    num = -1.0 + num * (end - begin) / (pow(2.0, 22) - 1);
    return num;
    
def plot():
    fig = plt.figure();
    ax = fig.add_subplot(111);
    x = np.arange(-1, 2, 0.01);
    y = fitness_fun(x);
    ax.plot(x, y);
    plt.show();
    
class Genome:
     
    def __init__(self, fitness, pos, begin, end):
        self.code = float2code(pos, begin, end);
        self.pos = code2float(self.code, begin, end);
        self.fitness = fitness(self.pos);

class GenAlg:
    
    def __init__(self, fitness, initParam):
        self.genList = [];
        gen1 = Genome(fitness, 5, -1, 2);
        gen2 = Genome(fitness, 7, -1, 2);
        gen3 = Genome(fitness, 10, -1, 2);
        gen4 = Genome(fitness, 13, -1, 2);
        gen5 = Genome(fitness, 15, -1, 2);
        gen1.fitness = 5;
        gen2.fitness = 7;
        gen3.fitness = 10;
        gen4.fitness = 13;
        gen5.fitness = 15;
        self.genList.append(gen1);
        self.genList.append(gen2);
        self.genList.append(gen3);
        self.genList.append(gen4);
        self.genList.append(gen5);
        print self.getChromoRoulette(self.genList).pos;
        
    
    def getChromoRoulette(self, genList):
        totalFitness = 0;
        for gen in genList:
            totalFitness += gen.fitness;
        
        slice = np.random.rand() * totalFitness;
        distance = 0.0;
        
        for gen in genList:
            distance += gen.fitness;
            if distance >= slice:
                return gen;

alg = GenAlg(fitness_fun, [-1, 2, 10]);
