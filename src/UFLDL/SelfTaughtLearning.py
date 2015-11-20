#coding: UTF-8
import struct
import numpy as np

def main():
    inputSize  = 28 * 28;
    numLabels  = 5;
    hiddenSize = 200;
    sparsityParam = 0.1;
    lamb = 3e-3;
    beta = 3;
