# coding: UTF-8
import os
from pyspark import SparkConf
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt

# 画出源数据
def plotDatas(datas):
    fig = plt.figure()
    fig.add_subplot(111);
    plt.imshow(datas)
    plt.show()

os.environ['SPARK_HOME'] = '/apps/spark/spark-1.4.1-bin-hadoop2.6';
 
conf = SparkConf().setAppName('MNIST CNN').setMaster('spark://localhost:7077');
sc = SparkContext(conf=conf);
 
datas = sc.textFile('hdfs://localhost:9000/datas/mnist/test-images.csv');
 
datas = datas.collect();

sc.stop();

