import os
import numpy as np
import operator
import csv
from pyspark import SparkConf
from pyspark import SparkContext

def loadTrainSet(filepath, sc):
    datas = sc.textFile(filepath, 30).filter(lambda line : line[0:5] != "label").map(
            lambda line : map(lambda x : int(x), line.split(",")));
    features = datas.map(lambda line : line[1:]);
    labels = datas.map(lambda line : line[0]);
    return features, labels;

def loadTestSet(filepath, sc):
    data = sc.textFile(filepath, 30).filter(lambda line : line[0:6] != "pixel0").map(
            lambda line : map(lambda x : int(x), line.split(",")));
    return data;

# def generateResultFile(filepath, data):
#     with open(filepath, 'wb') as myfile:
#         mywriter = csv.writer(myfile);
#         for i in data:
#             row = [i];
#             mywriter.writerow(row);

# os.environ["SPARK_HOME"] = "/apps/spark/spark-1.4.1-bin-hadoop2.6/";
 
conf = SparkConf().setAppName("Spark Test").setMaster("spark://spnode01:7077");
sc = SparkContext(conf=conf);
 
features, labels = loadTrainSet("hdfs://spnode01:9000/kaggle/DigitRecognizer/train.csv", sc);
  
m = features.count();
k = 5;
  
features = features.collect();
labels = labels.collect();
  
featuresBC = sc.broadcast(features);
labelsBC = sc.broadcast(labels);

testDatas = loadTestSet("hdfs://spnode01:9000/kaggle/DigitRecognizer/test.csv", sc);

testDatas.cache();

result = testDatas.map(lambda line : ((((np.tile(line, (m, 1)) - featuresBC.value) ** 2).sum(axis=1)) ** 0.5)
          .argsort()).map(lambda line : [line[i] for i in range(k)]).map(lambda line : map(lambda x : labelsBC.value[x], line)).map(lambda line : {key : line.count(key) for key in set(line)}).map(lambda line : sorted(line.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]);

# result = result.collect();
# generateResultFile('/home/hadoop/workdatas/kaggle/DigitRecognizer/result_spark.csv', result);

result.repartition(1).saveAsTextFile("hdfs://spnode01:9000/kaggle/DigitRecognizer/result.spark");

sc.stop();


