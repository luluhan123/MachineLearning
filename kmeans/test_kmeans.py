from numpy import *
import time
import matplotlib.pyplot as plt
from kmeans import kmeans,showCluster

## step 1: load data
dataSet = [[0,0],[3,8],[2,2],[1,1],[5,3],[4,8],[6,3],[5,4],[6,4],[7,5]]

## step 2: clustering...
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = kmeans(dataSet, k)
print('loss:',sum(clusterAssment[:,1]))

## step 3: show the result
showCluster(dataSet, k, centroids, clusterAssment)
