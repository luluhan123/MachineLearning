from math import log
import numpy as np
from treePlotter import getNumLeafs,getTreeDepth,createPlot

# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    # 为所有分类创建字典
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)

    return shannonEnt

# 计算基尼指数
def calcGini(dataSet):
    numEntries = len(dataSet)
    # 为所有分类创建字典
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
    Gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Gini -= np.square(prob)
    return Gini

# 按照给定特征划分数据集
# 获取第axis个特征值为value的数据
# 返回数据不包括第axis个特征
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择则好的数据集划分方式
def chooseBestFeatureToSPLIT(dataSet):
    # 特征数
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInforGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i]  for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoEntropy = baseEntropy - newEntropy

        if(infoEntropy > bestInforGain):
            bestInforGain = infoEntropy
            bestFeature = i
    return bestFeature

#多数表决的方法决定叶子节点的分类 ----  当所有的特征全部用完时仍属于多类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = np.operator.itemgetter(1), reverse = True)  #排序函数 operator中的
    return sortedClassCount[0][0]


# 创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征是返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSPLIT(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree)[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

# 产生测试数据
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

if __name__=="__main__":
    myDat, labels = createDataSet()
    #print(myDat)
    myTree = createTree(myDat, labels)
    #createPlot(myTree)
    #classify(myTree,labels,[1,0])
    storeTree(myTree,'classifierStorge.txt')
    print(grabTree('classifierStorge.txt'))