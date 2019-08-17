##!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2017/12/2 11:51
# @Author  : 深度眸
# @Email   : 1286304229@qq.com
# @File    : regTrees.py
# Description :CART分类回归树-由于原书采用py2实现，现在切换到py3存在很多要改的地方
# --------------------------------------
from numpy import *


# 将每行映射成浮点数
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(list(fltLine))  # 注意py3 map函数返回的是map对象，而不是list
    return dataMat


# 数据集进行二分
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1


# CART回归树的生成叶子节点函数
def regLeaf(dataSet):  # returns the value used for each leaf
    return mean(dataSet[:, -1])


# 最好特征选择依据：误差-样本y的均方差乘上样本总数=总方差，作为当前样本情况下的混乱程度
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 模型树的叶子节点生成函数
def linearSolve(dataSet):  # helper function used in two places
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  # 线性回归方程
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


# 模型树的误差计算
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 最核心代码-选择最佳特征和阈值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]  # 容许的误差下降值-防止过拟合-预剪枝
    tolN = ops[1]  # 切分的最小样本数-防止过拟合-预剪枝
    # if all the target variables are the same value: quit and return value
    # 如果y取值全部相等，则不需要在划分了，直接返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)  # 计算当前划分下的混乱程度，随着不断的划分，混乱程度应该是下降
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):  # 对每一个特征  n-1 是特性个数
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):  # 对每个特征所对应的特征值，也就是划分阈值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 基于该阈值进行样本二分
            # 不合适的划分方法
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:  # 划分后，混乱程度几乎没有下降，则不划分了
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 正式开始划分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    # 返回特征索引和对应的划分阈值
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


def createTree(dataSet, leafType=regLeaf, errType=regErr,
               ops=(1, 4)):  # assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
    # 停止条件
    if feat == None: return val  # if the splitting hit a stop condition return val

    retTree = {}
    retTree['spInd'] = feat  # 特征索引
    retTree['spVal'] = val  # 特征索引列对应的特征划分阈值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 开始递归
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 从上往下 递归查找两个叶子节点
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 后剪枝操作  待剪枝树  剪枝所需测试数据(一般是验证集)
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 经过N次迭代，找到了叶子节点，开始合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 没有合并前的误差和
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        # 合并后的误差和
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 回归树评估函数
def regTreeEval(model, inDat):
    return float(model)

# 模型树评估函数
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)



def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 测试数据输入的入口函数
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # test1
    # myDat=loadDataSet("ex00.txt")
    # myMat=mat(myDat)
    # print(createTree(myMat))

    # tes2
    # myDat1=loadDataSet("ex0.txt")
    # myMat1=mat(myDat1)
    # print(createTree(myMat1))

    # test3-后剪枝
    # myDat2=loadDataSet("ex2.txt")
    # myMat2=mat(myDat2)
    # tree=createTree(myMat2, ops=(0, 1))
    # print(tree)
    # myDatTest = loadDataSet("ex2test.txt")
    # myMatTest = mat(myDatTest)
    # print(prune(tree,myMatTest))

    # test4-模型树
    # myDat3 = loadDataSet("exp2.txt")
    # myMat3 = mat(myDat3)
    # print(createTree(myMat3, modelLeaf, modelErr, ops=(1, 10)))

    # test5-模型比较
    # 创建回归树
    trainMat=mat(loadDataSet("bikeSpeedVsIq_train.txt"))
    testMat=mat(loadDataSet("bikeSpeedVsIq_test.txt"))
    myTree=createTree(trainMat,ops=(1,20))
    yHat=createForeCast(myTree,testMat[:,0]) # 预测
    print(corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])
    # 创建模型树
    myTree = createTree(trainMat, modelLeaf,modelErr,ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0],modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    # 创建线性回归模型
    ws,X,Y=linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

