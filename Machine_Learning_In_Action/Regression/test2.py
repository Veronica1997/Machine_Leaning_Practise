# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2017/12/2 9:38
# @Author  : 深度眸
# @Email   : 1286304229@qq.com
# @File    : test2.py
# Description :局部加权线性回归模型-解决线性回归中存在的欠拟合问题    LWLR非参数学习方法
# 在预测新样本值时每次都会重新训练新的参数，也就是每次预测新的样本值都会依赖训练数据集合，所以每次的参数是不确定的

# 对每个点做预测时都必须使用整个数据集，因此当训练容量过大时，非参数学习算法需要占用更多的存储容量，计算速度较慢
# 对于样本的预测，其实就是对每个点的预测（或者说拟合），样本点有m个点，则预测出来的也是m个点，而这m个点不在一条直线上，
# 而是离散的点，因此最终拟合的'线性'（实为非线性）曲线则是m个点的（m-1条线段）组成的估计曲线
# http://blog.csdn.net/qq_28031525/article/details/60966234
# --------------------------------------
from numpy import *

def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 测试局部加权线性回归
def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def lwlrTestPlot(xArr, yArr, k=1.0):  # same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))  # easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    # 单点预测
    print(lwlr(xArr[0],xArr,yArr,1.0))
    # 全部数据点测试
    print(lwlrTest(xArr,xArr,yArr,0.001))
