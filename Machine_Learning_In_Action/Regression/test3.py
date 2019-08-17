# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2017/12/2 9:55
# @Author  : 深度眸
# @Email   : 1286304229@qq.com
# @File    : test3.py
# Description : 岭回归-解决过拟合问题
# 在增加约束时，普通的最小二乘法回归会得到与岭回归的一样的公式，也就是说在增加约束情况下，最少二乘法可以转化为岭回归
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

def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# 注意：一般都需要进行归一化，去均值和1方差处理，因为L2范数，如果W不在一个量级，会出现问题
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

if __name__ == '__main__':
    xArr, yArr = loadDataSet('abalone.txt')
    print(ridgeTest(xArr,yArr))
