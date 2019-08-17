##!/usr/bin/python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2017/12/2 9:25
# @Author  : 深度眸
# @Email   : 1286304229@qq.com
# @File    : test1.py
# Description : 标准回归方程求解-最小二乘法求解得到-容易欠拟合  LR有参数学习方法
# 在训练完所有数据之后得到一系列训练参数，然后根据训练参数来预测样本的值这时不再依赖之前的训练数据，参数是确定的
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


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


if __name__ == '__main__':
    xArr,yArr = loadDataSet('ex1.txt')
    ws = standRegres(xArr, yArr)
    xMat=mat(xArr)
    yMat=mat(yArr)
    yHat=xMat*ws
