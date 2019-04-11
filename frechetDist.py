# -*- coding: utf-8 -*-

"""
Fréchet distance
"""
import math
import numpy as np
import pandas as pd
import time
import csv

# Euclidean distance.
def euc_dist(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))

def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

def dis(ca,x,y,P,Q):
    for i in range(x):
        for j in range(y):
            if ca[i, j] > -1:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = euc_dist(P[0], Q[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i-1,0], euc_dist(P[i], Q[0]))
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0,j-1], euc_dist(P[0], Q[j]))
            elif i > 0 and j > 0:
                A = ca[i-1,j];
                B = ca[i-1,j-1];
                C = ca[i,j-1];
                ca[i, j] = max(min(A, B, C),
                               euc_dist(P[i], Q[j]))
            else:
                ca[i, j] = float("inf")
    return ca[i,j]

"""
Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""
def frechetDist(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    # print(ca)
    # return _c(ca,len(P)-1,len(Q)-1,P,Q)
    return dis(ca,len(P)-1,len(Q)-1,P,Q)


def loadCSV(filename,isNormalize=True):
    df = pd.read_csv(filename)  # 这个会直接默认读取到这个Excel的第一个表单
    data = np.array(df.loc[:, :])  # 主要数据，包含统计值
    # ---数据清洗，先归一化
    data = data[:,1:data.shape[1]]

    if(isNormalize):
        for i in range(data.shape[1]):
            sum = data[data.shape[0] - 1, i]
            # print("sum="+str(sum))
            for j in range(data.shape[0]-1):
                data[j,i] = data[j,i]/sum
                if(isNormalize):
                    data[j,i] = data[j,i] * 1000
    data = data[0:data.shape[0]-1,:]
    column_headers = list(df.columns.values) #标签头，用于索引
    column_headers = column_headers[1:column_headers.__len__()]
    # print(column_headers)
    return data,column_headers


def getDistanceToAll(filename,savename,isNormalize=True,isDivide=True):
    print("计算各个人和总体之间的FreChet距离")
    if(isNormalize):
        print("--------------有归一化：")
    else:
        print("--------------没有有归一化：")
    dataSet, dataHeader = loadCSV(filename,isNormalize)
    author_cnts = dataHeader.__len__() - 1
    disMatrix = np.zeros([1, author_cnts])
    standard = dataSet[:, 0]
    PQ_List = []
    for i in range(dataHeader.__len__()):
        P = []
        for j in range(dataSet.shape[0]):
            P.append((j, dataSet[j, i]))
        PQ_List.append(P)
    print("一共有" + str(author_cnts) + "个Commit次数超过100的开发者")
    for i in range(author_cnts):
        print("处理第" + str(i) + "个author:" + dataHeader[i + 1])
        disMatrix[0, i] = frechetDist(PQ_List[0], PQ_List[i + 1])
        if isDivide:
            if disMatrix[0, i] != 0:
                disMatrix[0, i] = 1/disMatrix[0, i]
            else:
                disMatrix[0, i] = 1
        print(dataHeader[0] + " Vs " + dataHeader[i + 1] + ",相似距离（值越小，越相似）为：" + str(disMatrix[0, i]))

    # 将结果写入CSV
    out = open(savename, 'w', newline='')
    # 设定写入模式
    csv_write = csv.writer(out, dialect='excel')
    # 写入具体内容
    header = dataHeader.copy()
    # header.insert(0," ")
    header[0] = " "
    csv_write.writerow(header)
    output = disMatrix[0].tolist()
    output.insert(0, "All")
    csv_write.writerow(output)

    out.close()

def getDistanceMatrix(filename,savename,isNormalize=True,isDivide=True):

    print("计算"+filename+"的FreChet距离")
    if(isNormalize):
        print("--------------有归一化：")
    else:
        print("--------------没有有归一化：")

    # 读取数据
    dataSet, dataHeader = loadCSV(filename,isNormalize)
    # 开发者数目（排除All）
    author_cnts = dataHeader.__len__() - 1
    print("一共有" + str(author_cnts) + "个Commit次数超过100的开发者")

    # 将结果写入CSV
    out = open(savename, 'w', newline='')
    # 设定写入模式
    csv_write = csv.writer(out, dialect='excel')
    # 写入具体内容
    header = dataHeader.copy()
    header[0] = " "
    csv_write.writerow(header)

    #输出的距离矩阵
    output = np.zeros((dataHeader.__len__(), dataHeader.__len__()))
    for i in range(dataHeader.__len__()):
        P = []  # 参照物
        for j in range(dataSet.shape[0]):
            P.append((j, dataSet[j, i]))
        for j in range(i + 1, dataHeader.__len__()):
            Q = []
            for k in range(dataSet.shape[0]):
                Q.append((k, dataSet[k, j]))
            res = frechetDist(P, Q)
            '''这里做了倒数处理'''
            if isDivide:
                if (res == 0):
                    res = 0
                else:
                    res = 1 / res
            print(dataHeader[i] + " Vs " + dataHeader[j] + ",相似距离（值越小，越相似）为：" + str(res))
            output[i, j] = res
            output[j, i] = res
    output = output.tolist()
    for i in range(1, dataHeader.__len__()):
        output[i][0] = dataHeader[i]
        csv_write.writerow(output[i])
    out.close()



if __name__ == '__main__':
    start = time.time()  # 计算程序运行时间

    '''按照周为单位'''
    filename = 'files/alluxio.csv'
    outname = 'outcomes/alluxio/alluxio'
    getDistanceToAll(filename,outname+'OvR_Normal_Divide.csv',True,True)
    getDistanceToAll(filename,outname+'OvR_Divide.csv',False,True) #不Normal是为了得到绝对距离（归一后相似度高 && 归一前值也高）
    getDistanceMatrix(filename,outname+'RvR_Normal_Divide.csv',True,True)
    getDistanceMatrix(filename,outname+'RvR_Divide.csv',False,True) #不Normal是为了得到绝对距离（归一后相似度高 && 归一前值也高）

    '''按照天为单位'''
    filename = 'files/alluxio_original.csv'
    outname = 'outcomes/alluxio_original/alluxio_original'
    getDistanceToAll(filename,outname+'OvR_Normal_Divide.csv',True,True)
    getDistanceToAll(filename,outname+'OvR_Divide.csv',False,True) #不Normal是为了得到绝对距离（归一后相似度高 && 归一前值也高）
    getDistanceMatrix(filename,outname+'RvR_Normal_Divide.csv',True,True)
    getDistanceMatrix(filename,outname+'RvR_Divide.csv',False,True) #不Normal是为了得到绝对距离（归一后相似度高 && 归一前值也高）

    end = time.time()
    print("运行时间：" + str(end - start) + "s")
