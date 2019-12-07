"""
============
Barcode Demo
============

This demo shows how to produce a one-dimensional image, or "bar code".
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def loadCSV():
    # df = pd.read_csv('Git-Repository-Miner-Codes.csv')  # 这个会直接默认读取到这个Excel的第一个表单
    df = pd.read_csv('files/alluxio.csv')  # 这个会直接默认读取到这个Excel的第一个表单
    # df = pd.read_csv('java-2018f-homework.csv')  # 这个会直接默认读取到这个Excel的第一个表单
    data = np.array(df.loc[:, :])  # 主要数据，包含统计值
    # ---数据清洗，先归一化
    data = data[:,1:data.shape[1]]

    # for i in range(data.shape[1]):
    #     sum = data[data.shape[0] - 1, i]
    #     print("sum="+str(sum))
    #     for j in range(data.shape[0]-1):
    #         data[j,i] = data[j,i]/sum

    data = data[0:data.shape[0]-1,:]
    column_headers = list(df.columns.values) #标签头，用于索引
    column_headers = column_headers[1:column_headers.__len__()]
    # print(column_headers)
    return data,column_headers

if __name__ == '__main__':

    # Fixing random state for reproducibility
    np.random.seed(19680801)


    # the bar
    # x = np.where(np.random.rand(500) > 0.7, 1.0, 0.0)

    dataSet,dataHeader = loadCSV()
    x1 = dataSet[:,0].astype('int32')
    x2 = dataSet[:,1].astype('int32')
    x3 = dataSet[:,2].astype('int32')
    x4 = dataSet[:,3].astype('int32')
    y = [1,2,3,4,0,0,0,12,1,0,0,0,1,2,3]
    y = np.array(y)
    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')

    fig = plt.figure()



    # a horizontal barcode
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.1], **axprops)
    ax1.imshow(x1.reshape((1, -1)), **barprops)

    ax2 = fig.add_axes([0.1, 0.3, 0.8, 0.1], **axprops)
    ax2.imshow(x2.reshape((1, -1)), **barprops)

    ax3 = fig.add_axes([0.1, 0.5, 0.8, 0.1], **axprops)
    ax3.imshow(x3.reshape((1, -1)), **barprops)

    ax4 = fig.add_axes([0.1, 0.7, 0.8, 0.1], **axprops)
    ax4.imshow(x4.reshape((1, -1)), **barprops)
    plt.savefig('barcode.png')
    plt.show()
    


