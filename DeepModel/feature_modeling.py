# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:46
# @Author   : JamesYang
# @File     : feature_modeling.py

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
# import minpy.numpy as np
import pandas as pd
import scipy.spatial.distance as dist


def data_preparation(filenames: list, ratio: float):
    """
    only for SFC 网络训练使用本接口进行数据的准备，传入特征文件的列表，将所有向量读取
    using ndarray format

    :param filenames: the file names to using
    :param ratio: split rate
    :return: train's and test's and the length of data
    """
    # 系统会默认第一列是第一个字段而不是index，这样在保存的时候就会凭空多处一列index，使用index_col=0可以避免
    # 如果文件不规则，行尾有分隔符，则可以设定index_col=False 来使得pandas不使用第一列作为行索引
    num = 0
    df = pd.DataFrame()
    for name in filenames:
        num = num + 1
        temp_df = pd.read_csv(name, index_col=0)
        temp_df['target'] = num
        df = df.append(temp_df, ignore_index=True)

    df = df.sample(frac=1.0).reset_index(drop=True)  # 先进行随机抽样，再重置索引

    df = df[['LinearComplexity', 'LongestRun', 'OverlappingTemplate', 'Runs', 'BlockFrequency', 'target']]

    data = np.array(df)
    # shuffled_indices = np.random.permutation(len(data)) 使用numpy进行数据打乱
    train_set_size = int(len(data) * ratio)
    train_indices = data[:train_set_size]
    test_indices = data[train_set_size:]
    test_data, test_target = np.hsplit(test_indices, [5])
    test_label = np.zeros(num)
    for i in test_target:
        temp = np.zeros(num)
        temp[int(i[0]) - 1] = 1
        # test_label = np.append(test_label, temp, axis=1)
        test_label = np.vstack((test_label, temp))
    test_label = test_label[1:train_set_size]
    train_data, train_target = np.hsplit(train_indices, [5])
    train_label = np.zeros(num)
    for i in train_target:
        temp = np.zeros(num)
        temp[int(i[0]) - 1] = 1
        train_label = np.vstack((train_label, temp))
    train_label = train_label[1:train_set_size + 1]
    return train_data, train_label, test_data, test_label


def NN_data_preparation(filenames: list, ratio: float, col_name='data_frame'):
    """
    for other NN 网络训练使用本接口进行数据的准备，传入特征文件的列表，将所有向量读取
    using ndarray format

    :param filenames: the file names to using
    :param ratio: split rate
    :param col_name: column name to extract default:data_frame
    :return: train's and test's
    """
    num = 0
    df = pd.DataFrame()
    for name in filenames:
        num = num + 1
        temp_df = pd.read_csv(name, index_col=0)
        temp_df['target'] = num
        df = df.append(temp_df[[col_name, 'target']], ignore_index=True)

    df = df.sample(frac=1.0).reset_index(drop=True)  # 打乱

    data = np.array(df)
    train_set_size = int(len(data) * ratio)
    train_indices = data[:train_set_size]
    test_indices = data[train_set_size:]
    test_data, test_target = np.hsplit(test_indices, [1])
    test_label = np.zeros(num)
    for i in test_target:
        temp = np.zeros(num)
        temp[int(i[0]) - 1] = 1
        test_label = np.vstack((test_label, temp))
    test_label = test_label[1:train_set_size]
    train_data, train_target = np.hsplit(train_indices, [1])
    train_label = np.zeros(num)
    for i in train_target:
        temp = np.zeros(num)
        temp[int(i[0]) - 1] = 1
        train_label = np.vstack((train_label, temp))
    train_label = train_label[1:train_set_size + 1]

    return train_data, train_label, test_data, test_label


def pre_process(train_imgs, test_imgs, pixels: int):
    bit = int(np.sqrt(pixels))
    d = np.zeros((1, bit, bit))
    for img in train_imgs:
        temp = img.tolist()[0]
        temp = np.array(literal_eval(temp))  # 将字符串格式转化为列表格式
        # temp = np.hstack((temp, np.mean(temp)))
        temp = temp.reshape((1, bit, bit))
        d = np.vstack((d, temp))
    train_data = d[1:train_imgs.shape[0] + 1]

    t = np.zeros((1, bit, bit))
    for img in test_imgs:
        temp = img.tolist()[0]
        temp = np.array(literal_eval(temp))  # 将字符串格式转化为列表格式
        # temp = np.hstack((temp, np.mean(temp)))
        temp = temp.reshape((1, bit, bit))
        t = np.vstack((t, temp))
    test_data = t[1:train_imgs.shape[0] + 1]

    avg = np.mean(train_data)
    dev = np.std(train_data)

    train_data -= avg
    train_data /= dev
    test_data -= avg
    test_data /= dev

    return train_data, test_data


# https://blog.csdn.net/xc_zhou/article/details/81535033
# https://blog.csdn.net/g_persist/article/details/80549377
# 距离介绍
def cosine_similarity_linear(x, y):
    # [-1,1]，值越大，说明夹角越大，两点相距就越远，相似度就越小
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def manhattan_distance(x, y):
    # [0,1]，同欧式距离一致，值越小，说明距离值越大，相似度越大。
    # np.sum(np.abs(x - y))
    return np.linalg.norm(x - y, ord=1)


def chebyshev_distance(x, y):
    return np.linalg.norm(x - y, ord=np.inf)


# 汉明距离的定义
# 两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。
# 例如字符串“1111”与“1001”之间的汉明距离为2。
def hamming_distance(x, y):
    # 向量相似度越高，对应的汉明距离越小
    smstr = np.nonzero(x - y)  # 不为0 的元素的下标
    return np.shape(smstr[0])[0]  # 获取长度


# 杰卡德相似系数
# 两个集合A和B的交集元素在A，B的并集中所占的比例
def jaccard_similarity_coefficient(x, y):
    # [0,1]，完全重叠时为1，无重叠项时为0，越接近1说明越相似。
    matv = np.array([x, y])
    ds = dist.pdist(matv, 'jaccard')  # pdist pairwise distance in nd n维成对距离
    return ds[0]


def compare_vector(file1, file2, fnc):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    result_list = []
    for i in range(1, 193):
        result_list.append(fnc(df1['{}'.format(i - 1)], df2['{}'.format(i - 1)]))
    return np.mean(result_list)


def compare_avg_distance(file1, file2, fnc):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    avg_list_1 = []
    avg_list_2 = []
    for i in range(1, 193):
        avg_list_1.append(np.mean(df1['{}'.format(i - 1)]))
        avg_list_2.append(np.mean(df2['{}'.format(i - 1)]))
    if fnc == None:
        plt.scatter(range(0, 192), avg_list_1)
        plt.savefig('aaa')
        plt.clf()
        plt.scatter(range(0, 192), avg_list_2)
        plt.savefig('bbb')
    else:
        return fnc(np.array(avg_list_1), np.array(avg_list_2))


def main(fnc_name):
    result = []
    for i in range(0, 1500):  # /CryptoClassification
        # result.append(compare_vector('F:/3DES_AES/3DES_res3{}.csv'.format(i),
        #                              'F:/3DES_AES/AES_res3{}.csv'.format(i), fnc_name))
        result.append(compare_avg_distance('F:/3DES_AES/3DES_res3{}.csv'.format(i),
                                           'F:/3DES_AES/AES_res3{}.csv'.format(i), fnc_name))
    index = np.argmax(result)
    index2 = np.argmin(result)
    print("max    " + str(index))
    print(result[index])
    print("min    " + str(index2))
    print(result[index2])


if __name__ == '__main__':
    # main(manhattan_distance)
    compare_avg_distance('../conv_feature_3DES.csv', '../conv_feature_AES.csv', None)
