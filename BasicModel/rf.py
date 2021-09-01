# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/17 16:06
# @Author   : Yuan Chuxuan
# @File     : rf.py

import joblib
from sklearn.decomposition import PCA


def RF_PCA(data, i):
    # deprecated
    feature = []
    for tmp in data:
        size = tmp[i].__len__()
        feature.append(tmp[i][0:size])
    # 特征降维处理
    size = 6
    pca = PCA(n_components=size)  # n_components设置降维到n维度
    feature = pca.fit_transform(feature)  # 将规则应用于训练集
    return feature


def RF(crypto_list, feature):
    """
    implement of Random Forest
    
    :param crypto_list: the target
    :param features: features
    :return: result list and confidence
    """
    dict = {"['3DES', 'AES']": 0.934, "['AES', 'Blowfish']": 0.882, "['3DES', 'Blowfish']": 0.943,
            "['AES', 'RSA']": 0.873,
            "['AES', 'SHA-1']": 0.981, "['3DES', 'RSA']": 0.891, "['3DES', 'SHA-1']": 0.984,
            "['Blowfish', 'RSA']": 0.875,
            "['Blowfish', 'SHA-1']": 0.973, "['RSA', 'SHA-1']": 0.983, "['3DES', 'RSA', 'SHA-1']": 0.751,
            "['Blowfish', 'RSA', 'SHA-1']": 0.744, "['AES', 'RSA', 'SHA-1']": 0.754,
            "['3DES', 'AES', 'Blowfish']": 0.876,
            "['AES', 'Blowfish', 'SHA-1']": 0.710, "['AES', 'Blowfish', 'RSA']": 0.712, "['3DES', 'AES', 'RSA']": 0.743,
            "['AES', 'DES', 'SHA-1']": 0.705, "['3DES', 'Blowfish', 'RSA']": 0.818,
            "['Blowfish', 'DES', 'SHA-1']": 0.721,
            "['3DES', 'Blowfish', 'AES', 'RSA']": 0.721, "['AES', 'Blowfish', 'DES', 'SHA-1']": 0.557,
            "['AES', 'Blowfish', 'RSA', 'SHA-1']": 0.594,
            "['AES', 'DES', 'RSA', 'SHA-1']": 0.586, "['Blowfish', 'DES', 'RSA', 'SHA-1']": 0.598,
            "['AES', 'Blowfish', 'DES', 'RSA', 'SHA-1']": 0.489}
    x = []  # 记录每条密文的预测的结果
    directory = './trained_model/RF/'
    col = []
    for i in range(0, 192):
        col.append(str(i))
    for i in range(0, len(crypto_list)):  # create the path of model
        directory += crypto_list[i]
        if i != len(crypto_list) - 1:
            directory += '_'

    directory += '.model'
    model = joblib.load(directory)
    x.append(model.predict(feature))

    return x, dict[str(crypto_list)]
