# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/17 16:06
# @Author   : Yuan Chuxuan
# @File     : svm.py

import joblib
from sklearn.decomposition import PCA


def SVM_PCA(data):
    # deprecated 特征降维处理
    pca = PCA(n_components=0.95).fit(data)  # n_components设置降维到n维度
    data = pca.transform(data)  # 将规则应用于训练集
    return data


def SVM(crypto_list, features):
    """
    SVM 实现函数，外界调用，load_and_test

    :param args: training args info
    :param features: list of features
    :return: result list and confidence
    """
    dict = {"['3DES', 'AES']": 0.907, "['AES', 'Blowfish']": 0.768, "['3DES', 'Blowfish']": 0.925,
            "['AES', 'RSA']": 0.713,
            "['AES', 'SHA-1']": 0.955, "['3DES', 'RSA']": 0.864, "['DES', 'SHA-1']": 0.961,
            "['Blowfish', 'RSA']": 0.758,
            "['Blowfish', 'SHA-1']": 0.933, "['RSA', 'SHA-1']": 0.957, "['DES', 'RSA', 'SHA-1']": 0.736,
            "['Blowfish', 'RSA', 'SHA-1']": 0.722, "['AES', 'RSA', 'SHA-1']": 0.741,
            "['3DES', 'AES', 'Blowfish']": 0.775,
            "['AES', 'Blowfish', 'SHA-1']": 0.710, "['AES', 'Blowfish', 'RSA']": 0.658, "['3DES', 'AES', 'RSA']": 0.684,
            "['AES', 'DES', 'SHA-1']": 0.686, "['3DES', 'Blowfish', 'RSA']": 0.707,
            "['Blowfish', 'DES', 'SHA-1']": 0.697,
            "['3DES', 'AES', 'Blowfish', 'RSA']": 0.625, "['AES', 'Blowfish', 'DES', 'SHA-1']": 0.548,
            "['AES', 'Blowfish', 'RSA', 'SHA-1']": 0.581,
            "['AES', 'DES', 'RSA', 'SHA-1']": 0.582, "['Blowfish', 'DES', 'RSA', 'SHA-1']": 0.586,
            "['AES', 'Blowfish', 'DES', 'RSA', 'SHA-1']": 0.485}

    directory = './trained_model/SVM/'
    predict = []
    for i in range(0, len(crypto_list)):
        directory += crypto_list[i]
        if i != len(crypto_list) - 1:
            directory += '_'
    directory += '.model'
    model = joblib.load(directory)
    predict.append(model.predict(features))
    return predict, dict[str(crypto_list)]
