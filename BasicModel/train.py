# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/17 16:07
# @Author   : Yuan Chuxuan
# @File     : train.py

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC

# setting font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

model_paths = []  # outlook this global para


def display_RF(data: list):
    # draw a fig for rf
    data = np.array(data)
    x_data = data[:, 0]
    y_data = data[:, 1]
    z_data = data[:, 2]
    weight = data[:, 3]
    color = []
    for i in weight:  # setting colors
        if i > 0.93:
            color.append('#DC143C')
        elif i > 0.91:
            color.append('#FF1493')
        elif i > 0.88:
            color.append('#FF69B4')
        elif i > 0.86:
            color.append('#EE82EE')
        elif i > 0.84:
            color.append('#FF00FF')
        elif i > 0.78:
            color.append('#9400D3')
        elif i > 0.74:
            color.append('#6A5ACD')
        elif i > 0.70:
            color.append('#0000CD')
        elif i > 0.66:
            color.append('#4169E1')
        elif i > 0.62:
            color.append('#1E90FF')
        elif i > 0.58:
            color.append('#B0E0E6')
        elif i > 0.54:
            color.append('#7FFFAA')
        elif i > 0.50:
            color.append('#00FF7F')
        elif i > 0.46:
            color.append('#3CB371')
        else:
            color.append('#008000')

    fig = plt.figure()
    ax = Axes3D(fig)  # draw a 3D fig
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.invert_xaxis()
    ax.scatter(x_data, y_data, z_data, c=color, label='parameter adjustment demonstration')

    ax.set_zlabel('md')
    ax.set_ylabel('mid')
    ax.set_xlabel('ne')

    address = '/home/ubuntu/CryptoClassification/verify_result/RF/parameter adjustment demonstration.png'  # attention 固定死的地址
    plt.savefig(address)
    plt.show()
    return address


def RF_conv_test(dataset, ne, mid, md, fea_num, flag, crypto_list):
    """
    新特征 RF参数测试

    :param dataset: 特征数据
    :param ne: n_estimators
    :param mid: min_impurity_decrease
    :param md: max_depth
    :param fea_num: 划分数据集，标签和特征的分界
    :param flag: True 保存模型
    :param crypto_list: 加密算法种类
    :return: 准确率
    """
    feature = []
    target = []
    for tmp in dataset:
        feature.append(tmp[0:fea_num])
        target.append(tmp[fea_num])
    # 分割训练集和测试集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(feature, target, test_size=0.3)

    # 决策树与随机森林
    rfc = RandomForestClassifier(n_estimators=ne, min_impurity_decrease=mid, criterion='gini', max_features='auto',
                                 oob_score=False,
                                 max_depth=md)
    # 训练生成决策树与随机森林
    rfc = rfc.fit(Xtrain, Ytrain)
    # 测试
    feature_imp = rfc.feature_importances_
    score_r = rfc.score(Xtest, Ytest)
    # 保存训练好的模型
    if flag:
        s = ""
        for i in range(0, len(crypto_list)):
            s += crypto_list[i]
            if i != len(crypto_list) - 1:
                s += '_'
        joblib.dump(rfc, '/home/ubuntu/CryptoClassification/self_model/RF/' + s + '.model')
        model_paths.append('/home/ubuntu/CryptoClassification/self_model/RF/' + s + '.model')

    return score_r, feature_imp  # feature_imp is unused


def SVM_test(dataset, gamma_range, c_range, dimension, crypto_list):
    """
    新特征 SVM参数测试

    :param dataset: 特征数据
    :param gamma_range: gamma参数的范围
    :param c_range: 惩罚系数的范围
    :param dimension: 特征维度 Attention（使用了不同维度的特征）
    :param crypto_list: 加密算法种类
    :return: 准确率最高的c,gamma和图片地址
    """
    feature = []
    target = []
    for tmp in dataset:
        feature.append(tmp[0:dimension])
        target.append(tmp[dimension])
    # 分割数据
    train_data, test_data, train_target, test_target = train_test_split(feature, target, test_size=0.3)

    param_dic = dict(gamma=gamma_range, C=c_range)
    estimator = SVC(kernel='rbf')
    # 交叉验证
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    estimate = GridSearchCV(estimator, param_grid=param_dic, cv=cv)  # unused
    # 最佳参数
    fig_data = []
    bg = 0
    bc = 0
    max = 0
    for gamma in gamma_range:
        for c in c_range:
            estimator = SVC(kernel='rbf', gamma=gamma, C=c)
            estimator.fit(train_data, train_target)
            score = estimator.score(test_data, test_target)
            fig_data.append([gamma, c, score])
            if score > max:
                max = score
                bg = gamma
                bc = c
    fig_data = np.array(fig_data)

    x_data = fig_data[:, 0]
    y_data = fig_data[:, 1]
    z_data = fig_data[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_data, y_data, z_data, c='b', label='parameter adjustment demonstration')
    ax.set_zlabel('acc')
    ax.set_ylabel('gamma')
    ax.set_xlabel('c')
    address = '/home/ubuntu/CryptoClassification/verify_result/SVM/parameter adjustment demonstration.png'
    plt.savefig(address)
    plt.show()

    return bg, bc, address


def SVM(dataset, g, c, dimension, crypto_list):
    """
    将最好的训练参数保存（注意重名问题）

    :param dataset: 特征数据
    :param g: same as gamma(best)
    :param c: same as C(best)
    :param dimension: 特征维度
    :param crypto_list: 加密算法种类
    :return: 准确率
    """
    feature = []
    target = []
    for tmp in dataset:
        feature.append(tmp[0:dimension])
        target.append(tmp[dimension])
    # 分割数据
    train_data, test_data, train_target, test_target = train_test_split(feature, target, test_size=0.3)

    estimator = SVC(kernel='rbf', C=c, gamma=g)
    # 开始训练
    estimator.fit(train_data, train_target, sample_weight=[])
    # 预测结果值
    test_target_predict = estimator.predict(test_data)  # unused
    score = estimator.score(test_data, test_target)
    s = ""
    for i in range(0, len(crypto_list)):
        s += crypto_list[i]
        if i != len(crypto_list) - 1:
            s += '_'
    joblib.dump(estimator, '/home/ubuntu/CryptoClassification/self_model/SVM/' + s + '.model')
    model_paths.append('/home/ubuntu/CryptoClassification/self_model/SVM/' + s + '.model')
    return score


def load_data(crypto_list, model_name):
    col = []
    d = 192
    # d为特征维度
    for i in range(1, d + 2):
        col.append(i)
    # 读取特征
    with open('/home/ubuntu/CryptoClassification/static/feature/conv_feature_train.csv') as f:
        df = pd.read_csv(f, usecols=col)
        data = np.array(df).tolist()
    return data, d


def train(model_name, crypto_list):
    """
    机器学习模块的训练接口

    :param model_name: 模型名称
    :param crypto_list: 加密算法的种类
    :return: 准确率，图片地址，模型地址
    """
    fig_paths = ''
    if model_name == 'RF':
        fig_data = []
        bne = 0
        bmid = 0
        bmd = 0
        data, dimension = load_data(crypto_list, 'RF')
        max = 0
        for ne in range(110, 160, 4):
            for mid in np.linspace(0.01, 0.2, 10):
                for md in range(2, 18, 2):
                    tmp_score = RF_conv_test(data, ne, mid, md, dimension, flag=False, crypto_list=crypto_list)[0]
                    fig_data.append([ne, mid, md, tmp_score])
                    if tmp_score > max:
                        max = tmp_score
                        bne = ne
                        bmid = mid
                        bmd = md
        fig_paths = display_RF(fig_data)
        acc = RF_conv_test(data, bne, bmid, bmd, dimension, flag=True, crypto_list=crypto_list)[0]

    elif model_name == 'SVM':
        data, dimension = load_data(crypto_list, 'SVM')
        gamma_range = np.linspace(0.00001, 0.001, 20)
        C_range = np.linspace(1, 10, 20)
        bestg, bestc, fig_paths = SVM_test(data, gamma_range, C_range, dimension=dimension, crypto_list=crypto_list)
        acc = SVM(data, bestg, bestc, dimension, crypto_list)

    return acc, fig_paths, model_paths[0]  # 只有一个值[0]
