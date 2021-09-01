# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/17 16:04
# @Author   : Yuan Chuxuan
# @File     : BM.py

import DM
from BasicModel import *

'''
此代码的注释是由袁楚轩完成的，有任何问题，请联系本人
'''


def get_feature(file_list: list, crypto_list: list, fuc: str, model_name):
    """
    创建特征文件
    用CnPo特征进行分类

    :param file_list: ones文件的地址
    :param crypto_list: 加密算法种类
    :param model_name: 模型名称
    :return: 特征的list[]
    """
    conv_feature = get_conv_feature(file_list, crypto_list, 192, fuc=fuc)
    return conv_feature


def load_and_test(file_list: list, model_name, args):
    """
     use the pre-trained model

    :param file_list: the directory of feature file
    :param model_name: the name of the model used.
    :param args: basic args
    :return: result list
    """
    # 访问特征
    ans = []
    confidence = []
    conv_feature_list = []
    # 利用网络获得ones特征
    ones_file_dir = DM.get_feature(file_list, 'classify')
    # 分类
    if model_name == 'RF':
        # 获得卷积特征
        conv_feature_list = get_feature([ones_file_dir], args.crypto_list, fuc='classify', model_name='RF')
        ans, confidence = RF(args.crypto_list, conv_feature_list)
    elif model_name == 'SVM':
        # 获得卷积特征
        conv_feature_list = get_feature([ones_file_dir], args.crypto_list, fuc='classify', model_name='SVM')
        ans, confidence = SVM(args.crypto_list, conv_feature_list)
    result = []
    ans = ans[0].tolist()
    for i in range(0, len(conv_feature_list)):  # make decision
        for j in range(0, 4):
            if ans[i] == j:
                result.append(args.crypto_list[j] + '加密')

    return result, confidence


def train_and_test(crypto_file_list: dict, model_name, args):
    """
    train and test the model

    :param crypto_file_list: 一个加密算法有一个list全部是文件
    :param model_name: the name of the model used.
    :param args: args used to train the model
    :return: acc, file_path, model_path
    """
    # 得到ones文件
    ones_file_dirs = []
    for tmp in args.crypto_list:
        ones_file_dirs.append(DM.get_feature(crypto_file_list[tmp], tmp))
    # 获得特征列表
    get_feature(ones_file_dirs, args.crypto_list, fuc='train', model_name=model_name)  # back  involve without
    # 训练
    acc, file_path, model_path = train(model_name, args.crypto_list)

    return acc * 100, file_path, model_path

# for routine please see backend.py
