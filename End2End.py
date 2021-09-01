# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/7/22 9:49
# @Author   : Yang Jiaxiong
# @File     : End2End.py

from BasicModel.Crypto_Embedding import *
from DM import get_feature
from DeepModel import data_loader


def train_CNN_SVM(crypto_file_lists: dict, args):
    """
    用户训练CNN-SVM模型
    """
    feature_file_dirs = []
    for tmp in args.crypto_list:
        feature_file_dirs.append(get_feature(crypto_file_lists[tmp], tmp))
    embedding, label, cnn_model_path = train_cnn(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                                 args.epoch, args.batch, args.loss_function, args.optimizer,
                                                 args.col_name, False)
    acc, svm_model_path = train_svm(args.crypto_list, embedding, label)
    return acc, cnn_model_path, svm_model_path


def test_CNN_SVM(file_list: list, args):
    """
    用户使用，根据已有的CNN-SVM模型实现密文分类
    :param file_list: file_lists to classify
    :param args: args
    :return: 分类结果和置信度
    """

    confident_dict = {"['3DES', 'AES']": 0.954, "['AES', 'Blowfish']": 0.966, "['3DES', 'Blowfish']": 0.971,
                      "['AES', 'RSA']": 0.874,
                      "['3DES', 'RSA']": 0.969,
                      "['Blowfish', 'RSA']": 0.968,
                      "['3DES', 'AES', 'Blowfish']": 0.947,
                      "['AES', 'Blowfish', 'RSA']": 0.919,
                      "['3DES', 'AES', 'RSA']": 0.896,
                      "['3DES', 'Blowfish', 'RSA']": 0.950,
                      "['3DES', 'AES', 'Blowfish', 'RSA']": 0.888}

    feature_file_dir = get_feature(file_list, 'classify')  # feature文件夹下面生成classify文件
    test_loader = data_loader(filenames=[feature_file_dir], model_name='CNN-SVM', col_name=args.col_name,
                              input_channel=args.input_channel)
    ans = CNN_SVM(args.crypto_list, test_loader)
    result = []
    ans = ans.tolist()
    for i in range(0, len(ans)):  # make decision
        for j in range(0, 4):
            if ans[i] == j:
                result.append(args.crypto_list[j])

    return result, confident_dict[str(args.crypto_list)]


def train_CNN_RF(crypto_file_lists: dict, args):
    """
    用户训练CNN-RF模型
    """
    feature_file_dirs = []
    for tmp in args.crypto_list:
        feature_file_dirs.append(get_feature(crypto_file_lists[tmp], tmp))
    embedding, label, cnn_model_path = train_cnn(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                                 args.epoch, args.batch, args.loss_function, args.optimizer,
                                                 args.col_name, False, type='CNN-RF')
    acc, rf_model_path = train_rf(args.crypto_list, embedding, label)
    return acc, cnn_model_path, rf_model_path


def test_CNN_RF(file_list: list, args):
    """
    用户使用，根据已有的CNN-RF模型实现密文分类
    :param file_list: file_lists to classify
    :param args: args
    :return: 分类结果和置信度
    """
    confident_dict = {"['3DES', 'AES']": 0.961, "['AES', 'Blowfish']": 0.959, "['3DES', 'Blowfish']": 0.968,
                      "['AES', 'RSA']": 0.901,
                      "['3DES', 'RSA']": 0.952,
                      "['Blowfish', 'RSA']": 0.964,
                      "['3DES', 'AES', 'Blowfish']": 0.951,
                      "['AES', 'Blowfish', 'RSA']": 0.892,
                      "['3DES', 'AES', 'RSA']": 0.889,
                      "['3DES', 'Blowfish', 'RSA']": 0.952,
                      "['3DES', 'AES', 'Blowfish', 'RSA']": 0.905}

    feature_file_dir = get_feature(file_list, 'classify')  # feature文件夹下面生成classify文件
    test_loader = data_loader(filenames=[feature_file_dir], model_name='CNN-RF', col_name=args.col_name,
                              input_channel=args.input_channel)
    ans = CNN_RF(args.crypto_list, test_loader)
    result = []
    ans = ans.tolist()
    for i in range(0, len(ans)):  # make decision
        for j in range(0, 4):
            if ans[i] == j:
                result.append(args.crypto_list[j])

    return result, confident_dict[str(args.crypto_list)]

# for routine please see demo.py and backend.py
