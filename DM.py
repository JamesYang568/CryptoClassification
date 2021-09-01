# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 17:48
# @Author   : JamesYang
# @File     : DM.py

import argparse

import pandas as pd

from DeepModel import *


def parseArgs():
    """
    if you want to run the demo, please define a function in main using argparse
    """
    parser = argparse.ArgumentParser(description='DeepLearning Model For ciphertext classification')
    parser.add_argument('--crypto_list', type=list, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--input_channel', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--ratio', type=float, default=0.8, metavar='N',
                        help='input batch size for training (default: 0.8)')
    parser.add_argument('--epoch', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 30)')
    parser.add_argument('--loss_function', type=str, default='MES', metavar='N',
                        help='input batch size for training (default: MES)')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='N',
                        help='input batch size for training (default: SGD)')
    parser.add_argument('--col_name', type=str, default='F_1024b', metavar='N',
                        help='input batch size for training (default: F_1024b)')
    parser.add_argument('--save_mode', type=bool, default=False, metavar='N',
                        help='input batch size for training (default: False)')

    return parser.parse_args()


def get_feature(file_list: list, crypto_name: str):
    """
    传入一个个密文原文地址和对应的加密算法名字。 \n
    Attention,本函数有两个入口，在train_and_test中，传入加密算法名，在load_and_test中，传入classify

    :return: 一个ones特征提取文件的路径
    """
    feature_list = []
    feature_file_dir = './static/feature/' + crypto_name + '_ones.csv'
    for file_dir in file_list:
        feature_list.append(bitwise_8(file_dir))
    df = pd.DataFrame({'data_frame': feature_list})
    df.to_csv(feature_file_dir)
    return feature_file_dir  # 特征文件ones的路径


def load_and_test(model_name: str, file_list: list, args):
    """
    use the pre-trained model

    :param model_name: the name of the model used.
    :param file_list: the directory of feature file
    :param args: basic args
    :return: result dict
    """
    feature_file_dir = get_feature(file_list, 'classify')  # feature文件夹下面生成classify文件
    # 读模型
    if args.save_mode == 'para':
        if model_name == 'SFC_basic':
            model = SFCNet(args.input_channel, args.category)
        elif model_name == 'SFC_advance':
            model = Improve_SFCNet(args.input_channel, args.category)
        elif model_name == 'SCNN':
            model = LeNet5(args.category)
        elif model_name == 'ResNet':
            model = Resnet_improve(args.category)
        elif model_name == 'VGG':
            model = VGG_11(args.category)
        elif model_name == 'AlexNet':
            model = AlexNet(args.category, True)
        elif model_name == 'ResNet50':
            model = ResNet50(args.category)
        else:
            model = ResNet101(args.category)
        model.load_state_dict(torch.load('trained_model/' + model_name + '/' + model_name + '_' + concat2str(
            args.crypto_list) + '_parameter.pkl'))

    else:
        model = torch.load(
            'trained_model/' + model_name + '/' + model_name + '_' + concat2str(args.crypto_list) + '_model.pkl')

    # 获取数据，测试
    if model_name == 'SFC_basic' or model_name == 'SFC_advance':
        test_data = data_loader(filenames=[feature_file_dir], model_name=model_name, col_name=args.col_name,
                                input_channel=args.input_channel)
        result = verify(model, test_data, len(test_data))
    else:
        test_loader = data_loader(filenames=[feature_file_dir], model_name=model_name, col_name=args.col_name,
                                  input_channel=args.input_channel)
        result = batch_verify(model, test_loader)

    final_result = []
    confidence = []
    cl = concat(args.crypto_list)
    for i in result:
        w, r = get_whose(cl, i)
        final_result.append(w)
        confidence.append(r)

    return final_result, confidence


def train_and_test(model_name, crypto_file_lists: dict, args):
    """
    train and test the model

    :param model_name: the name of the model used.
    :param crypto_file_lists: 一个加密算法有一个list全部是文件
    :param args: args used to train the model
    :return: result dict
    """
    # 做特征
    feature_file_dirs = []
    for tmp in args.crypto_list:
        feature_file_dirs.append(get_feature(crypto_file_lists[tmp], tmp))
    # train 得到训练的准确率，模型名称（路径），训练的曲线
    if model_name == 'SFC_basic':  # 需要的是对应的路径
        pic_dir, model_dir, tr_accuracy = SFCN(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                               args.epoch, args.batch, args.loss_function, args.optimizer,
                                               args.save_mode)
    elif model_name == 'SFC_advance':
        pic_dir, model_dir, tr_accuracy = SFCN(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                               args.epoch, args.batch, args.loss_function, args.optimizer,
                                               mode='advance', save_mode=args.save_mode)
    elif model_name == 'SCNN':
        pic_dir, model_dir, tr_accuracy = SCNN(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                               args.epoch, args.batch, args.loss_function, args.optimizer,
                                               args.col_name, args.save_mode)
    elif model_name == 'ResNet':
        pic_dir, model_dir, tr_accuracy = ResNet_18(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                                    args.epoch, args.batch, args.loss_function, args.optimizer,
                                                    args.col_name, args.save_mode)
    elif model_name == 'VGG':
        pic_dir, model_dir, tr_accuracy = VGG(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                              args.epoch, args.batch, args.loss_function, args.optimizer,
                                              args.col_name, args.save_mode)
    elif model_name == 'AlexNet':
        pic_dir, model_dir, tr_accuracy = AlexCNN(args.crypto_list, feature_file_dirs, args.input_channel, args.ratio,
                                                  args.epoch, args.batch, args.loss_function, args.optimizer,
                                                  args.col_name, args.save_mode)
    else:
        pic_dir, model_dir, tr_accuracy = ResNet_deeper(args.crypto_list, feature_file_dirs, args.input_channel,
                                                        model_name, args.ratio, args.epoch, args.batch,
                                                        args.loss_function, args.optimizer, args.col_name,
                                                        args.save_mode)

    return './verify_result/' + model_name + '/' + pic_dir, \
           './self_model/' + model_name + '/' + model_dir, \
           tr_accuracy / 100

# for routine please see backend.py
