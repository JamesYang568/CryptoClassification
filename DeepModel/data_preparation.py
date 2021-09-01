# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/4/3 16:14
# @Author   : JamesYang
# @File     : data_preparation.py

import pandas as pd
import numpy as np
from torch.autograd import Variable
from ast import literal_eval
import torch
from DeepModel.NNutil import get_batch


def data_loader(filenames: list, model_name: str, col_name: list, input_channel=1024, batch=300):
    df = pd.DataFrame()
    pos = []  # 分界的位置
    for name in filenames:
        temp_df = pd.read_csv(name, index_col=0)
        pos.append(len(temp_df))
        df = df.append(temp_df, ignore_index=True)
    # df = df.sample(frac=1.0).reset_index(drop=True) 直接验证不能将顺序打散
    df = df[col_name]
    data = np.array(df)

    if model_name == 'SFC' or model_name == 'SFC_advance':
        data = pre_process(data, input_channel)
        data = Variable(torch.from_numpy(data).float())
        return data
    elif model_name == '':
        label = np.empty(0)
        data = batch_pre_process(data, input_channel)
        for i in range(len(filenames)):
            label = np.concatenate((label, np.full(pos[i], i)), axis=0)
        data = Variable(torch.from_numpy(data).float())
        label = Variable(torch.from_numpy(label).float())
        loader = get_batch(data, label, batch, False)
        return loader
    else:
        data = batch_pre_process(data, input_channel)
        data = Variable(torch.from_numpy(data).float())
        loader = get_batch(data, None, batch, False)
        return loader


def batch_pre_process(data, pixels: int):
    bit = int(np.sqrt(pixels))
    d = np.zeros((1, bit, bit))
    for img in data:
        temp = img.tolist()[0]
        temp = np.array(literal_eval(temp))  # 将字符串格式转化为列表格式
        temp = temp.reshape((1, bit, bit))
        d = np.vstack((d, temp))
    data = d[1:data.shape[0] + 1]
    avg = np.mean(data)
    dev = np.std(data)
    data -= avg
    data /= dev
    return data


def pre_process(data, pixels: int):
    d = np.zeros(pixels)
    for img in data:
        temp = img.tolist()[0]
        temp = np.array(literal_eval(temp))  # 将字符串格式转化为列表格式
        d = np.vstack((d, temp))
    data = d[1:data.shape[0] + 1]
    avg = np.mean(data)
    dev = np.std(data)
    data -= avg
    data /= dev
    return data
