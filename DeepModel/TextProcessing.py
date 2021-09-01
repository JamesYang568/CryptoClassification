# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:40
# @Author   : JamesYang
# @File     : TextProcessing.py

import pandas as pd

"""
密文预处理
"""


def bitwise_32(file_dir):  # 每32个bit计算一下1的个数
    with open(file_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
    transform = []
    for i in range(1024):
        temp = line[i * 8:(i + 1) * 8]
        rlt = bin(int(temp, 16))[2:]
        transform.append(rlt.count('1'))
    return transform


def bitwise_16(file_dir):
    """
    读取一个密文数据并且生成特征  密文最少有4096个十六进制数
    :param file_dir: 文件地址
    :return: 特征向量  [1024]
    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
    transform = []
    for i in range(1024):  # 1024维的一个特征
        temp = line[i * 4:(i + 1) * 4]
        rlt = bin(int(temp, 16))[2:]
        transform.append(rlt.count('1'))
    return transform


def bitwise_8(file_dir):  # 每8个bit计算一下1的个数 2KB（2048个16进制数）
    """
    读取一个密文数据并且生成特征  密文最少有2048个十六进制数
    :param file_dir: 文件地址
    :return: 特征向量  [1024]
    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
    transform = []
    for i in range(1024):  # 1024维的一个特征
        temp = line[i * 2:(i + 1) * 2]
        if temp == "":
            continue
        rlt = bin(int(temp, 16))[2:]
        transform.append(rlt.count('1'))
    return transform


def bitwise_4(file_dir):  # 每4个bit计算一下1的个数
    with open(file_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
    transform = []
    for i in range(1024):
        temp = line[i * 1:(i + 1) * 1]
        rlt = bin(int(temp, 16))[2:]
        transform.append(rlt.count('1'))
    return transform


# deprecated below
def record(line: str):
    # 原本的特征提取方法
    transform = []
    # lens = 4100 // 4
    for i in range(1024):
        temp = line[i * 4:(i + 1) * 4]
        # 计算01串中1的个数
        rlt = bin(int(temp, 16))[2:]
        transform.append(rlt.count('1'))
        # 计算数值
        # rlt = int(temp, 16)
        # transform.append(int(rlt))
    return transform


def get_bitwise_16(filename):
    transforms = []
    counter = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            counter += 1
            line = line.replace('\n', '')
            if line == '""':
                continue
            if len(line) > 4096:
                transforms.append(str(record(line)))
            if counter > 8000:
                break

    dataframe = pd.DataFrame({'data_frame': transforms})
    new_filename = filename.replace('.csv', '') + '_ones.csv'
    dataframe.to_csv(new_filename, ',')
    return new_filename
    # ifile.writelines(transforms)


def get_bitwise_8(filename):
    transforms = []
    counter = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            counter += 1
            line = line.replace('\n', '')
            if line == '""':
                continue
            if len(line) > 4096:
                transforms.append(str(record(line)))
            if counter > 8000:
                break

    dataframe = pd.DataFrame({'data_frame': transforms})
    new_filename = filename.replace('.csv', '') + '_ones.csv'
    dataframe.to_csv(new_filename, ',')
    return new_filename


def get_bitwise_hash(filename):
    transforms = []
    counter = 0
    rq = ""
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            if line != '""':
                counter += 1
                rq += line
            if len(rq) > 4096:
                transforms.append(str(record(rq)))
                rq = ''
            if counter > 111000:
                break

    dataframe = pd.DataFrame({'data_frame': transforms})
    dataframe.to_csv('ciphertext/' + filename + '_ones.csv', ',')


if __name__ == '__main__':
    get_bitwise_16('Blowfish_CBC')
