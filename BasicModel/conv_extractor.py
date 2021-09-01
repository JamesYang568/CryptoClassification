import os

import numpy as np
import pandas as pd
import torch
from numpy import random
from torch import nn

from DeepModel.data_preparation import data_loader


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CnPo(nn.Module):
    """
    https://blog.csdn.net/shanglianlm/article/details/85165523  固定权重初始化
    """

    def __init__(self, dimension):
        super(CnPo, self).__init__()
        if dimension == 192:  # 192维特征
            self.net = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2))
        elif dimension == 64:  # 64维特征
            self.net = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=4))
        else:  # 75维特征
            self.net = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=3),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2))
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # chi方分布
                # chisq = torch.Tensor(np.random.chisquare(3, [3, 1, 3, 3]))  # 自由度和张量大小
                # m.weight = torch.nn.Parameter(chisq)
                # F分布
                fdis = torch.Tensor(np.random.f(3, 3, [3, 1, 3, 3]))
                m.weight = torch.nn.Parameter(fdis)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, data):
        return self.net(data)


def get_conv_feature(feature_list: list, crypto_list: list, dimension, fuc: str):
    """
    获取CnPo特征

    :param feature_list:  传入ones文件
    :param crypto_list:  传入加密算法种类
    :param dimension: 特征维数
    :param fuc: 是训练还是分类
    :return: 返回所有的特征列表list
    """
    fname = '/home/ubuntu/CryptoClassification/static/feature/conv_feature_' + fuc + '.csv'  # /home/ubuntu
    seed_torch(1030)
    saver_data = []
    net = CnPo(dimension)
    if fuc == 'train':  # 训练时需要的  包含标签
        saver_label = []
        loader = data_loader(feature_list, '', ['data_frame'])
        for step, (batch_x, batch_y) in enumerate(loader):
            x = net(batch_x)
            x = x.view(x.size(0), -1)
            saver_data.extend(x.detach().numpy().tolist())
            saver_label.extend(batch_y.detach().numpy().tolist())
        saver = pd.DataFrame(saver_data)
        saver.insert(dimension, '{}'.format(dimension), value=saver_label)
        saver.to_csv(fname, encoding='utf-8')
    elif fuc == 'classify':  # 验证时需要的，不用标签
        loader = data_loader(feature_list, 'classify', ['data_frame'])
        for step, batch_x in enumerate(loader):
            x = net(batch_x[0])
            x = x.view(x.size(0), -1)
            saver_data.extend(x.detach().numpy().tolist())
        df = pd.DataFrame(saver_data)
        df.to_csv(fname, encoding='utf-8')
    return saver_data
    # return fname, len(saver_data)
