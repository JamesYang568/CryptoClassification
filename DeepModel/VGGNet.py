# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:11
# @Author   : JamesYang
# @File     : VGGNet.py

import torch.nn.functional as func
from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


class VGG_Advance(nn.Module):
    """ implement of VGG thanks to https://zhuanlan.zhihu.com/p/263527295  \n
    https://www.cnblogs.com/sclu/p/14163969.html
    由于分类数目较少，因此对全连接层和输入进行了调整
    """

    def __init__(self, arch: list, num_classes):
        super(VGG_Advance, self).__init__()
        self.in_channels = 1
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(2 * 2 * 512, 512)  # 原7 * 7 * 512, 4096
        self.bn1 = nn.BatchNorm1d(512)  # 4096
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):  # kernel size=3
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = func.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = func.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = func.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = func.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        # out = func.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)  # 防止过拟合
        out = func.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = func.relu(out)
        return self.fc3(out)


def VGG_11(num_classes):
    return VGG_Advance([1, 1, 2, 2, 2], num_classes)  # 传入对应了几层


def VGG(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
        loss_function='MES', optimizer='SGD', col_name=None, save_mode=False):
    """
    the Classification method using a Simple VGG11.

    :param crypto_list: the crypto_algorithm to be classified.
    :param feature_file_dirs: the files dirs for features
    :param input_channel:  feature dims
    :param ratio: the ratio to split dataset. default: 0.8
    :param epoch: epoch number. default: 30
    :param batch: number for one batch. default: 100
    :param loss_function: loss function that you choose. default: MES
    :param optimizer: optimizer that you choose. default: SGD but not recommend
    :param col_name: the feature column chosen default: 'F_1024b'
    :param save_mode: how to save the model you trained. default: False means only save parameter
    :return: result figure's name and model's name and highest accuracy
    """

    num_classes = len(crypto_list)
    net = VGG_11(num_classes)

    train_data, train_label, test_data, test_label = NN_data_preparation(feature_file_dirs, ratio, col_name=col_name[0])
    train_data, test_data = pre_process(train_data, test_data, input_channel)  # 耗时比较长

    train_label = Variable(torch.from_numpy(train_label).float())  # 训练标签
    train_data = Variable(torch.from_numpy(train_data).float())  # 训练特征
    test_label = Variable(torch.from_numpy(test_label).float())  # 测试标签
    test_data = Variable(torch.from_numpy(test_data).float())  # 测试特征

    # 损失函数选择
    if loss_function == 'NLL':
        loss_fn = nn.NLLLoss()
    elif loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    # 优化器选择
    if optimizer == 'Adam':
        optim = torch.optim.Adam(net.parameters())
    elif optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(net.parameters(), alpha=0.9)
    elif optimizer == 'Adadelta':
        optim = torch.optim.Adadelta(net.parameters())
    else:
        optim = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.8)  # momentum小一点

    train_loader = get_batch(train_data, train_label, batch, True)
    test_loader = get_batch(test_data, test_label, batch, True)

    loss = []
    acc = []
    for i in range(1, epoch + 1):
        print("第" + str(i) + "轮")
        start = time.time()
        batch_train(net, train_loader, optim, loss_fn)
        end = time.time()
        print("消耗时间为：" + str(end - start))
        l, a, r = batch_test(net, test_loader, loss_fn)
        loss.append(l)
        acc.append(a)
    # save_train_result('VGGNet', crypto_list, loss, acc)  # 需要保存下来成为txt文件将本句注释即可
    pic_name = ''
    # pic_name = plot_fig('VGG', crypto_list, epoch, loss, acc)
    h_accuracy = max(acc)

    cl = concat2str(crypto_list)
    model_name = 'VGG_' + cl
    if save_mode:
        torch.save(net, '../self_model/VGG/' + model_name + '_model.pkl')
        model_name = model_name + '_model.pkl'
    else:
        torch.save(net.state_dict(), '../self_model/VGG/' + model_name + '_parameter.pkl')
        model_name = model_name + '_parameter.pkl'

    return pic_name, model_name, h_accuracy


# routine. Please first run feature modeling to get *_ones.csv
if __name__ == '__main__':
    VGG(['3DES', 'AES', 'Blowfish', 'RSA'],
        ['../static/feature/3DES_ones.csv', '../static/feature/AES_ones.csv', '../static/feature/Blowfish_ones.csv',
         '../static/feature/RSA_ones.csv'],
        input_channel=1024, loss_function='CE', optimizer='Adam', col_name=['data_frame'], batch=2000, epoch=8)



# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace)
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): ReLU(inplace)
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace)
#     (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace)
#     (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): ReLU(inplace)
#     (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (14): ReLU(inplace)
#     (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace)
#     (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): ReLU(inplace)
#     (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace)
#     (2): Dropout(p=0.5)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace)
#     (5): Dropout(p=0.5)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )
