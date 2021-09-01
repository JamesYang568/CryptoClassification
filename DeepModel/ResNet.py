# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:08
# @Author   : JamesYang
# @File     : ResNet.py

import torch.nn.functional as func
from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


# 定义make_layer
def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),  # 尺寸不变
        nn.BatchNorm2d(out_channel)
    )
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


# ResBlock 2 conv with the same size output
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return func.relu(out)


# 堆叠Resnet，见上表所示结构
class Resnet_improve(nn.Module):
    """ implement of ResNet thanks to https://www.jianshu.com/p/972e2c5e6871  \n
    https://zhuanlan.zhihu.com/p/42706477
    """

    def __init__(self, num_classes):
        super(Resnet_improve, self).__init__()
        self.pre = nn.Sequential(  # 第一层
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(3, 2, 1)
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = make_layer(64, 64, 2)
        self.layer2 = make_layer(64, 128, 2, 2)  # stride = 2
        self.layer3 = make_layer(128, 256, 2, 2)  # stride = 2
        # self.layer4 = make_layer(256, 512, 2)  # stride = 2
        self.avg = nn.AvgPool2d(4)  # 平均化
        self.classifier = nn.Sequential(nn.Linear(256, num_classes))  # 全连接

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


def ResNet_18(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
              loss_function='MES', optimizer='SGD', col_name=None, save_mode=False):
    """
    the Classification method using a Simple ResNet18.

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
    net = Resnet_improve(num_classes)
    net.cuda()
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
    pic_name = plot_fig('ResNet', crypto_list, epoch, loss, acc)
    # save_train_result('ResNet', crypto_list, loss, acc)
    h_accuracy = max(acc)

    cl = concat2str(crypto_list)
    model_name = 'ResNet_' + cl
    if save_mode:
        torch.save(net, '../self_model/ResNet/' + model_name + '_model.pkl')  # 注意文件路径
        model_name = model_name + '_model.pkl'
    else:
        torch.save(net.state_dict(), '../self_model/ResNet/' + model_name + '_parameter.pkl')
        model_name = model_name + '_parameter.pkl'

    return pic_name, model_name, h_accuracy


# routine. Please first run feature modeling to get *_ones.csv
if __name__ == '__main__':
    ResNet_18(['Blowfish', 'RSA'],
              ['../static/feature/Blowfish_ones.csv', '../static/feature/RSA_ones.csv'],
              input_channel=1024, loss_function='CE', optimizer='Adam', col_name=['data_frame'], batch=1500, epoch=13)
