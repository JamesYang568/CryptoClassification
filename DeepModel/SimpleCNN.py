# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:10
# @Author   : JamesYang
# @File     : SimpleCNN.py

import torch.nn.functional as func
from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


class LeNet5(nn.Module):
    """ implement of LeNet5 thanks to https://blog.csdn.net/jeryjeryjery/article/details/79426907 \n
    https://www.jianshu.com/p/4ce0bbfb05e8?from=timeline \n
    原始网络结构为：
        输入图像：32x32x1的灰度图像 \n
        卷积核 5x5 stride=1 padding=2 得到Conv1：28x28x6 \n
        池化层：2x2，stride=2 （后sigmoid激活）  得到Pool1：14x14x6 \n
        卷积核 5x5 stride=1  得到Conv2：10x10x16 \n
        池化层：2x2，stride=2  （后sigmoid激活）  得到Pool2：5x5x16 \n
        然后将Pool2展开，得到长度为400的向量 \n
        经过第一个全连接层，得到FC1，长度120 \n
        经过第二个全连接层，得到FC2，长度84 \n
        最后送入softmax回归，得到每个类的对应的概率值。
    """

    def __init__(self, num_classes, input_channel=1):
        r"""
        implement of LeNextX for advance fine training
        :param num_classes:  分类的数量
        :param input_channel:  输入维度
        """
        super(LeNet5, self).__init__()
        self.in_channel = input_channel
        self.num_classes = num_classes

        # Conv1d 情况下in_channels>out_channels  输入的向量的横线维度必须大于out_channels大小，数据不会凭空变多
        # https://www.cnblogs.com/mlgjb/p/11751260.html 讲解input格式
        self.conv1 = nn.Conv2d(1, out_channels=6, kernel_size=5, padding=2)
        # (32+2*padding（2）-kernel_size（5）)/1+1 = 32   下采样 32/2 =16
        self.conv2 = nn.Conv2d(6, out_channels=16, kernel_size=5)
        # (16-5)/1+1 = 12   下采样 12/2 = 6
        self.fc1 = nn.Linear(16 * 6 * 6, 100)  # 这里要求 16*6*6 * 100==9600*6
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = func.max_pool2d(self.conv1(x), (2, 2))  # 原始的模型使用的是 平均池化  这里采样和池化的顺序可以交换的
        x = torch.sigmoid(x)  # relu走一波:)
        x = func.max_pool2d(self.conv2(x), (2, 2))
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1)  # 不要少 0！！（或者(batch,-1)更多用）  view(-1,x) -1的位置是自动计算位置=总/x
        x = self.fc3(self.fc2(self.fc1(x)))
        # x = func.softmax(x, dim=1) 放在外面
        return x


def SCNN(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
         loss_function='MES', optimizer='SGD', col_name=None, save_mode=False):
    """
    the Classification method using a Simple CNN.

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
    num_classes = len(crypto_list)  # 分类数目
    grayscale = True  # 是否为灰度图
    SCNN = LeNet5(num_classes, grayscale)

    train_data, train_label, test_data, test_label = NN_data_preparation(feature_file_dirs, ratio, col_name=col_name[0])
    train_data, test_data = pre_process(train_data, test_data, input_channel)

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
        optim = torch.optim.Adam(SCNN.parameters())
    elif optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(SCNN.parameters(), alpha=0.9)
    elif optimizer == 'Adadelta':
        optim = torch.optim.Adadelta(SCNN.parameters())
    else:
        optim = torch.optim.SGD(SCNN.parameters(), lr=0.05, momentum=0.8)  # momentum小一点

    train_loader = get_batch(train_data, train_label, batch, True)
    test_loader = get_batch(test_data, test_label, batch, True)

    loss = []
    acc = []
    for i in range(1, epoch + 1):
        print("第" + str(i) + "轮")
        start = time.time()
        batch_train(SCNN, train_loader, optim, loss_fn)
        end = time.time()
        print("消耗时间为：" + str(end - start))
        l, a, r = batch_test(SCNN, test_loader, loss_fn)
        loss.append(l)
        acc.append(a)
    save_train_result('SCNN', crypto_list, loss, acc)  # 需要保存下来成为txt文件将本句注释即可
    pic_name = plot_fig('SCNN', crypto_list, epoch, loss, acc)
    h_accuracy = max(acc)

    cl = concat2str(crypto_list)
    model_name = 'SCNN_' + cl
    if save_mode:
        torch.save(SCNN, './self_model/SCNN/' + model_name + '_model.pkl')
        model_name = model_name + '_model.pkl'
    else:
        torch.save(SCNN.state_dict(), './self_model/SCNN/' + model_name + '_parameter.pkl')
        model_name = model_name + '_parameter.pkl'

    return pic_name, model_name, h_accuracy


# routine. Please first run feature modeling to get *_ones.csv
if __name__ == '__main__':
    SCNN(['3DES', 'AES', 'Blowfish', 'RSA'],  # 注意特征文件的顺序和加密的顺序需要一样
         ['../static/feature/3DES_ones.csv', '../static/feature/AES_ones.csv', '../static/feature/Blowfish_ones.csv',
          '../static/feature/RSA_ones.csv'],
         input_channel=1024, loss_function='CE', optimizer='Adam', col_name=['data_frame'], batch=1500, epoch=45)
