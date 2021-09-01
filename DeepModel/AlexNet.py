# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:07
# @Author   : JamesYang
# @File     : AlexNet.py

from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


class AlexNet(nn.Module):
    """
    implement of AlexNet thanks to https://blog.csdn.net/luoluonuoyasuolong/article/details/81750190 \n
    https://blog.csdn.net/shanglianlm/article/details/86424857 \n
    this Model is the AlexNetPlus
    ciphertext has preprocessed with counting ones methods
    """

    def __init__(self, num_classes, grayscale=False):
        super(AlexNet, self).__init__()
        self.grayscale = grayscale
        self.num_classes = num_classes
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, padding=1),  # stride改为1，原为4
            # 卷积核96个大小为 11*11
            # (32 + 2 * padding（1） - kernel_size（11）) / stride（1） + 1 = 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 24/2 = 12
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            # (12+2-3)/1+1 = 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 12/2 = 6
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            # (6+2-3)/1+1 = 6
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # (6+2-3)/1+1 = 6
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 原stride为1
            # (6+2-3)/1+1 = 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 6/2 = 3(因此这里是3*3的图片了)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x


def AlexCNN(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
            loss_function='MES', optimizer='SGD', col_name=None, save_mode=False):
    """
    the Classification method using AlexNet.

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
    # crypto_list = concat(crypto_list)
    num_classes = len(crypto_list)
    grayscale = True  # 是否为灰度图

    model = AlexNet(num_classes, grayscale)

    train_data, train_label, test_data, test_label = NN_data_preparation(feature_file_dirs, ratio, col_name[0])
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
        loss_fn = nn.MSELoss()

    # 优化器选择
    if optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters())
    elif optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(), alpha=0.9)
    elif optimizer == 'Adadelta':
        optim = torch.optim.Adadelta(model.parameters())
    else:
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)  # momentum小一点

    train_loader = get_batch(train_data, train_label, batch, True)
    test_loader = get_batch(test_data, test_label, batch, True)

    loss = []
    acc = []
    for i in range(1, epoch + 1):
        print("第" + str(i) + "轮")
        start = time.time()
        batch_train(model, train_loader, optim, loss_fn)
        end = time.time()
        print("训练消耗时间为：" + str(end - start))
        l, a, r = batch_test(model, test_loader, loss_fn)
        loss.append(l)
        acc.append(a)
    save_train_result('AlexNet', crypto_list, loss, acc)
    pic_name = plot_fig('AlexNet', crypto_list, epoch, loss, acc)
    h_accuracy = max(acc)

    cl = concat2str(crypto_list)
    model_name = 'AlexNet_' + cl
    if save_mode:
        torch.save(model, '../self_model/AlexNet/' + model_name + '_model.pkl')  # 注意文件路径
        model_name = model_name + '_model.pkl'
    else:
        torch.save(model.state_dict(), '../self_model/AlexNet/' + model_name + '_parameter.pkl')
        model_name = model_name + '_parameter.pkl'
    return pic_name, model_name, h_accuracy


# routine. Please first run feature modeling to get *_ones.csv
if __name__ == '__main__':
    AlexCNN(['Blowfish', 'RSA'],
            ['../static/feature/Blowfish_ones.csv', '../static/feature/RSA_ones.csv', ],
            input_channel=1024, loss_function='CE', optimizer='Adam', col_name=['data_frame'], batch=1500, epoch=2)
