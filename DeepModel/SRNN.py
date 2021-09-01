# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:12
# @Author   : JamesYang
# @File     : SRNN.py

from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


class RNN_Net(nn.Module):
    """ implement of RNN thanks to https://blog.csdn.net/beautiful77moon/article/details/105150723
    this script isn't used in web project, just for experimentation
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_Net, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(  # 也可以尝试使用跑LSTM
            input_size=input_size,  # 默认为32
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True  # 这样就可以是batch在最前面
        )
        self.liner = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 32, 32)
        out, hidden_prev = self.rnn(x, None)
        # 选取最后一次的r_out进行输出
        out = out[:, -1, :]  # 或者hidden_prev 里面 h_n,h_c 的h_n 是一样的
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.liner(out)
        return out


def SRNN(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
         loss_function='MES', optimizer='SGD', col_name=None, save_mode=False):
    """
    the Classification method using a Simple RNN.

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
    net = RNN_Net(32, 64, num_classes)  # 参数可调整，注意，由于传统的RNN要求的是一维特征输入，隐层个数需要着重考虑

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
    pic_name = plot_fig('SRNN', crypto_list, epoch, loss, acc)
    h_accuracy = max(acc)

    cl = concat2str(crypto_list)
    model_name = 'SRNN_' + cl
    if save_mode:
        torch.save(net, '../self_model/SRNN/' + model_name + '_model.pkl')  # 注意文件路径
        model_name = model_name + '_model.pkl'
    else:
        torch.save(net.state_dict(), '../self_model/SRNN/' + model_name + '_parameter.pkl')
        model_name = model_name + '_parameter.pkl'

    return pic_name, model_name, h_accuracy


# 效果不怎么好
if __name__ == '__main__':
    SRNN(['AES', '3DES', 'RSA'], ['../static/feature/AES_ones.csv', '../static/feature/3DES_ones.csv',
                                  '../static/feature/RSA_ones.csv'], input_channel=1024,
         loss_function='CE', optimizer='Adam', col_name=['data_frame'], batch=500, epoch=30)
