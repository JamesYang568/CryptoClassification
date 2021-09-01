# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 19:16
# @Author   : JamesYang
# @File     : NNutil.py

import matplotlib.pyplot as plt
import numpy as np
import time  # 在外面用到了
import torch
import torch.utils.data as Data
from torch import Tensor, nn

"""
专用于训练和测试，模型保存的工具脚本
请不要随意修改本文件，所有接口都在__init__.py文件中声明
"""


def train(model, inputs, labels: list, optimizer, loss_func, length):
    """
    only for SFC
    单条数据的训练，由于不再使用简单全连接模型，因此基本上已经废弃
    model: 将调用函数得到的网络模型传入，耦合

    :return: None
    """
    model.train()  # 声明开始训练
    for t in range(0, length):
        optimizer.zero_grad()  # 清空节点值
        out = model(inputs[t])  # 前向传播
        output = torch.softmax(out, dim=0)
        # 损失计算  由于之前有值运算因此需要要求可导
        if isinstance(loss_func, nn.MSELoss):  # 不同的损失函数要求的target格式不同
            loss = loss_func(output, labels[t].data)
        else:  # 这里NLLLoss和EC需要batch，所以loss function只能是MES
            loss = loss_func(output, labels[t].argmax(dim=0, keepdim=True).squeeze())

        loss.backward()  # 后向传播
        optimizer.step()  # 更新权值


def test(model, inputs, labels, loss_func, length):
    """
    only for SFC

    :return: result is the answer of Net  accuracy %  test_loss %
    """
    model.eval()  # 有输入数据，即使不训练，它也会改变权值
    test_loss = 0
    correct = 0
    result = []
    with torch.no_grad():
        for t in range(0, length):
            out = model(inputs[t])
            output = torch.softmax(out, dim=0)
            pred = output.argmax(dim=0, keepdim=True)
            result.append(pred.numpy().tolist())
            if isinstance(loss_func, nn.MSELoss):
                test_loss += loss_func(output, labels[t]).item()
            else:
                test_loss += loss_func(output, pred.squeeze()).item()
            # 比较是不是对的  dim=0表示二维中的列，dim=1在二维矩阵中表示行
            if pred.item() == labels[t].argmax(dim=0, keepdim=True).item():
                correct += 1
    test_loss /= length  # 求平均
    accuracy = correct * 100 / length
    print("平均损失为：" + str(test_loss * 100) + "%正确率为：" + str(accuracy) + "%")
    return test_loss, accuracy, result


def batch_train(model, loader: Data.DataLoader, optimizer, loss_func):
    """
    网络训练函数
    除了简单全连接SFC之外的所有网络都使用本接口

    :param model: 传入外层生成的模型
    :param loader: Data.DataLoader ，注意复用
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :return: None
    """
    model.train()  # 声明开始训练
    for step, (batch_x, batch_y) in enumerate(loader):  # same as input_data, input_label
        optimizer.zero_grad()
        out = model(batch_x)
        y_pre = torch.softmax(out, dim=1)
        if isinstance(loss_func, nn.MSELoss):
            loss = loss_func(y_pre, batch_y)
        else:
            loss = loss_func(y_pre, batch_y.argmax(dim=1, keepdim=True).squeeze())  # 降一个维度
        loss.backward()
        optimizer.step()


def batch_test(model, loader: Data.DataLoader, loss_func):
    """
    for All NN except SFC

    :return: result is the answer of Net  accuracy % , test_loss % , result
    """
    model.eval()  # 有输入数据，即使不训练，它也会改变权值
    test_loss = 0
    correct = 0
    result = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):
            out = model(batch_x)
            y_pre = torch.softmax(out, dim=1)
            result.extend(y_pre.numpy().tolist())
            n = batch_y.argmax(dim=1, keepdim=True)
            if isinstance(loss_func, nn.MSELoss):  # 不同的损失函数要求的target格式不同
                test_loss += loss_func(y_pre, batch_y).item()
            else:
                test_loss += loss_func(y_pre, n.squeeze()).item()
            m = y_pre.argmax(dim=1, keepdim=True)
            for i in range(m.shape[0]):
                if n[i].eq(m[i]):
                    correct += 1
    test_loss /= len(loader.dataset)  # 求平均
    accuracy = correct * 100 / len(loader.dataset)
    print("平均损失为：" + str(test_loss * 100) + "%正确率为：" + str(accuracy) + "%")
    return test_loss * 100, accuracy, result  # 略有冗余


def verify(model, inputs, length):
    """
    验证模型，用户使用新的数据进行分类时，需要调用本函数， for SFC

    :return: result list
    """
    model.eval()
    result = []
    with torch.no_grad():
        for t in range(0, length):
            out = model(inputs[t])
            output = torch.softmax(out, dim=0)
            result.extend(output.numpy().tolist())
    return result


def batch_verify(model, loader):
    """
    verify and classify the data provided by user
    for all NN except SFC

    :param model: 训练的模型
    :param loader: data loader
    :return: result list
    """
    model.eval()
    result = []
    with torch.no_grad():
        for step, batch_x in enumerate(loader):
            out = model(batch_x[0])
            y_pre = torch.softmax(out, dim=1)
            result.extend(y_pre.numpy().tolist())
    return result


def get_batch(data: Tensor, label, batch: int, shuffle: bool):
    """
    将多个特征向量组成一批

    :param data: 总数据
    :param label: 总标签
    :param batch: 一批数据个数
    :param shuffle: 是否打乱数据
    :return: 分批后加载器loader
    """

    data = data.unsqueeze(1)
    if label is None:  # please recall the subroutine to understand why
        torch_dataset = Data.TensorDataset(data)
    else:
        torch_dataset = Data.TensorDataset(data, label)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch,
        shuffle=shuffle,
    )
    return loader


def save_train_result(model_name, crypto_list: list, loss: list, acc: list):
    """
    save the result lists for loss and acc

    :return: write into file
    """
    cl = concat2str(crypto_list)
    filename = model_name + '_' + cl  # 注意文件名格式
    with open('./verify_result/' + model_name + '/' + filename + '_loss.txt', encoding='utf-8', mode='w') as l:
        l.writelines(str(o) + '\n' for o in loss)
    with open('./verify_result/' + model_name + '/' + filename + '_acc.txt', encoding='utf-8', mode='w') as a:
        a.writelines(str(o) + '\n' for o in acc)


def save_verify_result(model_name, crypto_list: list, result):
    """
    save the result lists for classified result

    :return: write into file
    """
    cl = concat2str(crypto_list)
    filename = model_name + '_' + cl
    with open('verify_result/' + model_name + '/' + filename + '_result.txt', encoding='utf-8', mode='w') as f:
        f.writelines(str(o) + '\n' for o in result)


def plot_fig(model_name, crypto_list: list, epoch: int, loss: list, acc: list):
    """
    draw a figure for loss and acc
    """
    cl = concat2str(crypto_list)
    pic_name = model_name + '_' + cl

    x = np.linspace(1, epoch, epoch)
    plt.figure(figsize=(4.5, 5))

    plt.subplot(211)
    plt.plot(x, loss)
    plt.xticks(np.arange(0, epoch + 1, epoch // 10))
    plt.title('Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('average loss %')
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(x, acc)
    plt.xticks(np.arange(0, epoch + 1, epoch // 10))
    smal = (min(acc) // 10) * 10  # set the axis
    plt.yticks(np.arange(smal - 5, 105, (105 - smal) // 5))
    plt.title('Validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy %')
    plt.tight_layout()

    pic_name = pic_name + '.png'
    plt.savefig('../verify_result/' + model_name + '/' + pic_name)
    plt.clf()
    return pic_name


def concat(crypto_list):
    sort_list = sorted(crypto_list, key=lambda i: i[0])
    return sort_list


def concat2str(crypto_list):
    """
    将加密算法名称按照首字母序连接
    """
    sort_list = concat(crypto_list)
    cl = ''
    for i in sort_list:
        cl = cl + i + '&'
    return cl[:-1]  # 删除最后一个&


def get_whose(crypto_list, i_list):
    # 返回对应算法名和其概率
    max_arg = np.argmax(i_list)
    return crypto_list[max_arg], i_list[max_arg]
