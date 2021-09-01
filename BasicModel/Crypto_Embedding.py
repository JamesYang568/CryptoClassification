# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/7/14 10:46
# @Author   : Yang Jiaxiong
# @File     : Crypto_Embedding.py

import joblib
import pandas as pd
import torch.nn.functional as func
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from DeepModel.NNutil import *
from DeepModel.feature_modeling import pre_process, NN_data_preparation


# LeNet网络实现
class LeNet(nn.Module):
    def __init__(self, num_classes, input_channel=1):
        """
        implement of LeNextX for advance fine training
        :param num_classes:  分类的数量
        :param input_channel:  输入维度
        """
        super(LeNet, self).__init__()
        self.in_channel = input_channel
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 6 * 6, 100)  # 这里要求 16*6*6 * 100==9600*6
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = func.max_pool2d(self.conv1(x), (2, 2))  # 原始的模型使用的是 平均池化  这里采样和池化的顺序可以交换的
        x = torch.sigmoid(x)  # relu走一波:)
        x = func.max_pool2d(self.conv2(x), (2, 2))
        x = torch.sigmoid(x)
        x = x.view(x.size(0), -1)  # 不要少 0！！（或者(batch,-1)更多用）  view(-1,x) -1的位置是自动计算位置=总/x
        x = self.fc2(self.fc1(x))
        output = self.fc3(x)
        return output, x


# 一个 x->x 层  用于在验证时替代LeNet的最后一个全连接层
class identity_layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 如何输入的就如何输出
        return x


# 训练模型
def train_svm(crypto_list: list, feature_vector, labels):
    """
    训练模型  CNN的最后一次embedding传入SVM作为输入
    """
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(feature_vector, labels, test_size=0.3)
    model = svm.SVC(kernel='rbf', gamma=0.02, C=1.5)
    model = model.fit(feature_vector, labels)
    s = "SVM_"
    for i in range(0, len(crypto_list)):
        s += crypto_list[i]
        if i != len(crypto_list) - 1:
            s += '_'
    model_path = '/home/ubuntu/CryptoClassification/self_model/CNN-SVM/' + s + '.model'
    joblib.dump(model, model_path)
    return model.score(Xtest, Ytest), model_path


def train_rf(crypto_list: list, feature_vector, labels):
    """
    训练模型  CNN的最后一次embedding传入RF作为输入
    """
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(feature_vector, labels, test_size=0.3)
    # 定义模型
    rfc = RandomForestClassifier(n_estimators=150, min_impurity_decrease=0.03, criterion='gini', max_features='sqrt',
                                 oob_score=False,
                                 max_depth=15)
    # 训练
    rfc.fit(Xtrain, Ytrain)
    # 测试
    score_r = rfc.score(Xtest, Ytest)
    # print(score_r)
    # 保存模型
    s = "RF_"
    for i in range(0, len(crypto_list)):
        s += crypto_list[i]
        if i != len(crypto_list) - 1:
            s += '_'
    model_path = './self_model/CNN-RF/' + s + '.model'
    joblib.dump(rfc, model_path)
    return score_r, model_path


def train_cnn(crypto_list: list, feature_file_dirs: list, input_channel: int, ratio=0.8, epoch=30, batch=100,
              loss_function='MES', optim='SGD', col_name=None, save_mode=False, type='CNN-SVM'):
    """
    训练CNN模型
    """
    num_classes = len(crypto_list)  # 分类数目
    grayscale = True  # 是否为灰度图
    model = LeNet(num_classes, grayscale)

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
    if optim == 'Adam':
        optim = torch.optim.Adam(model.parameters())
    elif optim == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(), alpha=0.9)
    elif optim == 'Adadelta':
        optim = torch.optim.Adadelta(model.parameters())
    else:
        optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.8)  # momentum小一点

    train_loader = get_batch(train_data, train_label, batch, True)

    embedding = []
    target = []
    for i in range(1, epoch + 1):
        target.clear()
        embedding.clear()
        print("第" + str(i) + "轮")
        start = time.time()
        model.train()  # 声明开始训练
        for step, (batch_x, batch_y) in enumerate(train_loader):  # same as input_data, input_label
            optim.zero_grad()
            out, ebd = model(batch_x)
            if i == epoch:
                embedding.extend(ebd.detach().numpy().tolist())
                tmp = batch_y.argmax(dim=1).detach().numpy().tolist()
                target.extend(tmp)
            y_pre = torch.softmax(out, dim=1)
            if isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(y_pre, batch_y)
            else:
                loss = loss_fn(y_pre, batch_y.argmax(dim=1, keepdim=True).squeeze())  # 降一个维度
            loss.backward()
            optim.step()
        end = time.time()
        print("消耗时间为：" + str(end - start))
        # batch_test(model, test_loader, loss_fn)  # 可要可不要
    cl = concat2str(crypto_list)
    model_name = 'NN_' + cl
    if save_mode:
        torch.save(model, './self_model/' + type + '/' + model_name + '_model.pkl')
        model_name = './self_model/' + type + '/' + model_name + '_model.pkl'
    else:
        torch.save(model.state_dict(),
                   './self_model/' + type + '/' + model_name + '_parameter.pkl')
        model_name = './self_model/' + type + '/' + model_name + '_parameter.pkl'

    return embedding, target, model_name


# 测试模型
def CNN_SVM(crypto_list: list, test_loader):
    """
    仅分类密文 将 CNN 和 SVM结合起来
    :param crypto_list: 密文列表
    :param test_loader: 测试数据
    :return: 密文的加密算法是谁
    """
    # 删除CNN最后一层
    cnn_model = LeNet(len(crypto_list))
    cnn_model.load_state_dict(torch.load(
        './trained_model/CNN-SVM/NN_' + concat2str(crypto_list) + '_parameter.pkl'))
    cnn_model.fc3 = identity_layer()
    # encoder部分
    cnn_model.eval()
    embedding = []
    with torch.no_grad():
        for step, batch_x in enumerate(test_loader):
            out = cnn_model(batch_x[0])
            embedding.extend(out[1].numpy().tolist())
    pd.DataFrame(embedding).to_csv("4embedding.csv")  # todo
    # 得到SVM模型
    directory = './trained_model/CNN-SVM/SVM_'
    for i in range(0, len(crypto_list)):
        directory += crypto_list[i]
        if i != len(crypto_list) - 1:
            directory += '_'
    directory += '.model'
    svm_model = joblib.load(directory)
    # 使用decoder部分
    ans = svm_model.predict(embedding)
    return ans


def CNN_RF(crypto_list: list, test_loader):
    """
    仅分类密文 将 CNN 和 RF结合起来
    :param crypto_list: 密文列表
    :param test_loader: 测试数据
    :return: 密文的加密算法是谁
    """
    # 删除CNN最后一层
    cnn_model = LeNet(len(crypto_list))
    cnn_model.load_state_dict(torch.load(
        './trained_model/CNN-RF/NN_' + concat2str(crypto_list) + '_parameter.pkl'))
    cnn_model.fc3 = identity_layer()
    # encoder部分
    cnn_model.eval()
    embedding = []
    with torch.no_grad():
        for step, batch_x in enumerate(test_loader):
            out = cnn_model(batch_x[0])
            embedding.extend(out[1].numpy().tolist())
    # 得到RF模型
    directory = './trained_model/CNN-RF/RF_'
    for i in range(0, len(crypto_list)):
        directory += crypto_list[i]
        if i != len(crypto_list) - 1:
            directory += '_'
    directory += '.model'
    rf_model = joblib.load(directory)
    # 使用decoder部分
    ans = rf_model.predict(embedding)
    return ans
