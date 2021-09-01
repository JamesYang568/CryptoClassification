# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/7/22 18:36
# @Author   : Yang Jiaxiong
# @File     : demo.py

#  本地运行，这个主要是获取准确率
import joblib
import pandas as pd
import torch
from torch.autograd import Variable

from BasicModel.Crypto_Embedding import LeNet, identity_layer
from DeepModel import get_batch, concat2str
from DeepModel.feature_modeling import pre_process, NN_data_preparation

if __name__ == '__main__':
    crypto_list = ['3DES', 'AES']
    feature_file_dirs = ['./static/feature/3DES_ones.csv', './static/feature/AES_ones.csv']
    num_classes = len(crypto_list)
    train_data, train_label, test_data, test_label = NN_data_preparation(feature_file_dirs, 0.9)
    train_data, test_data = pre_process(train_data, test_data, 1024)
    train_label = Variable(torch.from_numpy(train_label).float())  # 训练标签
    train_data = Variable(torch.from_numpy(train_data).float())  # 训练特征
    # test_label = Variable(torch.from_numpy(test_label).float())  # 测试标签
    # test_data = Variable(torch.from_numpy(test_data).float())  # 测试特征

    train_loader = get_batch(train_data, train_label, 200, False)

    cnn_model = LeNet(len(crypto_list))
    cnn_model.load_state_dict(torch.load('./self_model/CNN-RF/NN_' + concat2str(crypto_list) + '_parameter.pkl'))
    cnn_model.fc3 = identity_layer()

    cnn_model.eval()
    embedding = []
    with torch.no_grad():
        for step, batch_x in enumerate(train_loader):
            out = cnn_model(batch_x[0])
            embedding.extend(out[1].numpy().tolist())

    target = train_label.argmax(dim=1).detach().numpy().tolist()
    frame = pd.DataFrame(data=[embedding, target]).T
    frame.to_csv('./embedding.csv')
    directory = './self_model/CNN-RF/RF_'
    for i in range(0, len(crypto_list)):
        directory += crypto_list[i]
        if i != len(crypto_list) - 1:
            directory += '_'
    directory += '.model'
    rf_model = joblib.load(directory)
    score_s = rf_model.score(embedding, target)
    print(score_s)

# ['3DES', 'Blowfish' 0.9682914046121593 ], ['3DES', 'AES' 0.960691823899371],
# ['3DES', 'RSA' 0.9520440251572327], ['AES', 'Blowfish' 0.9588574423480084], ['Blowfish', 'RSA' 0.9638364779874213],
# ['AES', 'RSA' 0.9012054507337526], ['3DES', 'AES', 'Blowfish' 0.9509084556254368], ['3DES', 'Blowfish', 'RSA' 0.9519566736547869],
# ['3DES', 'AES', 'RSA'  0.8892382948986722], ['AES', 'Blowfish', 'RSA' 0.8923829489867225],
# ['3DES', 'AES', 'Blowfish', 'RSA' 0.9047431865828093]
