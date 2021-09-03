# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Time     : 2021/3/16 17:43
# @Author   : JamesYang
# @File     : backend.py

import os
import time

import BM
import DM
import End2End

'''
此代码的注释是由杨嘉雄完成的，有任何问题，请联系本人
'''


class Args:
    def __init__(self, crypto_list: list, category: int, input_channel: int, col_name: list, ratio=0.0, epoch=30,
                 batch=300, loss_function='CE', optimizer='Adam', save_mode='total'):
        self.crypto_list = crypto_list  # 传入有可能是被谁加密的加密函数列表
        self.category = category  # len(crypto_list)  这就代表了几分类
        self.ratio = ratio  # 默认传0
        self.input_channel = input_channel  # 特征的数目  神经网络要用  先传1024
        self.epoch = epoch  # 可以让用户选择的，默认30
        self.batch = batch  # 可以让用户选择的，默认300
        self.loss_function = loss_function  # 可以让用户选择的 'NLL'、'MSE'、'CE'三种  ，默认CE
        self.optimizer = optimizer  # 可以让用户选择的 'Adam'、'RMSprop'、'Adadelta'、'SGD'四种  ，默认Adam
        self.save_mode = save_mode  # 模型是如何保存的（total、para）  默认total
        self.col_name = col_name  # 选择的特征，他们的名字

    def change_save_mode(self, value):
        self.save_mode = value

    def change_crypto_list(self, c_list):
        self.crypto_list = c_list


def read_raw_files(file_dir):
    """
    读取文件夹下面的文件，用于自适应文件名。Attention ! 不能递归读取
    :param file_dir: 传入要读取的文件夹路径
    :return: 每个文件的地址和文件大小，list
    """
    file_list = os.listdir(file_dir)
    file_size = []
    file_dirs = []
    for file in file_list:
        path = file_dir + file
        file_dirs.append(path)
        file_size.append(os.path.getsize(path))
    return file_dirs, file_size


def getCipherResult(file_dir: str, model_name: str, args):
    """
    主调接口，用户上传待测密文文件，使用现有模型进行分类

    :param file_dir:  当前用户文件上传的顶层目录
    :param model_name:  验证所选取的模型名称  四选一
    :param args: 辅助参数
    :return: result-dict list and a timestamp
    """
    args.change_crypto_list(DM.concat(args.crypto_list))
    # 读文件夹  得到两个list [文件名],[大小]
    file_list, file_size = read_raw_files(file_dir)  # './static/CiphertextFile/test/'
    timestamp = time.localtime()
    # 开始执行核心分类功能
    if model_name == 'SVM' or model_name == 'RF':
        args.change_save_mode('total')  # 这里为了保证模型的读取模式正确，将此参数设置死
        result, confidence = BM.load_and_test(file_list, model_name, args)
    elif model_name == 'CNN-SVM':
        args.change_save_mode('para')
        result, confidence = End2End.test_CNN_SVM(file_list, args)
    elif model_name == 'CNN-RF':
        args.change_save_mode('para')
        result, confidence = End2End.test_CNN_RF(file_list, args)
    else:
        args.change_save_mode('para')
        result, confidence = DM.load_and_test(model_name, file_list, args)
    # 设置阈值
    threshold = 1 / args.category
    return_result = []

    # create result-dict list
    if isinstance(confidence, list):  # 判断是基础模块还是深度模块
        for i in range(len(result)):
            if confidence[i] < threshold:
                status = False
            else:
                status = True
            return_result.append({"fileName": file_list[i],
                                  "size": file_size[i],
                                  "status": status,
                                  "result": result[i],
                                  "confidence": confidence[i]})
    else:
        if confidence < threshold:
            status = False
        else:
            status = True
        for i in range(len(result)):
            return_result.append({"fileName": file_list[i],
                                  "size": file_size[i],
                                  "status": status,
                                  "result": result[i],
                                  "confidence": confidence})

    return return_result, str(timestamp)


def getTrainedModel(file_dir: str, model_name: str, args):
    """
    主调接口，用户上传已经分好类的密文文件，用于训练模型

    :param file_dir: 提供所有密文文件最上层路径
    :param model_name: 训练模型的种类
    :param args: 必备参数，针对不同的模型有不同的训练参数
    :return: result dict
    """
    args.change_crypto_list(DM.concat(args.crypto_list))  # 重新排序
    crypto_files = {}
    for i in args.crypto_list:  # 读取各个分好类的密文文件夹
        crypto_files[i] = read_raw_files(file_dir + i + '/')[0]
    timestamp = time.localtime()
    # 模型正式开始训练
    if model_name == 'SVM' or model_name == 'RF':
        args.change_save_mode('total')  # 保证模型的保存模式正确
        accuracy, pic_path, model_path = BM.train_and_test(crypto_files, model_name, args)
    elif model_name == 'CNN-SVM':
        args.change_save_mode('para')
        accuracy, cnn_model_path, svm_model_path = End2End.train_CNN_SVM(crypto_files, args)
        model_path = [cnn_model_path, svm_model_path]
        pic_path = ''
    elif model_name == 'CNN-RF':
        args.change_save_mode('para')
        accuracy, cnn_model_path, rf_model_path = End2End.train_CNN_RF(crypto_files, args)
        model_path = [cnn_model_path, rf_model_path]
        pic_path = ''
    else:
        args.change_save_mode('para')
        pic_path, model_path, accuracy = DM.train_and_test(model_name, crypto_files, args)

    # create result dict
    return {
        'model_name': model_name,
        'time': str(timestamp),
        'pic_dir': pic_path,
        'model_dir': model_path,
        'accuracy': accuracy
    }


# run this routine
if __name__ == '__main__':
    # time_start = time.time()
    #
    # 模型训练
    # args1 = Args(['3DES', 'AES', 'Blowfish', 'RSA'], 4, 1024, col_name=['data_frame'], batch=500, epoch=40, ratio=0.8,
    #              save_mode='para')
    # getTrainedModel('./static/CiphertextFile/Test/', 'CNN-SVM', args1)
    #
    # 模型验证
    args = Args(['3DES', 'AES', 'Blowfish', 'RSA'], 4, 1024, ['data_frame'], save_mode='para')
    getCipherResult('static/CiphertextFile/Test/Test/', 'CNN-SVM', args)

    # time_end = time.time()
    # print('totally cost', time_end - time_start)
