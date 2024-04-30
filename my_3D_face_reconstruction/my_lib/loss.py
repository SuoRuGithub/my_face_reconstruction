'''
这部分我们设计了训练神经网络时用到的损失函数
完全实现论文中所有的损失函数实在是难度太大了，所以在这里实现了感知损失函数和关键点函数
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实现论文中的“感知损失函数”
# 这部分的计算方法是计算生成的人脸图像和原来的人脸图像的深度特征（这个深度特征由一个神经网络提取）的余弦距离
def per_loss(id_featureA, id_featureB):
    '''
    此处输入的图像的形状：(batch_size, 3, H, W)
    '''
    cos_d = torch.sum(id_featureA * id_featureB, dim = -1)
    return torch.sum(1 - cos_d) / cos_d.shape[0]    # 这怎么和论文里不一样？

# 关键点损失函数
def landmark_loss(predict_lm, gt_lm):
    '''
    此处关键点的形状是：(batch_size, 68, 2)
    下面的权重的形状是：(1, 68)
    '''
    weight = np.ones([68])
    weight[28:31] = 20
    weight[-8:] = 20
    weight = np.expand_dims(weight, 0)  # 形状：(68) -> (1, 68)
    weight = torch.tensor(weight).to(device)

    loss = torch.sum((predict_lm - gt_lm) ** 2, dim = -1) * weight   # 原本的形状是(B, 68, 2)，沿着最后一个维度进行求和，形状变成(B, 68) (此处似乎有广播？)
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])    # 就是一个均方误差（带权重）
    return loss

# 正则项（暂略）

