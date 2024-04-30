"""
用来训练模型，基本是按照论文提供的思路实现的
但是论文中有很多优化的措施，比如分布式数据并行，这些我们就不涉及了
"""
import numpy as np
import torch
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms # ? 
import glob
import time 

import os

import my_lib.loss
from my_lib.model import myNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 训练设备
batch_size = 32     # 批大小

dataset_path = 'Model'

LR = 5e-5           # 学习率

para_num = 239      # 表示人脸所需要的参数

Last_epoch = -1      # 上次训练的迭代次数
Epoch_nums = 100    # 总共的迭代次数

image_size = 224

train_root = r'data/train'
val_root = r'data/val'
model_save_path = r'trained_model'
'''
# 创建数据集，下面这个函数返回一个包含所有图像的列表
root = r'data/FaceSample'
def make_dataset(root):
    images = []
    for root, _, fnames in sorted(os.walk(root, followlinks = True)):    # os.walk函数用来遍历目录中的文件，返回的是(root, dirs, files)一个三元组，即当前目录的地址，当前目录下所有目录的名字以及当前目录下文件的名字
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
class ImageDataset():
    def __init__(self, )
# 创造一个数据加载器
batch_size = 1
def make_dataloader(datasets):
    dataloader = torch.utils.data.DataLoader(datasets, 
                                         batch_size=batch_size,  # 批量大小
                                         shuffle=True,           # 是否打乱数据集
                                         num_workers=5 # 使用多个线程加载数据的工作进程数
                                        )
    return dataloader
'''
# 初始化神经网络
def init_weights(model):
    pass        # 我们直接用预先训练好的参数，不初始化了

def main(): 
    # 载入数据
    dataset_train = dset.ImageFolder(root = train_root,
                               transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))   # 是这样吗？
                               ]))
    
    dataset_val = dset.ImageFolder(root = val_root,
                               transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))   # 是这样吗？
                               ]))
    dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                             batch_size = batch_size,
                                             shuffle = True,
                                             num_workers = 5)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, 
                                             batch_size = batch_size,
                                             shuffle = True,
                                             num_workers = 5)
    print("数据集加载完毕！")
    print("训练集大小：", len(dataset_train), "测试集大小：", len(dataloader_val))

    print("当前使用的设备：", device)

    # 初始化神经网络结构
    net = myNN(para_num).to(device)
    print("神经网络初始化成功！")

    '''保存'''

    

    print("------------Training Start------------")
    for epoch in range(Last_epoch + 1, Epoch_nums):
        start = time.time()
        lr = net.optimizer.state_dict()['param_froups'][0]['lr']    # 获取当前的学习率
        loss = 0.

        net.train()
        for i, data in enumerate(dataloader_train):

            net.set_input()   # 解包数据，放进去图片和标志点的信息
            net.optimize()  # 怎么以上来就优化？？
        
    
        
    

    # 训练完成，保存模型
    print("结束了")
    torch.save(net.state_dict(), model_save_path + 'mymodel.pth')

if __name__ == '__main__':
    main()
