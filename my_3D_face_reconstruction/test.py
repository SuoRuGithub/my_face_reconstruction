import torch, random, random, os
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

root = r'data/FaceSample'

def make_dataset(root):
    images = []
    for root, _, fnames in sorted(os.walk(root, followlinks = True)):    # os.walk函数用来遍历目录中的文件，返回的是(root, dirs, files)一个三元组，即当前目录的地址，当前目录下所有目录的名字以及当前目录下文件的名字
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
batch_size = 1

def make_dataloader(datasets):
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(datasets, 
                                         batch_size=batch_size,  # 批量大小
                                         shuffle=True,           # 是否打乱数据集
                                         num_workers=5 # 使用多个线程加载数据的工作进程数
                                        )
    

train_root = r'data/train'
val_root = r'data/val'
image_size = 224
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





if __name__ == '__main__':
    main()