'''
这里我们定义了神经网络（论文中的R_Net），并写了关于提取深度特征的模型
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from my_lib.loss import per_loss, landmark_loss
from my_lib.arcface_torch.backbones import get_model
from my_lib.MorphableModel import ParametricFace

LR = 5e-5
params = 239
w_per = 0.2
w_lan = 1.6e-3

Last_epoch = -1

recog_net_path = r'checkpoints/recog_model/ms1mv3_arcface_r50_fp16'

# 这是我们用来提取图片的深层特征的神经网络，我们写一个初始化的方法，使得我们能够解包神经网络，再写一个前向传播的函数，就好了
# 我真是无了个语，这破东西有那么难吗？你一天在忙活什么？咱要是这么个水平就不要念书了
class RecogNet(nn.Module):
    def __init__(self, recog_net_path = recog_net_path, input_size = 224):
        super(RecogNet, self).__init__()
        '''解包预训练的，用来提取深度特征的神经网络，并将其设置为评估模式'''
        # 这里要用的是Arcface的模型，模型的相关定义文件应该是在lib/arcface_torch里面，我们直接拿来用就好了
        net = get_model(name = 'r50', fp16 = False)
        state_dict = torch.load('checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth', map_location = 'cpu')
        net.load_state_dict(state_dict)
        print("恭喜你，成功解包了预训练的模型")
        
        net.eval()  # 调整为评估模式
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2* x - 1    # 匿名函数，接受一个x参数，返回2x - 1，这句话的作用应该是将[0, 1]的像素值标准化为[-1, 1]
        self.input_size = input_size    #\

    def forward(self, image):
        feature = F.normalize(self.net(image),dim = -1, p = 2)
        return feature


# 这是用来重建人脸的神经网络的定义
class ReconNet(nn.Module):
    def __init__(self, custom_num = params):
        super(ReconNet, self).__init__()    # 调用基类的构造函数
        self.model = models.resnet50(pretrained = True)
        ## 下面两句是为了将resnet的最后的全连接层变成289个
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, custom_num)
        
    def forward(self, input_img):
        coeff = self.model(self.input_img)
        return coeff


# 下面这个类封装了计算输出，反响传播等功能
class myModel:
    def __init__(self, ReconNet, RecogNet):
        self.ReconNet = ReconNet
        self.RecogNet = RecogNet
        
        # 计算两个loss的方法
        self.compute_per_loss = per_loss
        self.compute_landmark_loss = landmark_loss

        # optimizer 属性
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 32, last_epoch = Last_epoch)      # 这个应该是用模拟退火调整学习率的

        # 3DMM模型
        self.face_model = ParametricFace()

    def set_input(self, input_image, gt_lm):
        '''这个函数里，我们设置下面这些输出：
        * 输入的图片
        * 图片对应的关键点
        * '''
        self.input_img = input_image
        self.gt_lm = gt_lm
    def forward(self):
        coeff = self.ReconNet(self.input_img)
        self.pre_coeff = coeff

        self.alpha = coeff[: 80]
        self.beta = coeff[80 : 144]
        self.gamma = coeff[]
        self.delta = coeff[114 : 224]

        self.face_model.set_coeff(self.alpha, self。beta, self.gamma, self.delta)   

        



    def predict(self):
        # 通过已经计算了的系数生成人脸并且生成人脸的二维图像并生成预测人脸的关键点坐标
        self.pred_face = self.face_model.compute_face()
        self.pred_lm = self.face_model.compute_landmark()

    def compute_losses(self):
        # 这个函数用来计算损失函数

        # 感知损失函数部分
        ## 首先使用提前训练好的神经网络直接计算输入图像和预测图像的深度特征
        pred_feat = self.RecogNet(self.pred_face)
        gt_feat = self.RecogNet(self.input_img)
        perloss = self.compute_per_loss(pred_feat, gt_feat)

        # 关键点损失函数部分
        lanloss = self.compute_landmark_loss(self.pred_lm, self.gt_lm)
        
        total_loss = w_lan * lanloss + w_per * perloss
        self.losses = total_loss
    def optimize(self):
        # 这个函数计算损失函数和梯度并更新参数
        self.forward()
        self.compute_losses()
        self.optimizer.zero_grad()
        self.losses.backward()
        self.optimizer.step()
    def save_mesh(self, name):
        # 
        pass





"""
class myNN(nn.Module):
    '''
    把我们的神经网络封装成一个类
    属性：model---一个修改过的resNet50
         losses---神经网络目前的损失函数
         pre_coeff---神经网络预测的系数
         optimizer---我们使用的优化器（Adam）
    方法：
        comput_losses---计算损失函数
        optimize---计算损失，计算梯度，优化神经网络参数
    '''
    def __init__(self, custom_num = params):
        super(myNN, self).__init__()    # 调用基类的构造函数
        # model 属性
        self.model = models.resnet50(pretrained = True)
        ## 下面两句是为了将resnet的最后的全连接层变成289个
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, custom_num)
        
        # 计算两个loss的方法
        self.compute_per_loss = per_loss
        self.compute_landmark_loss = landmark_loss

        # optimizer 属性
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 32, last_epoch = Last_epoch)      # 这个应该是用模拟退火调整学习率的

        # 用来计算特征的神经网络
        self.net_recog = wrap_recognet(recog_net_path)


    def set_input(self, input_image, gt_lm):
        '''这个函数里，我们设置下面这些输出：
        * 输入的图片
        * 图片对应的关键点
        * '''
        self.input_img = input_image
        self.gt_lm = gt_lm

    def forward(self):
        coeff = self.model(self.input_img)
        self.pre_coeff = coeff
    
    def setup(self):
        pass
    def compute_losses(self, input_image):
        # 这个函数用来计算损失函数

        # 感知损失函数部分
        ## 首先使用提前训练好的神经网络直接计算输入图像和预测图像的深度特征
        
        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        perloss = self.compute_per_loss(pred_feat, gt_feat)

        # 关键点损失函数部分
        lanloss = self.compute_landmark_loss(predict_lm, self.gt_lm)
        
        total_loss = w_lan * lanloss + w_per * perloss
        self.losses = total_loss
  
    def optimize(self, isTrain = True):
        # 这个函数计算损失函数和梯度并更新参数
        self.forward()
        self.compute_losses()
        self.optimizer.zero_grad()
        self.losses.backward()
        self.optimizer.step()    
"""