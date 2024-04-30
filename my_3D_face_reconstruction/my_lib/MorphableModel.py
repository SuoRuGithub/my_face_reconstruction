'''
这个脚本是和3DMM相关的内容
我们在这个脚本里写了一个ParametircFace的类，这个类导入了bfm的平均人脸模型，可以通过给定的系数计算出三维人脸模型，可以计算三维人脸模型的投影
可以计算人脸图像的关键点坐标
'''
import numpy as np
from scipy.io import loadmat
import torch 
import torch.nn.functional as F
from util.load_mats import transferBFM09
import os

class ParametricFace:
    def __init__(self, 
                bfm_folder='./BFM', 
                recenter=True,
                camera_distance=10.,
                init_lit=np.array([
                    0.8, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                focal=1015.,    # 焦距
                center=112.,    
                is_train=True,
                default_name='BFM_model_front.mat'):
        
        '''if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)'''
        
        model = loadmat(os.path.join(bfm_folder, default_name)) # loadmat用来加载一个.mat的人脸模型
        # mean face shape. [3*N,1]   其中，N是顶点的个数，所以是[3*N, 1]这么个形状
        self.mean_shape = model['meanshape'].astype(np.float32) # 平均人脸形状
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)     # 平均人脸纹理
        # texture basis. [3*N,80]   一个顶点一个颜色，而颜色用rgb表示，所以这里是3 * N
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]           ？？？
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1   # 顶点对应的面，减一是为了让索引从0开始
        # vertex indices for 68 landmarks. starts from 0. [68,1]    
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1    # 每个关键点对应的顶点的索引，为了从零计数，需要减去个1
        
        if is_train:
            # 我们没写那个误差，略
            pass
        if recenter:
            pass    # ?
        # 下面这个函数的作用是将该类的所有参数都从np数组转换为torch.tensor
        def to(self, device):
            self.device = device    
            for key, value in self.__dict__.items():    # self.__dict__存储了类的所有属性，在这里，我们使用.item方法来遍历所有的键值对
                if type(value).__module__ == np.__name__:   # 通过比较类型的模块名和numpy的模块名来判断类的属性是不是np数组
                    setattr(self, key, torch.tensor)    # 这个函数的作用是设置类的属性的数据类型
        
        def set_coeff(alpha, beta, gamma, delta):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
    
        def compute_shape(self):
            '''
            人脸形状（或者说最终建模出的三维图像中的顶点信息），是由形状和表情两个因素决定的
            这个函数会接受两个参数：形状参数alpha(B, 80)和表情参数beta(B, 64)，然后计算三维模型的顶点信息(B, N, 3)，其中N是顶点个数，3代表每个顶点的三个坐标
            '''
            batch_size = self.alpha.shape[0]
            identity = torch.einsum('ij,aj->ai', self.id_base, self.alpha)   # id_base: [3n, 80], alpha: [b, 80] -> identity: [b, 3n]
            expression = torch.einsum('ij,aj->ai', self.exp_base, self.beta) # exp_base: [3n, 64], beta: [b, 64] -> expression: [b, 3n]  # 这个函数还是挺好用的
            shape = identity + expression + self.mean_shape.reshape([1, -1])    # [b, 3n] + [b, 3n] + [3n, 1].reshape([1, -1])  #(事实上，[b, 3n] + [1, 3n]的时候会广播)
            return shape.reshape([batch_size, -1, 3]) 
        def compute_texture(self, normalize = True):
            '''
            人脸纹理，参数delta[b, 80]，最后输出人脸纹理[b, n, 3]
            '''
            batch_size = self.delta.shape[0]
            texture = torch.einsum('ij,aj->ai', self.tex_base, self.delta)    # tex_base[3n, 80], delta[b, 80] -> texture[b, 3n]
            texture = texture + self.mean_tex
            if normalize:
                face_texture = face_texture / 255.  # 这是干啥？
            return face_texture.reshape([batch_size, -1, 3])
        def compute_norm(self, face_shape):
            '''
            这个函数用来计算每个顶点对应的法向量
            顶点对应的法向量的定义是，将共享该顶点的三个边的法向量相加，再标准化得到的结果
            参数是人脸形状[B, N, 3]，也就是每个顶点的坐标
            返回所有顶点的法向量[B, N, 3]

            这会用到一点点数学知识，求法向量只需求出三角形的两个边，再作叉积就可以了，但是要标准化
            '''
            # face_shape: [B, N, 3], face_buf: [F, 3]
            # 我详细解释一下下面几行代码的意思，self.face_buf[:, 0]的作用是提取face_buf的第二个维度的第零个元素，换句话说，提取每个面的第一个顶点的索引
            # 所以self.face_buf[:, 0]是一个 [F, ]的向量
            # 然后我们再看face_shape[:, self.face_buf[:, 0]],这句代码的意思是：按照self.face_buf[:, 0]提供的索引从face_shape中的第二个维度（顶点索引）提取
            # 元素，所以经过这样的操作后，最终我们提取了每一个面的第0个顶点的坐标值，最后得到的向量的形状是[B, F, 3]
            v1 = face_shape[:, self.face_buf[:, 0]]
            v2 = face_shape[:, self.face_buf[:, 1]]
            v3 = face_shape[:, self.face_buf[:, 2]]
            
            # 然后我们可以计算每个边的两条边向量    [B, F, 3]
            e1 = v1 - v2
            e2 = v2 - v3
            # 叉积计算法向量
            face_norm = torch.cross(e1, e2, dim = -1)   # 在最后一个维度（3，表示向量坐标）作叉积，
            face_norm = F.normalize(face_norm, dim = -1, p = 2) # 使用二范数标准化，得到了面法向量[B, F, 3]
            face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)    # 我没看懂，为什么要新加一个点呢？
            # 我再来详细解释一遍下面这句话
            # self.point_buf是一个[N, 8]的向量，共享一个顶点的边有8个（？）
            # self.point_buf提取这八个索引
            # 然后我们从face_norm中提取对应的八个面对应的法向量，然后把它们加起来
            vertex_norm = torch.sum(face_norm[:, self.point_buf], dim = 2)
            vertex_norm = F.normalize(vertex_norm, dim = -1, p = 2)
            #### 这个计算顶点法向量的函数我看了有半个小时，真是要命啊
            return vertex_norm
        def compute_color(self, face_texture, face_norm):
            '''
            face_texture    [B, N, 3]
            face_norm       [B, N, 3]
            gamma           [B, 27]         # 这是一个球谐函数？什么鬼啊天哪
            '''
            batch_size = self.gamma.shape[0]
        