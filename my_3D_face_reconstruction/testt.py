# 测试一下神经网络的相关实现成功了没有
# 我要疯了，这个破包死活都导入不进来
# import sys 
# sys.path.append('/home/stu0/myCode/my_3D_face_reconstruction/my_lib')
# import my_lib.model

import my_lib.model as model        # 我不理解为什么我必须要这样才能把我的包引入，失笑着

import torchvision.transforms as transforms
from PIL import Image


image_path = r'/home/stu0/myCode/my_3D_face_reconstruction/my_lib/img_for_test.jpg'

def main():
    image = Image.open(image_path)
    tensor_image = transforms.ToTensor()(image)

    print(tensor_image.shape)

    RecogNet = model.RecogNet()
    ReconNet = model.myNN()


if __name__ == "__main__":
    main()
