'''
准备数据，包括读取人脸图像，生成关键点的坐标值
'''
import os 
import numpy as np
import argparse
from util.detect_lm68 import detect_68p,load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list
import warnings
warnings.filterwarnings("ignore") 

def data_prepare(folder_list,mode):
    # 下面这行代码用来加载检测器检测人脸的关键点
    lm_sess,input_op,output_op = load_lm_graph('./checkpoints/lm_model/68lm_detector.pb')

    # 读取数据，生成68个关键点和皮肤掩码
    for img_folder in folder_list:
        detect_68p(img_folder,lm_sess,input_op,output_op) # detect landmarks for images
        get_skin_mask(img_folder) # 生成图像的皮肤掩码

    # 皮肤掩码是一张二值图，标记了皮肤的位置，下面的代码获得所有皮肤掩码图的路径，放在列表中
    # create files that record path to all training data
    msks_list = []
    for img_folder in folder_list:
        path = os.path.join(img_folder, 'mask')
        msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or 
                                                    'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask/', '') for i in msks_list] # 获得图片的路径列表
    # 接下来是获得关键点的路径，把mask换成landmarks，然后在把后缀改成.txt
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]  
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]  
    
    lms_list_final, imgs_list_final, msks_list_final = check_list(lms_list, imgs_list, msks_list) # check if the path is valid
    write_list(lms_list_final, imgs_list_final, msks_list_final, mode=mode) # save files


if __name__ == 'main':
    print('Datasets: ')
    data_prepare()