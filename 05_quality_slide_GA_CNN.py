import argparse
import os
import time
import pickle

import logging
import torch
import numpy as np
# from models.resnet_cbam import *
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

home_dir = os.environ['HOME']
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='VacuumDexterousGrasp')
"""
    run$ python -m visdom.server (-port XXXX)    to activate Visdom before train/test
    python -m visdom.server -env_path ~/VacuumGrasp/vacuum-pointnet/NeuralNetwork/assets/logs/
"""
"""
    These parameters should be changed in different projects
"""
port_num = 8031
# python -m visdom.server -port 8031 -env_path ~/Dexterous_grasp_01/NeuralNetwork/assets/learned_models/Linr/2021011401/data

model_path = home_dir + '/Dexterous_grasp_01/NeuralNetwork/assets/learned_models/Linr/2021011401'
# model_path = home_dir + '/Dexterous_grasp_01/NeuralNetwork/assets/learned_models/Linr/2021050301'

path_grasps = home_dir + '/Dexterous_grasp_01/dataset'
path_pcs = home_dir + '/Dexterous_grasp_01/dataset'
parser.add_argument('--model-path', type=str, default=model_path)
parser.add_argument('--grasps-path', type=str, default=path_grasps, help='grasps path')
parser.add_argument('--pcs-path', type=str, default=path_pcs, help='point clouds path')
parser.add_argument('--test-grasps-path', type=str, default=path_grasps + '/test/000', help='test grasps path')
parser.add_argument('--batch-size', type=int, default=5)
args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available else False

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/'):
            for name in files:
                str = file_dir_ + '/' + name
                file_list.append(str)
    file_list.sort()
    return file_list

class Circle(object):
    def __init__(self, x=0.0, y=0.0, r=1.0):
        self.x = 1.0 * x
        self.y = 1.0 * y
        self.r = 1.0 * r

def calarea(c1, c2):
    dis = ((c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y)) ** 0.5
    if c1.r + c2.r < dis + 1e-10:
        return 0 # 两圆相离
    if c1.r - c2.r > dis - 1e-10: # 两园相含 c1包含c2
        return np.pi * c2.r * c2.r
    if c2.r - c1.r > dis - 1e-10: # 两园相含 c2包含c1
        return np.pi * c1.r * c1.r
    #两圆相交
    angle1 = np.arccos((c1.r * c1.r + dis * dis - c2.r * c2.r) / (2 * dis * c1.r))
    angle2 = np.arccos((c2.r * c2.r + dis * dis - c1.r * c1.r) / (2 * dis * c2.r))
    return c1.r * c1.r * angle1 + c2.r * c2.r * angle2 - np.sin(angle1) * c1.r * dis


def main():

    model_list_all = get_file_name(args.model_path)
    model_numbers = model_list_all.__len__()
    for i in range(0, model_numbers):
        if "ep200." in model_list_all[i]:
        # if "best_ave_loss." in model_list_all[i]:
            idx_model = 1 * i
            break
    model = torch.load(model_list_all[idx_model]).to(args.device)
    model = model.cpu()
    model_cuda = torch.load(model_list_all[idx_model]).cuda()
    # model = ResidualNet_Linr('ImageNet', 1, 80, att_type='CBAM')
    print('load model: {}'.format(model_list_all[idx_model]))

    # print(model)
    h_img = 41 * 2
    w_img = 41 * 2
    size_map = h_img - 24
    radius_gripper = 0.025
    data_pc = 4.0 * radius_gripper * np.ones([1, 1, h_img, w_img], dtype=np.float32)
    # data_pc = torch.from_numpy(data_pc).cuda()
    data_center = np.zeros([1, 1, 24, 24], dtype=np.float32)
    data_pc[0, 0, h_img//2-12:h_img//2+12, w_img//2-12:w_img//2+12] = data_center[0, 0, :, :]

    arr_quality = np.zeros([size_map, size_map])
    X = np.zeros([size_map, size_map])
    Y = np.zeros([size_map, size_map])

    step_radius = radius_gripper / 24.0
    offset = -1.0 * step_radius * float(h_img//2)
    step_slide = -offset * 2 / float(size_map)

    for i in range(0, size_map):
        for j in range(0, size_map):
            print(i*size_map + j)
            # X[i, j] = -0.5 * radius_gripper + float(i)/float(size_map)*radius_gripper
            # Y[i, j] = -0.5 * radius_gripper + float(j)/float(size_map)*radius_gripper
            X[i, j] = offset + float(i) * step_slide
            Y[i, j] = offset + float(j) * step_slide
            torch_pc = torch.from_numpy(data_pc[:, :, i:i+24, j:j+24]).cuda()
            s = model_cuda(torch_pc)
            arr_quality[i, j] = s.cpu().detach().numpy()
    print(arr_quality)
    np.save('quality_slide_GA_CNN.npy', arr_quality)
    '''
    max_value = np.max(np.max(arr_quality))
    min_value = np.min(np.min(arr_quality))

    max_range = 1.0 * max_value
    min_range = 0.15

    arr_quality = min_range + (max_range-min_range) * (arr_quality-min_value) / (max_value-min_value)
    print(arr_quality)
    print(np.min(np.min(arr_quality)), np.max(np.max(arr_quality)))
    '''

    line_width = 2
    font_size = 10
    fig, ax_1 = plt.subplots()
    ax = Axes3D(fig)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    surf = ax.plot_surface(X, Y, arr_quality, rstride=1, cstride=1, cmap='rainbow')
    # ax.set_xlabel("X (deg)", fontsize=font_size)
    # ax.set_ylabel("Y (deg)", fontsize=font_size)
    # ax.set_zlabel("Grasp quality", fontsize=font_size)
    # ax.set_zlim(0.1, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.set_size_inches(3.30, 2.02)  # width, height
    fig.set_dpi(300)
    plt.show()

    return True


if __name__ == "__main__":
    main()
