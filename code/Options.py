from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import functools
from data.image_folder import make_dataset
import os
from config_global import *

def My_dataload(index):
    # target_attack = True
    # universal_attack = False
    # path_root = 'E:\\Study\\dataset\\SALICON\\salicon_data\\' 
    # path_root = '/home/che-z/docker-czh/datasets/SALICON/'    
    # dir_A = 'images/val/'
    dir_A = path_root_A
    # dir_A = os.path.join(path_root, dir_A)
    A_paths = sorted(make_dataset(dir_A))
    # dir_B = 'maps/val/'
    dir_B = path_root_B
    # dir_B = os.path.join(path_root, dir_B)
    B_paths = sorted(make_dataset(dir_B))
    # dir_C = 'fixations_img/val/'
    dir_C = path_root_C
    # dir_C = os.path.join(path_root, dir_C)
    C_paths = sorted(make_dataset(dir_C))

    dataset_size = len(A_paths) 

    A_path = A_paths[index] 

    if(Serial_Minibatch_Attack==True):
        A_path = '/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/Ensemble_2/test_23/150_0.0002_50_perturbed_data.png'

    A = Image.open(A_path).convert('RGB') 
    # A = Image.open(A_path).convert('L') 
    # A = Image.open(A_path)
    # A = Image.open(A_path)
    A = A.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
    # A = A.resize((640, 480))
    # A.save("D:\\Study\\code\\Pytorch-AdvExm\\CIFAR10\\" + "image_inner" + ".png", "png")
    A = np.array(A)
    A = A / 255.0
    A_tensor = torch.from_numpy(A)

    # print("A_tensor is :", A_tensor, A_tensor.shape)
    # shan = tensor2saveimg(A_tensor, "image_inner")
    
    B_path = B_paths[index] 
    if(target_attack):
        B_path = B_paths[index + 3]
    if(universal_attack):
        B_path = B_paths[29] # 29(human) or 204(airplane)
    # B = Image.open(B_path).convert('RGB') 
    B = Image.open(B_path)
    B = B.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
    B = np.array(B)
    B = B / 255.0
    B_tensor = torch.from_numpy(B)

    C_path = C_paths[index] 
    if(target_attack):
        C_path = C_paths[index + 3] 
    if(universal_attack):
        C_path = C_paths[29] # 29(human) or 204(airplane)
    # C = Image.open(C_path).convert('RGB') 
    C = Image.open(C_path)
    C = C.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
    C = np.array(C)
    C = C / 255.0
    C_tensor = torch.from_numpy(C)

    # D_path = A_paths[index + 3] # this is for target attack
    # D_path = A_paths[24] # this is for universarial attack
    D_paths = A_paths
    if(target_attack):
        D_path = D_paths[index + 3]
    if(universal_attack):
        D_path = D_paths[29] # 29(human) or 204(airplane)
    D = Image.open(D_path).convert('RGB') 
    D = D.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
    D = np.array(D)
    D = D / 255.0 
    D_tensor = torch.from_numpy(D)

    E_path = '/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/Ensemble_2/test_19/150_0.0002_image.png'
    E = Image.open(E_path).convert('RGB') 
    E = E.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
    E = np.array(E)
    E = E / 255.0
    E_tensor = torch.from_numpy(E)


    input_dict = {'image': A_tensor, 'smap': B_tensor, 'fixation': C_tensor, 'targetimg': D_tensor, 'cleanimg': E_tensor}

    return input_dict


def tensor2saveimg(input, svname): # for tensor, visualization by saving as image, make sure that the input is a 4-dims tensor, and svname is a string
    min1 = torch.min(input)
    max1 = torch.max(input)
    input = (input - min1) / (max1 - min1 + 1e-8)
    input = input.squeeze(0) # for 4-dims input tensor
    input = input.cpu()
    input = input.detach().numpy()
    # print("input size :", input.size())
    input = np.transpose(input, (1,2,0)) * 255.0
    input = np.clip(input, 0, 255)
    input_sm = Image.fromarray(input.astype('uint8'))
    # input_sm.save("D:\\Study\\code\\Pytorch-AdvExm\\CIFAR10\\visresults\\" + svname + ".png", "png")
    input_sm.save(root_sv_path + svname + ".png", "png")

    return 0