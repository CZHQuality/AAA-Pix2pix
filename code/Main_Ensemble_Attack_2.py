# CUDA_VISIBLE_DEVICES=1 python3 Main_Ensemble_Attack.py

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
# from Attack_methods_library import saliency_attack_1_start, saliency_attack_1 as saliency_attack_start, saliency_attack
# from Attack_methods_library import attack_method_1 as attack_used # select attack method from library
# from Loss_functions import CCLoss, KLLoss, NSSLoss
from Loss_functions import * 
from Options import My_dataload, tensor2saveimg
from Pretrained_models import GazeGAN_1, SALICON_2, Globalpix2pix, GazeGAN_2, DCN_LSTM_1, DCN_2, SAM_VGG_1, SAM_VGG_2, SAM_ResNet, GazeGAN_CSC, SalGAN_BCE, DCN_Inception, DeepGaze_only_VGG, DCN_SAM_VGG

if(Used_attack_method_idx==1):
    from Attack_methods_library import attack_method_1 as attack_used
if(Used_attack_method_idx==2):
    from Attack_methods_library import attack_method_2 as attack_used
if(Used_attack_method_idx==3):
    from Attack_methods_library import attack_method_3 as attack_used
if(Used_attack_method_idx==4):
    from Attack_methods_library import attack_method_4 as attack_used
if(Used_attack_method_idx==5):
    from Attack_methods_library import attack_method_5 as attack_used
if(Used_attack_method_idx==6):
    from Attack_methods_library import attack_method_6 as attack_used
if(Used_attack_method_idx==7):
    from Attack_methods_library import attack_method_7 as attack_used
if(Used_attack_method_idx==8):
    from Attack_methods_library import attack_method_8 as attack_used
if(Used_attack_method_idx==9):
    from Attack_methods_library import attack_method_9 as attack_used
if(Used_attack_method_idx==10):
    from Attack_methods_library import attack_method_10 as attack_used
if(Used_attack_method_idx==11):
    from Attack_methods_library import attack_method_11 as attack_used
if(Used_attack_method_idx==12):
    from Attack_methods_library import attack_method_12 as attack_used
if(Used_attack_method_idx==13):
    from Attack_methods_library import attack_method_13 as attack_used   
if(Used_attack_method_idx==14):
    from Attack_methods_library import attack_method_14 as attack_used   
if(Used_attack_method_idx==15):
    from Attack_methods_library import attack_method_15 as attack_used   
if(Used_attack_method_idx==16):
    from Attack_methods_library import attack_method_16 as attack_used   

###############################################################################
# Global Vars
'''
epsilons = [ .0001] # step size of gradient update
# epsilons = [ .0002]
iter_num = 31 # The total amount of iterations
start_idx = 150 # the start idx of source images to be attacked
end_idx = 152 # the end idx of source images to be attacked

pretrained_model = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel/PretrainedModel/MyGazeGAN/latest_net_G.pth"
root_sv_path = "/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/GazeGAN/ensemble_targeted_1/"
'''
loss_record = []
loss_txt_path = root_sv_path
file = open(loss_txt_path + "lossrecord.txt", "w")

use_cuda=True
'''
non_target_attack = False
target_attack = False
universal_attack = True
'''
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network 
# model_1 = GazeGAN_1().to(device)
model_1 = SAM_VGG_1().to(device)
# model_1 = GazeGAN_1()
# model_1 = nn.DataParallel(model_1, device_ids=[0,1]) 
print("The structure of model_1 is:", model_1)
# model_2 = SALICON_2().to(device)
# model_2 = Globalpix2pix().to(device)
model_2 = SalGAN_BCE().to(device)
# model_2 = Globalpix2pix()
# model_2 = nn.DataParallel(model_2, device_ids=[0,1]) 
print("The structure of model_2 is:", model_2)
# model_3 = Globalpix2pix().to(device)

# Load the pretrained model
# model_1.load_state_dict(torch.load(pretrained_model_GazeGAN_1, map_location='cpu'))
model_1.load_state_dict(torch.load(pretrained_model_SAM_VGG_1, map_location='cpu'))
# model_2.load_state_dict(torch.load(pretrained_model_SALICON_2, map_location='cpu'))
# model_2.load_state_dict(torch.load(pretrained_model_Globalpix2pix, map_location='cpu'))
model_2.load_state_dict(torch.load(pretrained_model_SalGAN_BCE, map_location='cpu'))
# model_3.load_state_dict(torch.load(pretrained_model_Globalpix2pix, map_location='cpu'))

# Set the model in evaluation mode.
model_1.eval()
model_2.eval()
# model_3.eval()

# def test( model, device, epsilon ):
def test( model1, model2, device, epsilon ):

    adv_examples = []
    # Torch_downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    # Torch_upsp = nn.Upsample(scale_factor=8, mode='bilinear')
    for index_input in range(start_idx, end_idx): # idx of source images to be attacked in the dataset
        input_dict = My_dataload(index_input)
        data = input_dict['image']
        target = input_dict['smap'] # for L1, CC and NSS
        target_NSS = input_dict['fixation']
        data_target = input_dict['targetimg']
        cleanimg = input_dict['cleanimg']
        # print("data is :", data, data.shape)
        # print("target is :", target, target.shape)
        data_2 = data.numpy()
        data_target_2 = data_target.numpy()
        target_2 = target.numpy()
        target_NSS_2 = target_NSS.numpy()
        cleanimg_2 = cleanimg.numpy()
        # data_3 = data_2[:, :, 10:38, 10:38] # data.size() = torch.Size([1, 1, 64, 64]), pay attention : we must crop a 28X28 patch from our own data to satisfy the pretrained Lenet (fully connected layer)
        # print("data_2 :", data_2.shape)
        data_2 = np.transpose(data_2, (2,0,1)) # notice this step and the following expand_dims step are very important when using PIL load image as RGB
        data_target_2 = np.transpose(data_target_2, (2,0,1)) 
        cleanimg_2 = np.transpose(cleanimg_2, (2,0,1)) 
        # print("data_2 :", data_2.shape)
        data_2 = np.expand_dims(data_2, axis=0)
        data_target_2 = np.expand_dims(data_target_2, axis=0)
        cleanimg_2 = np.expand_dims(cleanimg_2, axis=0)
        # print("data_2 :", data_2.shape)
        # data_2 = np.resize(data_2, (1, 3, 480, 640))
        target_2 = np.resize(target_2, (1, 3, Unit_Height, Unit_Width))
        target_NSS_2 = np.resize(target_NSS_2, (1, 3, Unit_Height, Unit_Width))
        data = torch.from_numpy(data_2)
        data_target = torch.from_numpy(data_target_2)
        target = torch.from_numpy(target_2)
        target_NSS = torch.from_numpy(target_NSS_2)
        cleanimg = torch.from_numpy(cleanimg_2)
        # print("data is :", data, data.size())
        data = data.float()
        data_target = data_target.float()
        # print("data is :", data, data.size())
        target = target.float()
        target_NSS = target_NSS.float()
        cleanimg = cleanimg.float()

        # Send the data and label to the device
        data, target, target_NSS, data_target, cleanimg = data.to(device), target.to(device), target_NSS.to(device), data_target.to(device), cleanimg.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        cleanimg.requires_grad = True #!!!!! very important!!!

        # Forward pass the data through the model
        output_1, d6_1, d5_1 = model_1(data)
        output_target_1, d6_target_1, d5_target_1 = model_1(data_target)
        min1 = torch.min(output_1)
        max1 = torch.max(output_1)
        output_1 = (output_1 - min1) / (max1 - min1 + 1e-8)
        init_pred_1 = output_1
        # print("target is :", target.size())

        output_2, d6_2, d5_2 = model_2(data)
        output_target_2, d6_target_2, d5_target_2 = model_2(data_target)
        min2 = torch.min(output_2)
        max2 = torch.max(output_2)
        output_2 = (output_2 - min2) / (max2 - min2 + 1e-8)
        init_pred_2 = output_2

        '''
        output_3, d6_3, d5_3 = model_3(data)
        output_target_3, d6_target_3, d5_target_3 = model_3(data_target)
        min3 = torch.min(output_3)
        max3 = torch.max(output_3)
        output_3 = (output_3 - min3) / (max3 - min3 + 1e-8)
        init_pred_3 = output_3
        '''

        output = output_1 + output_2
        if(feature_space_attack==True):
            d6 = d6_1 + d6_2
            d6_target = d6_target_1 + d6_target_2
        else:
            d6 = torch.zeros(d6_1.size())
            d6_target = torch.zeros(d6_target_1.size())
        # d5 = d5_1 + d5_2
        # d5_target = d5_target_1 + d5_target_2

        # Record initial target image, target saliency map, and original source image
        shan = tensor2saveimg(target, str(index_input) + "_" + str(epsilon) + "_target")
        shan = tensor2saveimg(init_pred_1, str(index_input) + "_" + str(epsilon) + "_init_pred_1")
        shan = tensor2saveimg(init_pred_2, str(index_input) + "_" + str(epsilon) + "_init_pred_2")
        # shan = tensor2saveimg(init_pred_3, str(index_input) + "_" + str(epsilon) + "_init_pred_3")
        shan = tensor2saveimg(data, str(index_input) + "_" + str(epsilon) + "_image")

        path_ori_sm = root_sv_path + str(index_input) + "_" + str(epsilon) + "_init_pred_1.png"
        ori_saliency_map = Image.open(path_ori_sm).convert('RGB') 
        ori_saliency_map = ori_saliency_map.resize((Unit_Width, Unit_Height),Image.ANTIALIAS) 
        ori_saliency_map = np.array(ori_saliency_map)
        ori_saliency_map = ori_saliency_map / 255.0
        ori_saliency_map = torch.from_numpy(ori_saliency_map)
        ori_saliency_map = ori_saliency_map.numpy()
        ori_saliency_map = np.transpose(ori_saliency_map, (2,0,1))
        ori_saliency_map = np.expand_dims(ori_saliency_map, axis=0)
        ori_saliency_map = torch.from_numpy(ori_saliency_map)
        ori_saliency_map = ori_saliency_map.float()
        ori_saliency_map = ori_saliency_map.to(device)
        
        original_source_image = data
        if(Serial_Minibatch_Attack==True):
            original_source_image = cleanimg
        '''
        if(Serial_Minibatch_Attack==True):
            original_source_image = Image.open('/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/Ensemble_2/test_19/150_0.0002_image.png').convert('RGB')  #XXXXXXXXXXXXXXXXXXXXX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            original_source_image = original_source_image.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
            original_source_image = np.array(original_source_image)
            original_source_image = original_source_image / 255.0
            original_source_image = torch.from_numpy(original_source_image)
            original_source_image = original_source_image.numpy()
            original_source_image = np.transpose(original_source_image, (2,0,1))
            original_source_image = np.expand_dims(original_source_image, axis=0)
            original_source_image = torch.from_numpy(original_source_image)
            original_source_image = original_source_image.float()
            original_source_image = original_source_image.to(device)
        '''

        criterion_L1 = nn.L1Loss()
        criterion_CC = CCLoss(gpu_ids)
        criterion_KL = KLLoss(gpu_ids)
        criterion_NSS = NSSLoss(gpu_ids)
        if(use_L1_perceptual == True): 
            criterion_percp = nn.L1Loss()
        elif(use_L2_perceptual == True):
            criterion_percp = nn.MSELoss() # L2 loss is mean square error loss
        elif(use_SSIM_perceptual == True):
            criterion_percp = SSIM(gpu_ids)
        else:
            criterion_percp = nn.L1Loss() # default loss, but do not calculate its gradient, just for logging the loss
            # pass # for L_{infinity} constraint using very small gradient descend step

        if(feature_space_attack==True):
            # loss = wei_L1 * criterion_L1(output, target) + wei_CC * criterion_CC(output, target) + wei_KL * criterion_KL(output, target) + wei_NSS * criterion_NSS(output, target_NSS) + wei_Repre * criterion_KL(d6, d6_target) 
            loss = wei_Repre * criterion_KL(d6, d6_target)
        if(image_space_attack==True):
            # print("WARNING!!! This main function is used to generate adversarial attack from latent/feature space!!! Please use the main function dedicated for image-space attack!!!")
            loss = wei_L1 * criterion_L1(output, target) + wei_CC * criterion_CC(output, target) + wei_KL * criterion_KL(output, target) + wei_NSS * criterion_NSS(output, target_NSS) 
        if((Used_attack_method_idx == 12 or Used_attack_method_idx == 13) and feature_space_attack==True):
            criterion_percp = nn.MSELoss()
            loss = max(wei_KL * criterion_KL(d6, d6_target), 0)
        if((Used_attack_method_idx == 12 or Used_attack_method_idx == 13) and image_space_attack==True):
            criterion_percp = nn.MSELoss()
            loss = max(wei_KL * criterion_KL(output, target), 0)
        print("loss_first_step is :", loss)
        file.write(str(loss.item()) + '\n') #
        
        # Zero all existing gradients
        model_1.zero_grad()
        model_2.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True)

        # Collect datagrad
        data_grad = data.grad.data
        temp_prev_grad_init = 0
        
        # Call Saliency Attack Methods
        attack_perform = attack_used()
        perturbed_data, temp_prev_grad = attack_perform.saliency_attack_start(data, epsilon, data_grad, temp_prev_grad_init)
        # print("perturbed_data is :", perturbed_data, perturbed_data.shape)
        sum_mom_init = torch.pow(temp_prev_grad, 1)
        # print("temp_prev_grad and start sum_mom_init are :", temp_prev_grad, sum_mom_init, sum_mom_init.size())
        
        shan = tensor2saveimg(perturbed_data, str(index_input) + "_" + str(epsilon) + "_perturbed_data")

        # Re-predict the saliency map of the perturbed adversarial example
        output_per_1, d6_per_1, d5_per_1 = model_1(perturbed_data) # output saliency map of the last layer, and two intermediate feature maps.
        shan = tensor2saveimg(output_per_1, str(index_input) + "_" + str(epsilon) + "_peroutput_1")
        output_per_2, d6_per_2, d5_per_2 = model_2(perturbed_data) # output saliency map of the last layer, and two intermediate feature maps.
        shan = tensor2saveimg(output_per_2, str(index_input) + "_" + str(epsilon) + "_peroutput_2")

        # data_step = perturbed_data # notice that we cannont directly load the data_step as perturbed_data because we should set the data_step as "varaible", i.e. leaf node, so as to backward the gradient
        svname = str(index_input) + "_" + str(epsilon) + "_perturbed_data"
        path_per = root_sv_path + svname + ".png"
        print("path is :", path_per)
        data_step = Image.open(path_per).convert('RGB') 
        data_step = data_step.resize((Unit_Width, Unit_Height),Image.ANTIALIAS) 
        data_step = np.array(data_step)
        data_step = data_step / 255.0
        data_step = torch.from_numpy(data_step)
        data_step = data_step.numpy()
        data_step = np.transpose(data_step, (2,0,1))
        data_step = np.expand_dims(data_step, axis=0)
        data_step = torch.from_numpy(data_step)
        data_step = data_step.float()

        # epsilon_step = 0.002 # iteration step of loop
        # epsilon_step = 0.0001 # iteration step of loop
        epsilon_step = epsilons[0]
        # the above steps only generate the adversarial example by the first iteration 

        temp_prev_grad_step = temp_prev_grad # initial the temp_prev_grad_step using the 1st iteration grad
        # sum_mom_step = sum_mom_init * 0 
        # sum_mom_step_nodecay = sum_mom_init * 0
        sum_mom_step = torch.pow(sum_mom_init, 2)
        sum_mom_step_nodecay = torch.pow(sum_mom_init, 2)

        # for step in range(1, 100): # we set the iteration steps as 50
        for step in range(1, iter_num): # we set the iteration steps as 50

            data_step, target, target_NSS, data_target = data_step.to(device), target.to(device), target_NSS.to(device), data_target.to(device)
            data_step.requires_grad = True
            # output_step = model(data_step)
            output_step_1, d6_step_1, d5_step_1 = model_1(data_step)
            output_step_2, d6_step_2, d5_step_2 = model_2(data_step)

            output_step = output_step_1 + output_step_2
            # output_step = 0.5*output_step_1 + 0.5*output_step_2

            if(feature_space_attack == True):
                d6_step = d6_step_1 + d6_step_2
                # d5_step = d5_step_1 + d5_step_2
            else:
                d6_step = torch.zeros(d6_step_1.size())

            if(feature_space_attack == True and Used_attack_method_idx != 10 and Used_attack_method_idx != 12 and Used_attack_method_idx != 13):
                loss_step = wei_Repre * criterion_KL(d6_step, d6_target)  # data_step is the iteratively updated adv. example 
            if(image_space_attack == True and Used_attack_method_idx != 10 and Used_attack_method_idx != 12 and Used_attack_method_idx != 13):
                loss_step = wei_KL * criterion_KL(output_step, target)  # data_step is the iteratively updated adv. example 
            if(use_L1_perceptual == True and Used_attack_method_idx != 10 and Used_attack_method_idx != 12 and Used_attack_method_idx != 13): 
                loss_step = loss_step + wei_percep * criterion_percp(data_step, original_source_image)
                # print("original_source_image is", original_source_image)
                # print("data_step is", data_step)
                # print("data is", data)
            if(Used_attack_method_idx == 10 and image_space_attack == True): # Used_attack_method_idx == 10 is hot/cold attack, which uses different loss compared other attacks. Only supports image-space attack
                # print("WARNING!!!Hot/Cold Attack still does NOT support feature-space attack!!!Please use other attacks!!!")
                criterion_percp = SSIM(gpu_ids)
                loss_step = wei_KL * criterion_KL(output_step, target) - wei_KL * criterion_KL(output_step, ori_saliency_map) + wei_percep_SSIM * criterion_percp(data_step, original_source_image)
            if(Used_attack_method_idx == 10 and feature_space_attack == True): # Used_attack_method_idx == 10 is hot/cold attack, which uses different loss compared other attacks. Only supports image-space attack
                print("WARNING!!!Hot/Cold Attack still does NOT support feature-space attack!!!Please use other attacks!!!")
                # criterion_percp = SSIM(gpu_ids)
                # loss_step = wei_KL * criterion_KL(output_step, target) - wei_KL * criterion_KL(output_step, ori_saliency_map) + wei_percep_SSIM * criterion_percp(data_step, original_source_image)
            if((Used_attack_method_idx == 12 or Used_attack_method_idx == 13) and feature_space_attack == True): # extended C&W attack
                criterion_percp = nn.MSELoss() # L2 loss is mean square error loss
                loss_step = max(wei_KL * criterion_KL(d6_step, d6_target), 0) + B_CONST * criterion_percp(data_step, original_source_image)
            if((Used_attack_method_idx == 12 or Used_attack_method_idx == 13) and image_space_attack == True): # extended C&W attack
                criterion_percp = nn.MSELoss() # L2 loss is mean square error loss
                loss_step = max(wei_KL * criterion_KL(output_step, target), 0) + B_CONST * criterion_percp(data_step, original_source_image)

            if(Used_attack_method_idx == 12 or Used_attack_method_idx == 13):
                print("Attack Performance Loss and Perceptual Constraint Loss are:", wei_KL*criterion_KL(output_step, target), B_CONST*criterion_percp(data_step, original_source_image))
            else:
                print("Attack Performance Loss and Perceptual Constraint Loss are:", wei_KL*criterion_KL(output_step, target), wei_percep*criterion_percp(data_step, original_source_image))
            '''
            if(feature_space_attack == True):
                loss_step = wei_Repre * criterion_KL(d6_step, d6_target) + wei_percep * criterion_L1(data_step, original_source_image) # data_step is the iteratively updated adv. example 
            elif(Use_KL == True):
                loss_step = wei_KL * criterion_KL(output_step, target) + wei_percep * criterion_L1(data_step, original_source_image)
            elif(Use_CC == True):
                loss_step = wei_CC * criterion_CC(output_step, target) + wei_percep * criterion_L1(data_step, original_source_image)
            elif(Use_NSS == True):
                loss_step = wei_NSS * criterion_NSS(output_step, target_NSS)  + wei_percep * criterion_L1(data_step, original_source_image)
            elif(Use_L1 == True):
                loss_step = wei_L1 * criterion_L1(output_step, target) + wei_percep * criterion_L1(data_step, original_source_image)
            '''
            # print("KL loss and perceptual loss are:", wei_KL*criterion_KL(output_step, target), wei_percep*criterion_L1(data_step, original_source_image))
            # loss_step = wei_Repre * criterion_KL(d6_step, d6_target)  #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            # loss_step = wei_Repre * criterion_KL(d6_step, d6_target) + wei_Repre * criterion_CC(d6_step, d6_target) + wei_Repre * criterion_L1(d6_step, d6_target) 
            # loss_step = wei_L1 * criterion_L1(output_step, target) + wei_CC * criterion_CC(output_step, target) + wei_KL * criterion_KL(output_step, target) + wei_NSS * criterion_NSS(output_step, target_NSS) + wei_Repre * criterion_KL(d6_step, d6_target) 
            print("loss step is :", loss_step, step)
            # file.write(str(loss_step.item()) + '\n')
            file.write('total loss:' + str(loss_step.item()) + '\n')
            if(Used_attack_method_idx == 12 or Used_attack_method_idx == 13):
                file.write("attack performance loss:" + str(wei_KL*criterion_KL(output_step, target)) + "perceptual constraint loss" + str(B_CONST*criterion_percp(data_step, original_source_image)) + '\n')
            else:
                file.write("attack performance loss:" + str(wei_KL*criterion_KL(output_step, target)) + "perceptual constraint loss" + str(wei_percep*criterion_percp(data_step, original_source_image)) + '\n')
            
            model_1.zero_grad()
            model_2.zero_grad()

            # loss_step.backward()
            loss_step.backward(retain_graph=True)
            # print("data_step is :", data_step)
            # print("data_step.grad is :", data_step.grad)
            data_step_grad = data_step.grad.data
            
            if(Used_attack_method_idx == 1 or Used_attack_method_idx == 7 or Used_attack_method_idx == 9 or Used_attack_method_idx == 13 or Used_attack_method_idx == 15):
                # sum_mom_step = torch.pow(sum_mom_step, 1) + torch.pow(temp_prev_grad_step, 1)
                beta_2m = 0.9
                # sum_mom_step = beta_2m * torch.pow(sum_mom_step, 1) + (1 - beta_2m) * torch.pow(temp_prev_grad_step, 1)
                sum_mom_step = beta_2m * torch.pow(sum_mom_step, 1) + (1 - beta_2m) * torch.pow(temp_prev_grad_step, 2)
                # sum_mom_step = beta_2m * torch.pow(sum_mom_step, 1) + (1 - beta_2m) * torch.pow(data_step_grad, 1)
                # sum_mom_step = torch.pow(sum_mom_step, 2) + torch.pow(temp_prev_grad_step, 2) # you can choose 1-norm or squared
            elif(Used_attack_method_idx == 2 or Used_attack_method_idx == 8):
                beta_2m = 0.9
                sum_mom_step_nodecay = sum_mom_step_nodecay + torch.pow(data_step_grad, 2)
                average = ( sum_mom_step_nodecay ) / (step+1)
                # sum_mom_step = sum_mom_step / step # calculate the Expectation of Square Summation of past "step" iterations, so as to pay more attention to current gradient rather than past accumulated gradients
                sum_mom_step = beta_2m * average + (1 - beta_2m) * torch.pow(data_step_grad, 2) # Only using the past gradient as the 2nd momentum 
                # sum_mom_step = beta_2m * sum_mom_step + (1 - beta_2m) * torch.pow(data_step_grad, 2) # Normal way, but perform bad
            elif(Used_attack_method_idx == 3):
                # sum_mom_step = sum_mom_step + torch.pow(data_step_grad, 2) # normal way, but performs bad
                # sum_mom_step = sum_mom_step / step
                sum_mom_step = torch.pow(sum_mom_step, 1) + torch.pow(data_step_grad, 2) # notice that, we update this by "data_step_grad(current gradient)", which is very very different attack1, i.e."temp_prev_grad_step(current 1st momentum)". 
            elif(Used_attack_method_idx == 4 or Used_attack_method_idx == 5 or Used_attack_method_idx == 6 or Used_attack_method_idx == 10 or Used_attack_method_idx == 11): # MI-FGSM, I-FGSM and I-FGV methods do NOT use the 2nd moomentum
                pass
            elif(Used_attack_method_idx == 14 or Used_attack_method_idx == 12 or Used_attack_method_idx == 16):
                beta_2m = 0.99
                # sum_mom_step = beta_2m * torch.pow(sum_mom_step, 1) + (1 - beta_2m) * torch.pow(temp_prev_grad_step, 2)
                sum_mom_step = beta_2m * torch.pow(sum_mom_step, 1) + (1 - beta_2m) * torch.pow(data_step_grad, 2)
                

            # perturbed_data_step, temp_prev_grad_step, first_mom = attack_perform.saliency_attack(data_step, epsilon_step, data_step_grad, temp_prev_grad_step, sum_mom_step)
            perturbed_data_step, temp_prev_grad_step = attack_perform.saliency_attack(data_step, epsilon_step, data_step_grad, temp_prev_grad_step, sum_mom_step)
            # print("sum_mom_step and temp_prev_grad_step are:", sum_mom_step, temp_prev_grad_step, sum_mom_step.size())

            # output_per = model(perturbed_data_step)
            output_per_1, d6_per_1, d5_per_1 = model_1(perturbed_data_step)
            output_per_2, d6_per_2, d5_per_2 = model_2(perturbed_data_step)

            shan = tensor2saveimg(perturbed_data_step, str(index_input) + "_" + str(epsilon) + "_perturbed_data_step")
            
            svname = str(index_input) + "_" + str(epsilon) + "_perturbed_data_step"
            # path_per = "D:\\Study\\code\\Pytorch-AdvExm\\CIFAR10\\visresults\\" + svname + ".png"
            path_per = root_sv_path + svname + ".png"
            # print("path is :", path_per)
            data_step = Image.open(path_per).convert('RGB') 
            data_step = data_step.resize((Unit_Width, Unit_Height),Image.ANTIALIAS)
            data_step = np.array(data_step)
            data_step = data_step / 255.0
            data_step = torch.from_numpy(data_step)
            data_step = data_step.numpy()
            data_step = np.transpose(data_step, (2,0,1))
            data_step = np.expand_dims(data_step, axis=0)
            data_step = torch.from_numpy(data_step)
            data_step = data_step.float()
            '''
            if( ((step % 10) == 0) or (step < 10)):
                print("step is : ", step)
                shan = tensor2saveimg(perturbed_data_step, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_perturbed_data")
                shan = tensor2saveimg(output_per_1, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_peroutput_1")
                shan = tensor2saveimg(output_per_2, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_peroutput_2")
            '''
            if(step < 51):
                print("step is : ", step)
                shan = tensor2saveimg(perturbed_data_step, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_perturbed_data")
                shan = tensor2saveimg(output_per_1, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_peroutput_1")
                shan = tensor2saveimg(output_per_2, str(index_input) + "_" + str(epsilon) + "_" + str(step) + "_peroutput_2")
    file.close()
    # Return the attack performance and an adversarial example
    return adv_examples


######################################################################
# Run Attack

examples = []

# Run test for each epsilon
for eps in epsilons:
    # ex = test(model, device, test_loader, eps)
    # ex = test(model_1, device, eps) 
    ex = test(model_1, model_2, device, eps) 
    # accuracies.append(acc)
    examples.append(ex)

