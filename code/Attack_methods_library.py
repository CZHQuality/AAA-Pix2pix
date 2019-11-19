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

# saliency_attack_1: Improved Adam plus a Random step, plus explicit Optimized perceptual constraint loss: L1, L2 or SSIM

# non_target_attack = False
# target_attack = False
# universal_attack = True

#######################################################################################
# using both 1st and 2nd-plus momentum, and Random perturbation, and Optimized Perceptual Constraint
# under clip constraint
# This method, we use square sum and sqrt to calculate the 2_nd momentum
class attack_method_1(object):

    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # grad_mean = torch.sum(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 

        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = miu_factor*pre_mom + data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # print("next_grad is :", next_grad)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # grad_mean = torch.sum(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 

        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = miu_factor*pre_mom + data_grad
        # next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("unit_vector_data_grad is :", unit_vector_data_grad)

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom) # this operation will result in NaN: https://blog.csdn.net/ONE_SIX_MIX/article/details/90322472
        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = sec_mom * 1e5
        # sec_mom = torch.abs(sum_mom)
        # sec_mom = sum_mom
        # print("sec_mom sqrt is:", sec_mom)
        
        # gamma_random_pub = -1*(1e-4)
        # gamma_random_pub = -1*(1e-2)
        # gamma_random_pub = 1*(1e-4)
        gamma_random_pub = 1*(1e-2)
        # Random_add_peb = torch.rand(next_grad.size()).to('cuda')
        Random_add_peb = torch.randn(next_grad.size()).to('cuda')
        Random_add_peb = Random_add_peb * gamma_random_pub
        # print("next_grad, Random_add_peb is:", next_grad, Random_add_peb, Random_add_peb.size())
        
        next_grad = next_grad + Random_add_peb

        First_mom = next_grad

        next_grad = next_grad / (sec_mom + 1e-8)
        # print("sum_mom is :", sum_mom)
        # print("sec_mom is :", sec_mom)
        
        # next_grad = next_grad / (sec_mom + 1e-2) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # next_grad = next_grad / (sec_mom + 1e-8) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # print("raw data_grad is :", data_grad)
        # print("unit_vector_data_grad is :", unit_vector_data_grad)
        # print("next_grad is :", next_grad)
        # print("sec_mom is :", sec_mom)
        return perturbed_image, next_grad
        # return perturbed_image, First_mom

#######################################################################################
# Adam-I-FGSM attack, without using exponentially decaying for 1st momentum
# using both 1st and 2nd momentum. Only pay attention to the current gradient to update the 2nd momentum.
# Notice that the 2nd momentum of
# this method uses "gradient of previous time step" as "increment", which grows too slow, and 
# results in heavy pepper/salt noise
class attack_method_2(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    # def saliency_attack_2(image, epsilon, data_grad, pre_mom): 
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()
        First_mom = next_grad
        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)

        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)
        sec_mom = sec_mom * 1e5

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)
        # epsilon = epsilon / (sec_mom + 1e-8)
        
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, First_mom

#######################################################################################
# Normal Adam-I-FGSM attack version, using decaying for both 1st and 2nd momentum
# using both 1st and 2nd momentum. Using the Square Summation of all past gradients as the 2nd momentum
# The difference between this method and the second method is how to update the 2nd momentum.
# Notice that the 2nd momentum of
# this method uses "gradient of previous time step" as "increment", which grows too slow, and 
# results in heavy pepper/salt noise
class attack_method_3(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        # miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        miu_factor = 0.9
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + (1-miu_factor) * data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    # saliency_attack
    # def saliency_attack_2(image, epsilon, data_grad, pre_mom): 
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        # miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        miu_factor = 0.9
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        # next_grad = miu_factor*pre_mom + unit_vector_data_grad
        next_grad = miu_factor*pre_mom + (1-miu_factor)*unit_vector_data_grad
        # next_grad = next_grad.sign()
        First_mom = next_grad

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)

        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)
        sec_mom = sec_mom * 1e5

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # return perturbed_image, next_grad
        return perturbed_image, First_mom


#######################################################################################
# MIM attack: Normal MI-FGSM attack, which uses only 1st momentum
class attack_method_4(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # this is based on the MI-FGSM pipeline, with momentum
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        next_grad = next_grad.sign()
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom):
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        next_grad = next_grad.sign()
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

#######################################################################################
# Normal I-FGSM attack, without any momentum, which will result in severe overfitting on
# threat model, thus the transferability is poor.
class attack_method_5(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign() # use sign function to process
        # sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad
    
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

#######################################################################################
# Normal I-FGV attack, without any momentum, which will result in severe overfitting on
# threat model, thus the transferability is poor. The difference between this method and 
# the 5th method is "sign" operation. Also results in overfitting on the threat model, and 
# obtains bad transferability
class attack_method_6(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad
    
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

#######################################################################################
# AdaGrad/RMSprop-attack-1, which only uses 2nd-plus momentum without 1st momentum
class attack_method_7(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom)
        sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)

        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

#######################################################################################
# AdaGrad/RMSprop-attack-2, which only uses 2nd momentum without 1st momentum
class attack_method_8(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        First_mom = next_grad

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom)
        sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)

        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # return perturbed_image, next_grad
        return perturbed_image, First_mom

#######################################################################################
# using both 1st and 2nd-plus momentum, and Optimized Perceptual Constraint,
# but do not use the Random perturbation item
class attack_method_9(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    # saliency_attack
    # def saliency_attack_2(image, epsilon, data_grad, pre_mom): # this is based on the MI-FGSM pipeline, with momentum
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom)
        sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad


#######################################################################################
# Hot/Cold method, using SSIM as perceptual constraint, in optimization process, closing
# the gap between prediction and Hot label (target attention map), at the same time, reducing the
# gap between prediction and Cold label (original attention map).
#  without any momentum, which will result in severe overfitting on
# threat model, thus the transferability is poor. The difference between this method and 
# the 5th method is "sign" operation. Also results in overfitting on the threat model, and 
# obtains bad transferability
class attack_method_10(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad
    
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

#######################################################################################
# PGD attack, which starts from a lot of random points of L_{infinity} ball of the original data
# it update perturbation by Normal I-FGSM attack, without any momentum, which is dubbed as the
# strongest "first-order" adversary
class attack_method_11(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign() # use sign function to process
        # sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image

        gamma_random_pub = 1*(1e-2)
        Random_add_peb = torch.randn(sign_data_grad.size()).to('cuda')
        Random_add_peb = Random_add_peb * gamma_random_pub
        sign_data_grad = sign_data_grad + Random_add_peb

        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad
    
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): # this is based on the iterative-FGSM pipeline
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        sign_data_grad = data_grad # use original function
        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*sign_data_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*sign_data_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

#######################################################################################
# C&W attack: C&W attack is dedicated to attack image classifition models. We extend this method to attention/saliency
# task, explained as follows:  We first adopt the "min-max" to normalize the predicted and targeted saliency map
# into range [0, 1], then we divide each element of the saliency map to the summation of all elements of the saliency
# map. This way, each saliency map (or feature map) could be ragarded as a "distribution". Then, we extend the attack
# performance loss proposed by C&W as: Loss_{attack} = max{ KLd[ SM_{pre}, SM_{tar} ], 0 }. This way, the loss_{attack} >= 0
# only when the distribution of SM_{pre} is the same as SM_{tar}.
# We follow the original paper, and set the gradient descend method for C&W as:
# Normal Adam-I-FGSM attack, using both 1st and 2nd momentum. Only pay attention to the current gradient to update the 2nd momentum.
# Notice that the 2nd momentum of
# this method uses "gradient of previous time step" as "increment", which grows too slow, and 
# results in heavy pepper/salt noise
# We ad the binary search step to find the optimal weighting hyperparameter
# "B_CONST" to balance the attack performance and the perceptual constraint loss. We follow the C&W attack,
# and set initial "B_CONST" as 1e-3, then we use a binary search to find the optimal "B_CONST" for 9 iterations
class attack_method_12(object):
    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + (1-miu_factor)*unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1) # this is valid and successes
        # perturbed_image = (torch.tanh(perturbed_image) + 1.0) * 0.5 # C&W improve the box-constraint as a natively constraint, 
                                                                    # this way adv. exp is still in range [0, 1]. But we found that this method fails to generate
                                                                    #  valid adv.exp due to OVER EXPOSURE
        # NOR_OPER = nn.Softmax() # fail
        # NOR_OPER = nn.Tanh() # fail
        # perturbed_image = NOR_OPER(perturbed_image)
        # min1 = torch.min(perturbed_image)
        # max1 = torch.max(perturbed_image)
        # output_1 = (perturbed_image - min1) / (max1 - min1 + 1e-8)
        # Return the perturbed image
        return perturbed_image, next_grad

    # def saliency_attack_2(image, epsilon, data_grad, pre_mom): 
    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8 
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + (1-miu_factor)*unit_vector_data_grad
        # next_grad = next_grad.sign()
        First_mom = next_grad

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)

        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)
        # sec_mom = sec_mom * 1e5

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-8)
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1) # this is valid and successes
        # perturbed_image = (torch.tanh(perturbed_image) + 1.0) * 0.5 # C&W improve the box-constraint as a natively constraint, this way adv. exp is still in range [0, 1]
        # NOR_OPER = nn.Softmax() # fail
        # NOR_OPER = nn.Tanh() # fail
        # perturbed_image = (NOR_OPER(perturbed_image) + 1.0) * 0.5
        # min1 = torch.min(perturbed_image)
        # max1 = torch.max(perturbed_image)
        # output_1 = (perturbed_image - min1) / (max1 - min1 + 1e-8)
        # print("perturbed_image is:", perturbed_image)
        # Return the perturbed image
        # return perturbed_image, next_grad
        return perturbed_image, First_mom

#######################################################################################
# Improved C&W attack using our gradient descend method: 
# C&W attack is dedicated to attack image classifition models. We extend this method to attention/saliency
# task, explained as follows:  We first adopt the "min-max" to normalize the predicted and targeted saliency map
# into range [0, 1], then we divide each element of the saliency map to the summation of all elements of the saliency
# map. This way, each saliency map (or feature map) could be ragarded as a "distribution". Then, we extend the attack
# performance loss proposed by C&W as: Loss_{attack} = max{ KLd[ SM_{pre}, SM_{tar} ], 0 }. This way, the loss_{attack} >= 0
# only when the distribution of SM_{pre} is the same as SM_{tar}.
# We improve from the original paper, and set the gradient descend method for C&W as our new gradient descend method using
# 1st momentum and 2nd-plus momentum and Random perturbation, and Optimized Perceptual Constraint
class attack_method_13(object):

    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # min1 = torch.min(perturbed_image)
        # max1 = torch.max(perturbed_image)
        # output_1 = (perturbed_image - min1) / (max1 - min1 + 1e-8)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom)
        sec_mom = torch.abs(sum_mom)
        # print("sec_mom sqrt is:", sec_mom)
        
        # gamma_random_pub = -1*(1e-4)
        # gamma_random_pub = -1*(1e-2)
        # gamma_random_pub = 1*(1e-4)
        gamma_random_pub = 1*(1e-2)
        # Random_add_peb = torch.rand(next_grad.size()).to('cuda')
        Random_add_peb = torch.randn(next_grad.size()).to('cuda')
        Random_add_peb = Random_add_peb * gamma_random_pub
        # print("next_grad, Random_add_peb is:", next_grad, Random_add_peb, Random_add_peb.size())
        
        next_grad = next_grad + Random_add_peb

        # next_grad = next_grad / (sec_mom + 1e-8)
        next_grad = next_grad / (sec_mom + 1e-2)
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # min1 = torch.min(perturbed_image)
        # max1 = torch.max(perturbed_image)
        # output_1 = (perturbed_image - min1) / (max1 - min1 + 1e-8)
        # Return the perturbed image
        return perturbed_image, next_grad

# Our proposed intra-batch attack
# option: use normalized gradient: "unit_vector_data_grad" (lower perceptibility, and is robust to gradient scale change)
# or raw gradient : data_grad (higher preceptibility)
class attack_method_14(object):

    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        # grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        grad_mean = torch.sum(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        # next_grad = miu_factor*pre_mom + (1-miu_factor) * data_grad
        next_grad = miu_factor*pre_mom + (1-miu_factor) * unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        # grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        grad_mean = torch.sum(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        # next_grad = miu_factor*pre_mom + (1-miu_factor) * data_grad
        next_grad = miu_factor*pre_mom + (1-miu_factor) * unit_vector_data_grad
        # next_grad = next_grad.sign()

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = sec_mom * 1e5
        # sec_mom = torch.abs(sum_mom)
        # sec_mom = sum_mom
        # print("sec_mom sqrt is:", sec_mom)
        
        # gamma_random_pub = -1*(1e-4)
        # gamma_random_pub = -1*(1e-2)
        # gamma_random_pub = 1*(1e-4)
        # gamma_random_pub = 1*(1e-2)
        # Random_add_peb = torch.rand(next_grad.size()).to('cuda')
        # Random_add_peb = torch.randn(next_grad.size()).to('cuda')
        # Random_add_peb = Random_add_peb * gamma_random_pub
        # print("next_grad, Random_add_peb is:", next_grad, Random_add_peb, Random_add_peb.size())
        
        # next_grad = next_grad + Random_add_peb

        First_mom = next_grad

        next_grad = next_grad / (sec_mom + 1e-8)
        # print("sec_mom is :", sec_mom)
        # next_grad = next_grad / (sec_mom + 1e-2) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # next_grad = next_grad / (sec_mom + 1e-8) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        # epsilon = epsilon * 1e-5
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # return perturbed_image, next_grad
        # print("raw data_grad is :", data_grad)
        # print("unit_vector_data_grad is :", unit_vector_data_grad)
        # print("next_grad is :", next_grad)
        # print("sec_mom is :", sec_mom)
        return perturbed_image, First_mom
        # return perturbed_image, next_grad, First_mom

# attack_method_1, use decay for the 1_st momentum by setting miu_factor = 0.9
class attack_method_15(object):

    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        # miu_factor = 1.0
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        next_grad = miu_factor*pre_mom + (1-miu_factor)*unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # print("next_grad is :", next_grad)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        # miu_factor = 1.0 # the weights to control how many momentum we should take from the previous momentum
        miu_factor = 0.9
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        # next_grad = miu_factor*pre_mom + unit_vector_data_grad
        # next_grad = next_grad.sign()

        # print("unit_vector_data_grad is :", unit_vector_data_grad)

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        # sec_mom = torch.sqrt(sum_mom) # this operation will result in NaN: https://blog.csdn.net/ONE_SIX_MIX/article/details/90322472
        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = sec_mom * 1e5
        # sec_mom = torch.abs(sum_mom)
        # sec_mom = sum_mom
        # print("sec_mom sqrt is:", sec_mom)
        
        # gamma_random_pub = -1*(1e-4)
        # gamma_random_pub = -1*(1e-2)
        # gamma_random_pub = 1*(1e-4)
        gamma_random_pub = 1*(1e-2)
        # Random_add_peb = torch.rand(next_grad.size()).to('cuda')
        Random_add_peb = torch.randn(unit_vector_data_grad.size()).to('cuda')
        Random_add_peb = Random_add_peb * gamma_random_pub
        # print("next_grad, Random_add_peb is:", next_grad, Random_add_peb, Random_add_peb.size())
        
        next_grad = miu_factor*pre_mom + (1-miu_factor-gamma_random_pub)*unit_vector_data_grad + Random_add_peb
        # next_grad = next_grad + Random_add_peb

        First_mom = next_grad

        # next_grad = next_grad / (sec_mom + 1e-8)
        # print("sum_mom is :", sum_mom)
        # print("sec_mom is :", sec_mom)
        
        next_grad = next_grad / (sec_mom + 1e-2) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # next_grad = next_grad / (sec_mom + 1e-8) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # print("next_grad is :", next_grad)
        return perturbed_image, next_grad
        # return perturbed_image, First_mom


# A failure Adam attack, which produce very sparse but very perceptual noticeable perturbations
# The key point is: using raw "data_grad" or normalized "unit_vector_data_grad"
# in the first iteration, if use "data_grad", and in other iterations use "unit_vector_data_grad" will
# result in very sparse perturbation
class attack_method_16(object):

    def saliency_attack_start(self, image, epsilon, data_grad, pre_mom): 
        # Collect the element-wise sign of the data gradient
        # sign_data_grad = data_grad.sign() # use sign function to process
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 

        next_grad = miu_factor*pre_mom + (1-miu_factor) * unit_vector_data_grad
        # next_grad = miu_factor*pre_mom + (1-miu_factor) * data_grad
        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image, next_grad

    def saliency_attack(self, image, epsilon, data_grad, pre_mom, sum_mom): 
        miu_factor = 0.9 # the weights to control how many momentum we should take from the previous momentum
        unit_vector_data_grad = torch.abs(data_grad.view(1, -1))  
        grad_mean = torch.mean(unit_vector_data_grad) + 1e-8
        # print("grad_mean data_grad unit_vector_data_grad is:", grad_mean, data_grad.size(), unit_vector_data_grad)
        unit_vector_data_grad = data_grad / grad_mean 
        # next_grad = miu_factor*pre_mom + (1-miu_factor) * data_grad
        next_grad = miu_factor*pre_mom + (1-miu_factor) * unit_vector_data_grad
        # next_grad = next_grad.sign()

        # sec_mom = torch.abs(sum_mom.view(1, -1))
        # sec_mom = torch.mean(sec_mom) 
        # sec_mom = torch.sqrt(sec_mom)
        sec_mom = torch.sqrt(sum_mom + 1e-8)
        # sec_mom = sec_mom * 1e5
        # sec_mom = torch.abs(sum_mom)
        # sec_mom = sum_mom
        # print("sec_mom sqrt is:", sec_mom)
        
        # gamma_random_pub = -1*(1e-4)
        # gamma_random_pub = -1*(1e-2)
        # gamma_random_pub = 1*(1e-4)
        # gamma_random_pub = 1*(1e-2)
        # Random_add_peb = torch.rand(next_grad.size()).to('cuda')
        # Random_add_peb = torch.randn(next_grad.size()).to('cuda')
        # Random_add_peb = Random_add_peb * gamma_random_pub
        # print("next_grad, Random_add_peb is:", next_grad, Random_add_peb, Random_add_peb.size())
        
        # next_grad = next_grad + Random_add_peb

        First_mom = next_grad

        # next_grad = next_grad / (sec_mom + 1e-8)
        # print("sec_mom is :", sec_mom)
        next_grad = next_grad / (sec_mom + 1e-2) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # next_grad = next_grad / (sec_mom + 1e-8) # 1e-2 is a smoothing term to avoid division by 0. 
                                                #Also 1e-2 is in the same scale with initial sec_mom. sec_mom convergences to 1e+00 or 1e+01 scale
        # epsilon = epsilon / (sec_mom + 1e-8)

        # next_grad = next_grad.sign()

        # print("sign_data_grad are :", sign_data_grad)
        # sign_data_grad contains 3 values : -1, 0 and 1
        # Create the perturbed image by adjusting each pixel of the input image
        # epsilon = epsilon * 1e-5
        if(non_target_attack):
            perturbed_image = image + epsilon*next_grad   # for 
        elif(target_attack):
            perturbed_image = image - epsilon*next_grad 
        elif(universal_attack):
            perturbed_image = image - epsilon*next_grad 
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        # return perturbed_image, next_grad
        # print("raw data_grad is :", data_grad)
        # print("unit_vector_data_grad is :", unit_vector_data_grad)
        # print("next_grad is :", next_grad)
        # print("sec_mom is :", sec_mom)
        return perturbed_image, First_mom
