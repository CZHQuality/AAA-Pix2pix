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

from DCN_lib.modules.modulated_dcn import ModulatedDeformConvPack
from DCN_lib.modules.modulated_dcn import DeformRoIPooling 
from DCN_lib.modules.modulated_dcn import ModulatedDeformRoIPoolingPack
from DCN_lib.modules.modulated_dcn import ModulatedDeformConv

class GazeGAN_1(nn.Module): # based on U-Net, change the VGG of SalGAN as U-Net
    # def __init__(self, input_nc, output_nc, ngf=64):
    def __init__(self, input_nc = 3, output_nc = 3, ngf=64):
        super(GazeGAN_1, self).__init__()   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock

        # encoder of generator
        self.en_conv1 = encoderconv_2(input_nc, 64)
        self.en_conv2 = encoderconv_2(64, 128)
        self.en_conv3 = encoderconv_2(128, 256)
        self.en_conv4 = encoderconv_2(256, 512)
        self.en_conv5 = encoderconv_2(512, 1024)
        self.en_conv6 = encoderconv_2(1024, 1024)
        
        self.res_1 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        self.de_conv1 = decoderconv_3(1024, 1024)
        self.de_conv2 = decoderconv_2(1024+1024, 512)
        self.de_conv3 = decoderconv_2(512+512, 256)
        self.de_conv4 = decoderconv_2(256+256, 128)
        self.de_conv5 = decoderconv_2(128+128, 64)
        self.de_conv6 = decoderconv_2(64+64, output_nc)
        # bottle-neck layer
        self.dimr_conv1 = dimredconv(output_nc, output_nc)

    def forward(self, input):
        e1 = self.en_conv1(input)
        e2 = self.en_conv2(e1)
        e3 = self.en_conv3(e2)
        e4 = self.en_conv4(e3)
        e5 = self.en_conv5(e4)
        e6 = self.en_conv6(e5)

        res1 = self.res_1(e6)
        # print("res1:", res1)
        res2 = self.res_2(res1)
        res3 = self.res_2(res2) # to do: fix res_2 as res_3
        res4 = self.res_2(res3) # to do: fix res_2 as res_4

        d1 = self.de_conv1(res4)
        d2 = self.de_conv2(torch.cat([d1, e5], dim=1))
        d3 = self.de_conv3(torch.cat([d2, e4], dim=1))
        d4 = self.de_conv4(torch.cat([d3, e3], dim=1))
        d5 = self.de_conv5(torch.cat([d4, e2], dim=1))
        d6 = self.de_conv6(torch.cat([d5, e1], dim=1))    
        d7 = self.dimr_conv1(d6)
        
        out = d7 # the real final output
        # out = torch.squeeze(e1, 0)

        out1 = res4
        # print("e1 size is :", e1.size())
        # out1 = out1[:, 0:3, :, :] # this is right
        out1 = torch.mean(out1, 1) # mean across channel direction
        bat, hei, wei = out1.size()
        # print("bat hei wei", bat, hei, wei)
        out2 = torch.zeros(3, hei, wei)
        out2[0, :, :] = out1
        out2[1, :, :] = out1
        out2[2, :, :] = out1
        out2 = out2.unsqueeze(0)
        # print("out2 of model1:", out2)
        out2 = torch.nn.functional.upsample_bilinear(out2, size=[120, 160])
        
        # print("out2 size is :", out2.size())
        # return out, d4, e6
        return out, out2, e6
        # return out, out2, e6

class SALICON_2(nn.Module): # SALICON_2 model, which uses single VGG net as feature extractor
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(SALICON_2, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(128, affine=False)
        
        self.conv7_1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.norm7 = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1 = nn.Tanh()

        self.upsp = nn.Upsample(scale_factor=8, mode='bilinear')
        
        # self.upsp = nn.Upsample(scale_factor=4, mode='bilinear')
        
    def forward(self, input):
        e1 = self.conv1_1(input)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())
        
        res1 = self.res_1(e5)
        res2 = self.res_2(res1)
        
        d1 = self.conv6_1(e5)
        d1 = self.norm6(d1)
        d1 = self.relu6_1(d1)
        # print("d1 size is :", d1.size())
        
        d2 = self.conv7_1(d1)
        # d2 = self.norm7(d2)
        d2 = self.relu7_1(d2)
        # print("d2 size is :", d2.size())

        d3 = self.conv9_1(d2)
        d3 = self.tanh9_1(d3)

        out1 = d1
        out1 = torch.mean(out1, 1) # mean across channel direction
        bat, hei, wei = out1.size()
        # print("second model: bat hei wei", bat, hei, wei)
        out2 = torch.zeros(1, 3, hei, wei)
        out2[0, 0, :, :] = out1
        out2[0, 1, :, :] = out1
        out2[0, 2, :, :] = out1
        
        d3_upsp = self.upsp(d3) # resize from 60*80 to 480*640, for summarion with other model predictions
        out2_upsp = self.upsp(out2)
        d1_upsp = self.upsp(d1)

        out1 = e4
        out1 = torch.mean(out1, 1) # mean across channel direction
        bat, hei, wei = out1.size()
        # print("bat hei wei", bat, hei, wei)
        out2 = torch.zeros(3, hei, wei)
        out2[0, :, :] = out1
        out2[1, :, :] = out1
        out2[2, :, :] = out1
        out2 = out2.unsqueeze(0)
        out2 = torch.nn.functional.upsample_bilinear(out2, size=[120, 160])
        
        # return d3, d2, d1
        return d3_upsp, out2, d1_upsp

class Localpix2pix(nn.Module): # Local pix2pix model
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=4, n_blocks_global=4, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(Localpix2pix, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock 
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = Globalpix2pix(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1]) # Here, the 1st output_prev is the output feature map of the global generator   
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1): # This code is motivated to design a multi-local-generator architecture, 
                                                                     # although the original paper only use one local generator
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1') 
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev) # the encoder part of local generator, concat with the final output of global generator, 
                                                                                  # then get into the decoder of the local generator, Notice that it's a direct "add" operation, not concat
        return output_prev

class Globalpix2pix(nn.Module): # Globalpix2pix model
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Globalpix2pix, self).__init__()        
        # activation = nn.ReLU(True)   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock     

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        # return self.model(input)  
        out1 = torch.zeros(1, 3, 60, 80)
        out2 = torch.zeros(1, 3, 60, 80)
        out3 = self.model(input)

        out1 = out3
        # print("e1 size is :", e1.size())
        # out1 = out1[:, 0:3, :, :] # this is right
        out1 = torch.mean(out1, 1) # mean across channel direction
        bat, hei, wei = out1.size()
        # print("bat hei wei", bat, hei, wei)
        out2 = torch.zeros(3, hei, wei)
        out2[0, :, :] = out1
        out2[1, :, :] = out1
        out2[2, :, :] = out1
        out2 = out2.unsqueeze(0)
        out2 = torch.nn.functional.upsample_bilinear(out2, size=[120, 160])
    
        return out3, out2, out1

class GazeGAN_2(nn.Module): # GazeGAN which adopts local and global U-Nets as generator
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(GazeGAN_2, self).__init__()   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock

        # encoder of generator
        self.en_conv1 = encoderconv_2(input_nc, 64)
        self.en_conv2 = encoderconv_2(64, 128)
        # self.en_conv3 = encoderconv_2(128, 256) # local only
        self.en_conv3 = encoderconv_2(128+128, 256) # global + local
        self.en_conv4 = encoderconv_2(256, 512)
        self.en_conv5 = encoderconv_2(512, 1024)
        self.en_conv6 = encoderconv_2(1024, 1024)
        # self.en_conv7 = encoderconv_2(1024, 1024)
        # self.en_conv8 = encoderconv_2(1024, 1024)
        
        self.res_1 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        
        self.de_conv1 = decoderconv_3(1024, 1024)
        self.de_conv2 = decoderconv_2(1024+1024, 512)
        self.de_conv3 = decoderconv_2(512+512, 256)
        self.de_conv4 = decoderconv_2(256+256, 128)
        # self.de_conv5 = decoderconv_2(128+128, 64) # local only
        self.de_conv5 = decoderconv_2(128+256, 64) # global + local
        self.de_conv6 = decoderconv_2(64+64, output_nc)

        # bottle-neck layer
        self.dimr_conv1 = dimredconv(output_nc, output_nc)
        # 2X downsampling model for input image
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.en_conv1_g = encoderconv_2(input_nc, 128)
        self.en_conv2_g = encoderconv_2(128, 256)
        self.en_conv3_g = encoderconv_2(256, 512)
        self.en_conv4_g = encoderconv_2(512, 1024)
        self.en_conv5_g = encoderconv_2(1024, 1024)
        self.res_g = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.de_conv1_g = decoderconv_3(1024, 1024)
        self.de_conv2_g = decoderconv_2(1024+1024, 512)
        self.de_conv3_g = decoderconv_2(512+512, 256)
        self.de_conv4_g = decoderconv_2(256+256, 128)
        self.de_conv5_g = decoderconv_2(128+128, output_nc)

    def forward(self, input):
        input_ds = self.downsample(input)


        # global U-Net
        e1_g = self.en_conv1_g(input_ds)
        e2_g = self.en_conv2_g(e1_g)
        e3_g = self.en_conv3_g(e2_g)
        e4_g = self.en_conv4_g(e3_g)
        e5_g = self.en_conv5_g(e4_g)

        res1_g = self.res_g(e5_g)
        res2_g = self.res_g(res1_g)
        res3_g = self.res_g(res2_g)
        res4_g = self.res_g(res3_g)
        # print("res1_g :", res1_g.size())

        d1_g = self.de_conv1_g(res4_g)
        # print("d1_g and e4_g are :", d1_g.size(), e4_g.size())
        d2_g = self.de_conv2_g(torch.cat([d1_g, e4_g], dim=1))
        # print("d2_g and e3_g are :", d2_g.size(), e3_g.size())
        d3_g = self.de_conv3_g(torch.cat([d2_g, e3_g], dim=1))
        # print("d3_g and e2_g are :", d3_g.size(), e2_g.size())
        d4_g = self.de_conv4_g(torch.cat([d3_g, e2_g], dim=1))
        # print("d4_g and e1_g are :", d4_g.size(), e1_g.size())
        d5_g = self.de_conv5_g(torch.cat([d4_g, e1_g], dim=1))
        # print("d5_g are :", d5_g.size()) 
        d6_g = self.dimr_conv1(d5_g) # d6_g is a small output saliency map

        # local U-Net
        e1 = self.en_conv1(input)
        # print("size of input is :", input.size())
        # print("size of e1 is :", e1.size())
        e2_local = self.en_conv2(e1)

        # e2 = torch.add(d4_g, e2_local) # pooling the global features into local UNet
        # e2 = d4_g + e2_local # pooling the global features into local UNet
        e2 = torch.cat([e2_local, d4_g], dim=1)

        # print("size of e2 is :", e2.size())
        e3 = self.en_conv3(e2)
        # print("size of e3 is :", e3.size())
        e4 = self.en_conv4(e3)
        # print("size of e4 is :", e4.size())
        e5 = self.en_conv5(e4)
        # print("size of e5 is :", e5.size())
        e6 = self.en_conv6(e5)
        # print("size of e6 is :", e6.size())

        res1 = self.res_1(e6)
        res2 = self.res_2(res1)
        res3 = self.res_2(res2)
        res4 = self.res_2(res3)

        # d1 = self.de_conv1(e6)
        d1 = self.de_conv1(res4)
        # print("d1 and e5 are :", d1.size(), e5.size())
        d2 = self.de_conv2(torch.cat([d1, e5], dim=1))
        # d2 = self.de_conv2(d1)
        # print("d2 and e4 are :", d2.size(), e4.size())
        d3 = self.de_conv3(torch.cat([d2, e4], dim=1))
        # d3 = self.de_conv3(d2)
        # print("d3 and e3 are :", d3.size(), e3.size())
        d4 = self.de_conv4(torch.cat([d3, e3], dim=1))
        # d4 = self.de_conv4(d3)
        #print("d4 and e2 are :", d4.size(), e2.size())
        d5 = self.de_conv5(torch.cat([d4, e2], dim=1))
        # d5 = self.de_conv5(d4)
        # print("d5 and e1 are :", d5.size(), e1.size())
        d6 = self.de_conv6(torch.cat([d5, e1], dim=1))
        
        # d6 = self.de_conv6(d5)
        
        d7 = self.dimr_conv1(d6)

        
        
        out = d7 # the real final output
        # out = torch.squeeze(e1, 0)

        # out = e1
        # out = out[0:2, :, :]
        # print("out is :", out, out.size())
        '''
        out1 = d5
        # out = out1[0:1, 0:3, :, :] # this is right
        out = torch.mean(out1, 1) # mean across 64 channel direction
        out = torch.unsqueeze(out, 0)
        print("out1 size is :", out1.size())
        print("out size is :", out.size())
        print("d7 size is :", d7.size())
        '''
        # return out, res4, d3

        
        return out, d3, d4


class DCN_LSTM_1(nn.Module): # Proposed saliency model equipped with 2D-modulated-deformable convolution, and ConvLSTM refinement module.
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(DCN_LSTM_1, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)
 
        self.conv1_1_h = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1_h = nn.ReLU(inplace=True)
        self.conv1_2_h = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1_h = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2_h = nn.ReLU(inplace=True)
        self.max1_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1_h = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1_h = nn.ReLU(inplace=True)
        self.conv2_2_h = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2_h = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2_h = nn.ReLU(inplace=True)
        self.max2_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1_h = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1_h = nn.ReLU(inplace=True)
        self.conv3_2_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2_h = nn.ReLU(inplace=True)
        self.conv3_3_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3_h = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3_h = nn.ReLU(inplace=True)
        self.max3_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1_h = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1_h = nn.ReLU(inplace=True)
        self.conv4_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2_h = nn.ReLU(inplace=True)
        self.conv4_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4_h = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3_h = nn.ReLU(inplace=True)
        self.max4_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1_h = nn.ReLU(inplace=True)
        self.conv5_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2_h = nn.ReLU(inplace=True)
        self.conv5_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5_h = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3_h = nn.ReLU(inplace=True)
        self.max5_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512 + 512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        # self.dcn_1 = ModulatedDeformConvPack(512, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        # self.dcn_2 = ModulatedDeformConvPack(512, 256, kernel_size=(5,5), stride=1, padding=2, deformable_groups=2, no_bias=True).cuda()
        
        self.onemulone_ird_1_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_1 = SELayer(384, reduction=8)
        self.onemulone_ird_1_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_1 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_1 = nn.ReLU(inplace=True)

        self.onemulone_ird_2_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_2 = SELayer(384, reduction=8)
        self.onemulone_ird_2_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_2 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_2 = nn.ReLU(inplace=True)

        self.onemulone_ird_3_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_3 = SELayer(384, reduction=8)
        self.onemulone_ird_3_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_3 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_3 = nn.ReLU(inplace=True)

        self.onemulone_ird_4_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_4 = SELayer(384, reduction=8)
        self.onemulone_ird_4_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_4 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_4 = nn.ReLU(inplace=True)

        self.onemulone_ird_5_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_5 = SELayer(384, reduction=8)
        self.onemulone_ird_5_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_5 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_5 = nn.ReLU(inplace=True)

        self.onemulone_ird_6_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_6 = SELayer(384, reduction=8)
        self.onemulone_ird_6_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_6 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_6 = nn.ReLU(inplace=True)

        self.onemulone_ird_7_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_7 = SELayer(384, reduction=8)
        self.onemulone_ird_7_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_7 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_7 = nn.ReLU(inplace=True)

        self.onemulone_ird_8_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_8 = SELayer(384, reduction=8)
        self.onemulone_ird_8_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_8 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_8 = nn.ReLU(inplace=True)

        self.onemulone_ird_9_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_9 = SELayer(384, reduction=8)
        self.onemulone_ird_9_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_9 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_9 = nn.ReLU(inplace=True)

        self.onemulone_ird_10_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_10 = SELayer(384, reduction=8)
        self.onemulone_ird_10_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_10 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_10 = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        # self.ConvLSTM_layer_1 = MyLSTM(input_size=(16,20), input_dim=1024, hidden_dim=[1024], 
                           # kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.ConvLSTM_layer_1 = MyLSTM(input_size=(64,80), input_dim=128, hidden_dim=[128], 
                            kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=1024, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')
            
        
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())

        res1 = self.res_1(e5)
        res2 = self.res_2(res1)
        res3 = self.res_1(res2)
        res4 = self.res_2(res3)

        e5_upsp = self.conv6_1(res4)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())
  
        e1_h = self.conv1_1_h(input)
        # e1_h = self.relu1_1_h(e1_h)
        # e1_h = self.conv1_2_h(e1_h)
        e1_h = self.norm1_h(e1_h)
        e1_h = self.relu1_2_h(e1_h)
        e1_h = self.max1_h(e1_h)
        # print("e1_h size is :", e1_h.size())
        
        e2_h = self.conv2_1_h(e1_h)
        # e2_h = self.relu2_1_h(e2_h)
        # e2_h = self.conv2_2_h(e2_h)
        e2_h = self.norm2_h(e2_h)
        e2_h = self.relu2_2_h(e2_h)
        e2_h = self.max2_h(e2_h)
        # print("e2_h size is :", e2_h.size())
        
        e3_h = self.conv3_1_h(e2_h)
        # e3_h = self.relu3_1_h(e3_h)
        # e3_h = self.conv3_2_h(e3_h)
        # e3_h = self.relu3_2_h(e3_h)
        # e3_h = self.conv3_3_h(e3_h)
        e3_h = self.norm3_h(e3_h)
        e3_h = self.relu3_3_h(e3_h)
        e3_h = self.max3_h(e3_h)
        # print("e3_h size is :", e3_h.size())

        ird_1_A = self.onemulone_ird_1_A(e3_h)
        ird_1_B = self.onemulone_ird_1_B(e3_h)
        ird_1_B = self.dcn_ird_1_B(ird_1_B)
        ird_1_C = self.onemulone_ird_1_C(e3_h)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_concat_ori = torch.cat([ird_1_A, ird_1_B, ird_1_C], dim=1)
        # print("size ofird_1_concat_ori is:", ird_1_concat_ori.size())
        ird_1_concat_se = self.se_1(ird_1_concat_ori)
        # print("size ofird_1_concat_se is:", ird_1_concat_se.size())
        ird_1_concat_residual = self.onemulone_ird_1_D(ird_1_concat_se)
        # ird_1_concat_residual = self.norm_ird_1(ird_1_concat_residual)
        # ird_1_concat_residual = self.relu_ird_1(ird_1_concat_residual)
        ird_1 = e3_h + ird_1_concat_residual
        # print("ird_1 size is:", ird_1.size())
        ird_1 = self.relu(ird_1)

        ird_2_A = self.onemulone_ird_2_A(ird_1)
        ird_2_B = self.onemulone_ird_2_B(ird_1)
        ird_2_B = self.dcn_ird_2_B(ird_2_B)
        ird_2_C = self.onemulone_ird_2_C(ird_1)
        ird_2_C = self.dcn_ird_2_C(ird_2_C) # two 3X3 equals one 5X5 deformable conv
        ird_2_C = self.dcn_ird_2_C(ird_2_C)
        ird_2_concat_ori = torch.cat([ird_2_A, ird_2_B, ird_2_C], dim=1)
        # print("size ofird_2_concat_ori is:", ird_2_concat_ori.size())
        ird_2_concat_se = self.se_2(ird_2_concat_ori)
        # print("size ofird_2_concat_se is:", ird_2_concat_se.size())
        ird_2_concat_residual = self.onemulone_ird_2_D(ird_2_concat_se)
        # ird_2_concat_residual = self.norm_ird_2(ird_2_concat_residual)
        # ird_2_concat_residual = self.relu_ird_2(ird_2_concat_residual)
        ird_2 = ird_1 + ird_2_concat_residual
        # print("ird_2 size is:", ird_2.size()) 
        ird_2 = self.relu(ird_2)      

        ird_3_A = self.onemulone_ird_3_A(ird_2)
        ird_3_B = self.onemulone_ird_3_B(ird_2)
        ird_3_B = self.dcn_ird_3_B(ird_3_B)
        ird_3_C = self.onemulone_ird_3_C(ird_2)
        ird_3_C = self.dcn_ird_3_C(ird_3_C) # two 3X3 equals one 5X5 deformable conv
        ird_3_C = self.dcn_ird_3_C(ird_3_C)
        ird_3_concat_ori = torch.cat([ird_3_A, ird_3_B, ird_3_C], dim=1)
        # print("size ofird_3_concat_ori is:", ird_3_concat_ori.size())
        ird_3_concat_se = self.se_3(ird_3_concat_ori)
        # print("size ofird_3_concat_se is:", ird_3_concat_se.size())
        ird_3_concat_residual = self.onemulone_ird_3_D(ird_3_concat_se)
        # ird_3_concat_residual = self.norm_ird_3(ird_3_concat_residual)
        # ird_3_concat_residual = self.relu_ird_3(ird_3_concat_residual)
        ird_3 = ird_2 + ird_3_concat_residual
        # print("ird_3 size is:", ird_3.size())
        ird_3 = self.relu(ird_3)

        ird_4_A = self.onemulone_ird_4_A(ird_3)
        ird_4_B = self.onemulone_ird_4_B(ird_3)
        ird_4_B = self.dcn_ird_4_B(ird_4_B)
        ird_4_C = self.onemulone_ird_4_C(ird_3)
        ird_4_C = self.dcn_ird_4_C(ird_4_C) # two 3X3 equals one 5X5 deformable conv
        ird_4_C = self.dcn_ird_4_C(ird_4_C)
        ird_4_concat_ori = torch.cat([ird_4_A, ird_4_B, ird_4_C], dim=1)
        # print("size ofird_4_concat_ori is:", ird_4_concat_ori.size())
        ird_4_concat_se = self.se_4(ird_4_concat_ori)
        # print("size ofird_4_concat_se is:", ird_4_concat_se.size())
        ird_4_concat_residual = self.onemulone_ird_4_D(ird_4_concat_se)
        # ird_4_concat_residual = self.norm_ird_4(ird_4_concat_residual)
        # ird_4_concat_residual = self.relu_ird_4(ird_4_concat_residual)
        ird_4 = ird_3 + ird_4_concat_residual
        # print("ird_4 size is:", ird_4.size())
        ird_4 = self.relu(ird_4)

        e4_h = self.conv4_1_h(ird_4)
        # e4_h = self.relu4_1_h(e4_h)
        # e4_h = self.conv4_2_h(e4_h)
        # e4_h = self.relu4_2_h(e4_h)
        # e4_h = self.conv4_3_h(e4_h)
        e4_h = self.norm4_h(e4_h)
        e4_h = self.relu4_3_h(e4_h)
        e4_h = self.max4_h(e4_h)
        # print("e4_h size is :", e4_h.size())

        ird_5_A = self.onemulone_ird_5_A(e4_h)
        ird_5_B = self.onemulone_ird_5_B(e4_h)
        ird_5_B = self.dcn_ird_5_B(ird_5_B)
        ird_5_C = self.onemulone_ird_5_C(e4_h)
        ird_5_C = self.dcn_ird_5_C(ird_5_C) # two 3X3 equals one 5X5 deformable conv
        ird_5_C = self.dcn_ird_5_C(ird_5_C)
        ird_5_concat_ori = torch.cat([ird_5_A, ird_5_B, ird_5_C], dim=1)
        # print("size ofird_5_concat_ori is:", ird_5_concat_ori.size())
        ird_5_concat_se = self.se_5(ird_5_concat_ori)
        # print("size ofird_5_concat_se is:", ird_5_concat_se.size())
        ird_5_concat_residual = self.onemulone_ird_5_D(ird_5_concat_se)
        # ird_5_concat_residual = self.norm_ird_5(ird_5_concat_residual)
        # ird_5_concat_residual = self.relu_ird_5(ird_5_concat_residual)
        ird_5 = e4_h + ird_5_concat_residual
        # print("ird_5 size is:", ird_5.size())
        ird_5 = self.relu(ird_5)

        ird_6_A = self.onemulone_ird_6_A(ird_5)
        ird_6_B = self.onemulone_ird_6_B(ird_5)
        ird_6_B = self.dcn_ird_6_B(ird_6_B)
        ird_6_C = self.onemulone_ird_6_C(ird_5)
        ird_6_C = self.dcn_ird_6_C(ird_6_C) # two 3X3 equals one 5X5 deformable conv
        ird_6_C = self.dcn_ird_6_C(ird_6_C)
        ird_6_concat_ori = torch.cat([ird_6_A, ird_6_B, ird_6_C], dim=1)
        # print("size ofird_6_concat_ori is:", ird_6_concat_ori.size())
        ird_6_concat_se = self.se_6(ird_6_concat_ori)
        # print("size ofird_6_concat_se is:", ird_6_concat_se.size())
        ird_6_concat_residual = self.onemulone_ird_6_D(ird_6_concat_se)
        # ird_6_concat_residual = self.norm_ird_6(ird_6_concat_residual)
        # ird_6_concat_residual = self.relu_ird_6(ird_6_concat_residual)
        ird_6 = ird_5 + ird_6_concat_residual
        # print("ird_6 size is:", ird_6.size())
        ird_6 = self.relu(ird_6)

        ird_7_A = self.onemulone_ird_7_A(ird_6)
        ird_7_B = self.onemulone_ird_7_B(ird_6)
        ird_7_B = self.dcn_ird_7_B(ird_7_B)
        ird_7_C = self.onemulone_ird_7_C(ird_6)
        ird_7_C = self.dcn_ird_7_C(ird_7_C) # two 3X3 equals one 5X5 deformable conv
        ird_7_C = self.dcn_ird_7_C(ird_7_C)
        ird_7_concat_ori = torch.cat([ird_7_A, ird_7_B, ird_7_C], dim=1)
        # print("size ofird_7_concat_ori is:", ird_7_concat_ori.size())
        ird_7_concat_se = self.se_7(ird_7_concat_ori)
        # print("size ofird_7_concat_se is:", ird_7_concat_se.size())
        ird_7_concat_residual = self.onemulone_ird_7_D(ird_7_concat_se)
        # ird_7_concat_residual = self.norm_ird_7(ird_7_concat_residual)
        # ird_7_concat_residual = self.relu_ird_7(ird_7_concat_residual)
        ird_7 = ird_6 + ird_7_concat_residual
        # print("ird_7 size is:", ird_7.size())
        ird_7 = self.relu(ird_7)

        ird_8_A = self.onemulone_ird_8_A(ird_7)
        ird_8_B = self.onemulone_ird_8_B(ird_7)
        ird_8_B = self.dcn_ird_8_B(ird_8_B)
        ird_8_C = self.onemulone_ird_8_C(ird_7)
        ird_8_C = self.dcn_ird_8_C(ird_8_C) # two 3X3 equals one 5X5 deformable conv
        ird_8_C = self.dcn_ird_8_C(ird_8_C)
        ird_8_concat_ori = torch.cat([ird_8_A, ird_8_B, ird_8_C], dim=1)
        # print("size ofird_8_concat_ori is:", ird_8_concat_ori.size())
        ird_8_concat_se = self.se_8(ird_8_concat_ori)
        # print("size ofird_8_concat_se is:", ird_8_concat_se.size())
        ird_8_concat_residual = self.onemulone_ird_8_D(ird_8_concat_se)
        # ird_8_concat_residual = self.norm_ird_8(ird_8_concat_residual)
        # ird_8_concat_residual = self.relu_ird_8(ird_8_concat_residual)
        ird_8 = ird_7 + ird_8_concat_residual
        # print("ird_8 size is:", ird_8.size())
        ird_8 = self.relu(ird_8)

        e5_h = self.conv5_1_h(ird_8)
        # e5_h = self.relu5_1_h(e5_h)
        # e5_h = self.conv5_2_h(e5_h)
        # e5_h = self.relu5_2_h(e5_h)
        # e5_h = self.conv5_3_h(e5_h)
        e5_h = self.norm5_h(e5_h)
        e5_h = self.relu5_3_h(e5_h)
        e5_h = self.max5_h(e5_h)
        # print("e5_h size is :", e5_h.size())

        e5_h_concat = torch.cat([e5_upsp, e5_h], dim=1)
        # print("shape of e5_h_concat:", e5_h_concat.size())

        # e5_sd = torch.cat([e5_s, e5_d], dim=1)
        e5_sd = self.channel_spatial_gate_1(e5_h_concat)
        # print("e5_sd size is :", e5_sd.size())

        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_sd)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())
        
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        # print("d2_h size is :", d2_h.size())

        d2_h_sequence_unit = torch.unsqueeze(d2_h, dim=1)
        # print("shape of d2_h_sequence_unit:", d2_h_sequence_unit.size())

        # e5_h_sequence_1 = []
        d2_h_sequence_2 = d2_h_sequence_unit

        for n_sequence in range(4):
            # e5_h_sequence_1.append(e5_h_sequence_unit)
            d2_h_sequence_2 = torch.cat([d2_h_sequence_2, d2_h_sequence_unit], dim=1)


        # print("shape of e5_h_sequence_1:", e5_h_sequence_1.size())
        # print("shape of d2_h_sequence_2:", d2_h_sequence_2.size())

        layer_output_list, last_state_list = self.ConvLSTM_layer_1(d2_h_sequence_2)
        # print("layer_output_list is:", layer_output_list)
        d2_h_refine = layer_output_list[0]
        d2_h_refine = d2_h_refine[:, -1, :, :, :]
        # print("shape of d2_h_refine:", d2_h_refine.size())


        
        # d3_h = self.conv8_1_h(d2_h)
        d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        
        d4_h_upsp = self.upsp_output(d4_out)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])

        return d4_h_upsp, d1_h, d3_h


class DCN_2(nn.Module): # Saliency model using 2D Spatial Deformable Convolution, SE-channel-attention (borrow from SE-Net), 
    # inception-like spatial attention (borrom from Tiantian Wang et.al, CVPR2018), without using ConvLSTM
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(DCN_2, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)
 
        self.conv1_1_h = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1_h = nn.ReLU(inplace=True)
        self.conv1_2_h = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1_h = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2_h = nn.ReLU(inplace=True)
        self.max1_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1_h = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1_h = nn.ReLU(inplace=True)
        self.conv2_2_h = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2_h = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2_h = nn.ReLU(inplace=True)
        self.max2_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1_h = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1_h = nn.ReLU(inplace=True)
        self.conv3_2_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2_h = nn.ReLU(inplace=True)
        self.conv3_3_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3_h = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3_h = nn.ReLU(inplace=True)
        self.max3_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1_h = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1_h = nn.ReLU(inplace=True)
        self.conv4_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2_h = nn.ReLU(inplace=True)
        self.conv4_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4_h = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3_h = nn.ReLU(inplace=True)
        self.max4_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1_h = nn.ReLU(inplace=True)
        self.conv5_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2_h = nn.ReLU(inplace=True)
        self.conv5_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5_h = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3_h = nn.ReLU(inplace=True)
        self.max5_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512 + 512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        self.onemulone_ird_1_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_1 = SELayer(384, reduction=8)
        self.onemulone_ird_1_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_1 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_1 = nn.ReLU(inplace=True)

        self.onemulone_ird_2_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_2 = SELayer(384, reduction=8)
        self.onemulone_ird_2_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_2 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_2 = nn.ReLU(inplace=True)

        self.onemulone_ird_3_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_3 = SELayer(384, reduction=8)
        self.onemulone_ird_3_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_3 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_3 = nn.ReLU(inplace=True)

        self.onemulone_ird_4_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_4 = SELayer(384, reduction=8)
        self.onemulone_ird_4_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_4 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_4 = nn.ReLU(inplace=True)

        self.onemulone_ird_5_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_5 = SELayer(384, reduction=8)
        self.onemulone_ird_5_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_5 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_5 = nn.ReLU(inplace=True)

        self.onemulone_ird_6_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_6 = SELayer(384, reduction=8)
        self.onemulone_ird_6_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_6 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_6 = nn.ReLU(inplace=True)

        self.onemulone_ird_7_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_7 = SELayer(384, reduction=8)
        self.onemulone_ird_7_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_7 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_7 = nn.ReLU(inplace=True)

        self.onemulone_ird_8_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_8 = SELayer(384, reduction=8)
        self.onemulone_ird_8_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_8 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_8 = nn.ReLU(inplace=True)

        self.onemulone_ird_9_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_9 = SELayer(384, reduction=8)
        self.onemulone_ird_9_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_9 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_9 = nn.ReLU(inplace=True)

        self.onemulone_ird_10_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_10 = SELayer(384, reduction=8)
        self.onemulone_ird_10_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_10 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_10 = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=1024, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')
            
        
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())

        res1 = self.res_1(e5)
        res2 = self.res_2(res1)
        res3 = self.res_1(res2)
        res4 = self.res_2(res3)

        e5_upsp = self.conv6_1(res4)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())
 
        e1_h = self.conv1_1_h(input)
        e1_h = self.norm1_h(e1_h)
        e1_h = self.relu1_2_h(e1_h)
        e1_h = self.max1_h(e1_h)
        # print("e1_h size is :", e1_h.size())
        
        e2_h = self.conv2_1_h(e1_h)
        e2_h = self.norm2_h(e2_h)
        e2_h = self.relu2_2_h(e2_h)
        e2_h = self.max2_h(e2_h)
        # print("e2_h size is :", e2_h.size())
        
        e3_h = self.conv3_1_h(e2_h)
        e3_h = self.norm3_h(e3_h)
        e3_h = self.relu3_3_h(e3_h)
        e3_h = self.max3_h(e3_h)
        # print("e3_h size is :", e3_h.size())

        ird_1_A = self.onemulone_ird_1_A(e3_h)
        ird_1_B = self.onemulone_ird_1_B(e3_h)
        ird_1_B = self.dcn_ird_1_B(ird_1_B)
        ird_1_C = self.onemulone_ird_1_C(e3_h)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_concat_ori = torch.cat([ird_1_A, ird_1_B, ird_1_C], dim=1)
        # print("size ofird_1_concat_ori is:", ird_1_concat_ori.size())
        ird_1_concat_se = self.se_1(ird_1_concat_ori)
        # print("size ofird_1_concat_se is:", ird_1_concat_se.size())
        ird_1_concat_residual = self.onemulone_ird_1_D(ird_1_concat_se)
        # ird_1_concat_residual = self.norm_ird_1(ird_1_concat_residual)
        # ird_1_concat_residual = self.relu_ird_1(ird_1_concat_residual)
        ird_1 = e3_h + ird_1_concat_residual
        # print("ird_1 size is:", ird_1.size())
        ird_1 = self.relu(ird_1)

        ird_2_A = self.onemulone_ird_2_A(ird_1)
        ird_2_B = self.onemulone_ird_2_B(ird_1)
        ird_2_B = self.dcn_ird_2_B(ird_2_B)
        ird_2_C = self.onemulone_ird_2_C(ird_1)
        ird_2_C = self.dcn_ird_2_C(ird_2_C) # two 3X3 equals one 5X5 deformable conv
        ird_2_C = self.dcn_ird_2_C(ird_2_C)
        ird_2_concat_ori = torch.cat([ird_2_A, ird_2_B, ird_2_C], dim=1)
        # print("size ofird_2_concat_ori is:", ird_2_concat_ori.size())
        ird_2_concat_se = self.se_2(ird_2_concat_ori)
        # print("size ofird_2_concat_se is:", ird_2_concat_se.size())
        ird_2_concat_residual = self.onemulone_ird_2_D(ird_2_concat_se)
        # ird_2_concat_residual = self.norm_ird_2(ird_2_concat_residual)
        # ird_2_concat_residual = self.relu_ird_2(ird_2_concat_residual)
        ird_2 = ird_1 + ird_2_concat_residual
        # print("ird_2 size is:", ird_2.size()) 
        ird_2 = self.relu(ird_2)      

        ird_3_A = self.onemulone_ird_3_A(ird_2)
        ird_3_B = self.onemulone_ird_3_B(ird_2)
        ird_3_B = self.dcn_ird_3_B(ird_3_B)
        ird_3_C = self.onemulone_ird_3_C(ird_2)
        ird_3_C = self.dcn_ird_3_C(ird_3_C) # two 3X3 equals one 5X5 deformable conv
        ird_3_C = self.dcn_ird_3_C(ird_3_C)
        ird_3_concat_ori = torch.cat([ird_3_A, ird_3_B, ird_3_C], dim=1)
        # print("size ofird_3_concat_ori is:", ird_3_concat_ori.size())
        ird_3_concat_se = self.se_3(ird_3_concat_ori)
        # print("size ofird_3_concat_se is:", ird_3_concat_se.size())
        ird_3_concat_residual = self.onemulone_ird_3_D(ird_3_concat_se)
        # ird_3_concat_residual = self.norm_ird_3(ird_3_concat_residual)
        # ird_3_concat_residual = self.relu_ird_3(ird_3_concat_residual)
        ird_3 = ird_2 + ird_3_concat_residual
        # print("ird_3 size is:", ird_3.size())
        ird_3 = self.relu(ird_3)

        ird_4_A = self.onemulone_ird_4_A(ird_3)
        ird_4_B = self.onemulone_ird_4_B(ird_3)
        ird_4_B = self.dcn_ird_4_B(ird_4_B)
        ird_4_C = self.onemulone_ird_4_C(ird_3)
        ird_4_C = self.dcn_ird_4_C(ird_4_C) # two 3X3 equals one 5X5 deformable conv
        ird_4_C = self.dcn_ird_4_C(ird_4_C)
        ird_4_concat_ori = torch.cat([ird_4_A, ird_4_B, ird_4_C], dim=1)
        # print("size ofird_4_concat_ori is:", ird_4_concat_ori.size())
        ird_4_concat_se = self.se_4(ird_4_concat_ori)
        # print("size ofird_4_concat_se is:", ird_4_concat_se.size())
        ird_4_concat_residual = self.onemulone_ird_4_D(ird_4_concat_se)
        # ird_4_concat_residual = self.norm_ird_4(ird_4_concat_residual)
        # ird_4_concat_residual = self.relu_ird_4(ird_4_concat_residual)
        ird_4 = ird_3 + ird_4_concat_residual
        # print("ird_4 size is:", ird_4.size())
        ird_4 = self.relu(ird_4)

        e4_h = self.conv4_1_h(ird_4)
        # e4_h = self.relu4_1_h(e4_h)
        # e4_h = self.conv4_2_h(e4_h)
        # e4_h = self.relu4_2_h(e4_h)
        # e4_h = self.conv4_3_h(e4_h)
        e4_h = self.norm4_h(e4_h)
        e4_h = self.relu4_3_h(e4_h)
        e4_h = self.max4_h(e4_h)
        # print("e4_h size is :", e4_h.size())

        ird_5_A = self.onemulone_ird_5_A(e4_h)
        ird_5_B = self.onemulone_ird_5_B(e4_h)
        ird_5_B = self.dcn_ird_5_B(ird_5_B)
        ird_5_C = self.onemulone_ird_5_C(e4_h)
        ird_5_C = self.dcn_ird_5_C(ird_5_C) # two 3X3 equals one 5X5 deformable conv
        ird_5_C = self.dcn_ird_5_C(ird_5_C)
        ird_5_concat_ori = torch.cat([ird_5_A, ird_5_B, ird_5_C], dim=1)
        # print("size ofird_5_concat_ori is:", ird_5_concat_ori.size())
        ird_5_concat_se = self.se_5(ird_5_concat_ori)
        # print("size ofird_5_concat_se is:", ird_5_concat_se.size())
        ird_5_concat_residual = self.onemulone_ird_5_D(ird_5_concat_se)
        # ird_5_concat_residual = self.norm_ird_5(ird_5_concat_residual)
        # ird_5_concat_residual = self.relu_ird_5(ird_5_concat_residual)
        ird_5 = e4_h + ird_5_concat_residual
        # print("ird_5 size is:", ird_5.size())
        ird_5 = self.relu(ird_5)

        ird_6_A = self.onemulone_ird_6_A(ird_5)
        ird_6_B = self.onemulone_ird_6_B(ird_5)
        ird_6_B = self.dcn_ird_6_B(ird_6_B)
        ird_6_C = self.onemulone_ird_6_C(ird_5)
        ird_6_C = self.dcn_ird_6_C(ird_6_C) # two 3X3 equals one 5X5 deformable conv
        ird_6_C = self.dcn_ird_6_C(ird_6_C)
        ird_6_concat_ori = torch.cat([ird_6_A, ird_6_B, ird_6_C], dim=1)
        # print("size ofird_6_concat_ori is:", ird_6_concat_ori.size())
        ird_6_concat_se = self.se_6(ird_6_concat_ori)
        # print("size ofird_6_concat_se is:", ird_6_concat_se.size())
        ird_6_concat_residual = self.onemulone_ird_6_D(ird_6_concat_se)
        # ird_6_concat_residual = self.norm_ird_6(ird_6_concat_residual)
        # ird_6_concat_residual = self.relu_ird_6(ird_6_concat_residual)
        ird_6 = ird_5 + ird_6_concat_residual
        # print("ird_6 size is:", ird_6.size())
        ird_6 = self.relu(ird_6)

        ird_7_A = self.onemulone_ird_7_A(ird_6)
        ird_7_B = self.onemulone_ird_7_B(ird_6)
        ird_7_B = self.dcn_ird_7_B(ird_7_B)
        ird_7_C = self.onemulone_ird_7_C(ird_6)
        ird_7_C = self.dcn_ird_7_C(ird_7_C) # two 3X3 equals one 5X5 deformable conv
        ird_7_C = self.dcn_ird_7_C(ird_7_C)
        ird_7_concat_ori = torch.cat([ird_7_A, ird_7_B, ird_7_C], dim=1)
        # print("size ofird_7_concat_ori is:", ird_7_concat_ori.size())
        ird_7_concat_se = self.se_7(ird_7_concat_ori)
        # print("size ofird_7_concat_se is:", ird_7_concat_se.size())
        ird_7_concat_residual = self.onemulone_ird_7_D(ird_7_concat_se)
        # ird_7_concat_residual = self.norm_ird_7(ird_7_concat_residual)
        # ird_7_concat_residual = self.relu_ird_7(ird_7_concat_residual)
        ird_7 = ird_6 + ird_7_concat_residual
        # print("ird_7 size is:", ird_7.size())
        ird_7 = self.relu(ird_7)

        ird_8_A = self.onemulone_ird_8_A(ird_7)
        ird_8_B = self.onemulone_ird_8_B(ird_7)
        ird_8_B = self.dcn_ird_8_B(ird_8_B)
        ird_8_C = self.onemulone_ird_8_C(ird_7)
        ird_8_C = self.dcn_ird_8_C(ird_8_C) # two 3X3 equals one 5X5 deformable conv
        ird_8_C = self.dcn_ird_8_C(ird_8_C)
        ird_8_concat_ori = torch.cat([ird_8_A, ird_8_B, ird_8_C], dim=1)
        # print("size ofird_8_concat_ori is:", ird_8_concat_ori.size())
        ird_8_concat_se = self.se_8(ird_8_concat_ori)
        # print("size ofird_8_concat_se is:", ird_8_concat_se.size())
        ird_8_concat_residual = self.onemulone_ird_8_D(ird_8_concat_se)
        # ird_8_concat_residual = self.norm_ird_8(ird_8_concat_residual)
        # ird_8_concat_residual = self.relu_ird_8(ird_8_concat_residual)
        ird_8 = ird_7 + ird_8_concat_residual
        # print("ird_8 size is:", ird_8.size())
        ird_8 = self.relu(ird_8)

        e5_h = self.conv5_1_h(ird_8)
        # e5_h = self.relu5_1_h(e5_h)
        # e5_h = self.conv5_2_h(e5_h)
        # e5_h = self.relu5_2_h(e5_h)
        # e5_h = self.conv5_3_h(e5_h)
        e5_h = self.norm5_h(e5_h)
        e5_h = self.relu5_3_h(e5_h)
        e5_h = self.max5_h(e5_h)
        # print("e5_h size is :", e5_h.size())

        e5_h_concat = torch.cat([e5_upsp, e5_h], dim=1)
        # print("shape of e5_h_concat:", e5_h_concat.size())

        # e5_sd = torch.cat([e5_s, e5_d], dim=1)
        e5_sd = self.channel_spatial_gate_1(e5_h_concat)
        # print("e5_sd size is :", e5_sd.size())

        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_sd)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())
        
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        # print("d2_h size is :", d2_h.size())

  
        d3_h = self.conv8_1_h(d2_h)
        # d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
        d4_h_upsp = self.upsp_output(d4_out)

        return d4_h_upsp, e5_h_concat, d2_h
        # return d4_out

class SAM_VGG_1(nn.Module): # SAM_VGG using ConvLSTM and dilated convolution, but without using softmax spatial attention mechanism
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(SAM_VGG_1, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        


        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=1,  padding=0) # change
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=1) # change padding=1,
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2,  padding=1) # change padding=1
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2,  padding=1) # change padding=1
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_3 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_4 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)

        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        # self.ConvLSTM_layer_1 = MyLSTM(input_size=(16,20), input_dim=1024, hidden_dim=[1024], 
                           # kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.ConvLSTM_layer_1 = MyLSTM(input_size=(48,64), input_dim=128, hidden_dim=[128], 
                            kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        # self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=1024, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')

            
        
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())

        e5_upsp = self.conv6_1(e5)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())


        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_upsp)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())
        '''
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        print("d2_h size is :", d2_h.size())
        '''
        
        d2_h = d1_h
        d2_h_sequence_unit = torch.unsqueeze(d2_h, dim=1)
        # print("shape of d2_h_sequence_unit:", d2_h_sequence_unit.size())


        # e5_h_sequence_1 = []
        d2_h_sequence_2 = d2_h_sequence_unit

        for n_sequence in range(4):
            # e5_h_sequence_1.append(e5_h_sequence_unit)
            d2_h_sequence_2 = torch.cat([d2_h_sequence_2, d2_h_sequence_unit], dim=1)


        # print("shape of e5_h_sequence_1:", e5_h_sequence_1.size())
        # print("shape of d2_h_sequence_2:", d2_h_sequence_2.size())

        layer_output_list, last_state_list = self.ConvLSTM_layer_1(d2_h_sequence_2)
        # print("layer_output_list is:", layer_output_list.size()) # list has no shape
        d2_h_refine = layer_output_list[0]
        # print("shape of d2_h_refine:", d2_h_refine.size())
        d2_h_refine = d2_h_refine[:, -1, :, :, :]
        # print("shape of d2_h_refine:", d2_h_refine.size())


        
        # d3_h = self.conv8_1_h(d2_h)
        d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        d4_upsp = self.upsp_output(d4_out)
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
        
        return d4_upsp, d1_h, e5
        # return d4_out

class SAM_VGG_2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(SAM_VGG_2, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=1,  padding=0) # change
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=1) # change padding=1,
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2,  padding=1) # change padding=1
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2,  padding=1) # change padding=1
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)

        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        # self.ConvLSTM_layer_1 = MyLSTM(input_size=(16,20), input_dim=1024, hidden_dim=[1024], 
                           # kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.ConvLSTM_layer_1 = MyLSTM(input_size=(48,64), input_dim=128, hidden_dim=[128], 
                            kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        # self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=1024, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')

        self.s_attention_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_ird_1 = nn.ReLU(inplace=True)
        self.softmax_1 = nn.Softmax()
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
            
        
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())

        e5_upsp = self.conv6_1(e5)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())


        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_upsp)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())
        '''
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        print("d2_h size is :", d2_h.size())
        '''
        s_attention_map = self.s_attention_conv(d1_h) # spatial attention mechnism
        s_attention_map = self.relu_ird_1(s_attention_map) 
        s_attention_map = self.softmax_1(s_attention_map)
        s_attention_map = s_attention_map.repeat(1, 128, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d2_h = d1_h.mul(s_attention_map)

        d2_h_sequence_unit = torch.unsqueeze(d2_h, dim=1)
        # print("shape of d2_h_sequence_unit:", d2_h_sequence_unit.size())


        # e5_h_sequence_1 = []
        d2_h_sequence_2 = d2_h_sequence_unit

        for n_sequence in range(4):
            # e5_h_sequence_1.append(e5_h_sequence_unit)
            d2_h_sequence_2 = torch.cat([d2_h_sequence_2, d2_h_sequence_unit], dim=1)


        # print("shape of e5_h_sequence_1:", e5_h_sequence_1.size())
        # print("shape of d2_h_sequence_2:", d2_h_sequence_2.size())

        layer_output_list, last_state_list = self.ConvLSTM_layer_1(d2_h_sequence_2)
        # print("layer_output_list is:", layer_output_list.size()) # list has no shape
        d2_h_refine = layer_output_list[0]
        # print("shape of d2_h_refine:", d2_h_refine.size())
        d2_h_refine = d2_h_refine[:, -1, :, :, :]
        # print("shape of d2_h_refine:", d2_h_refine.size())


        
        # d3_h = self.conv8_1_h(d2_h)
        d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        d4_upsp = self.upsp_output(d4_out)
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
        
        return d4_upsp, d1_h, e5_upsp
        # return d4_out

class SAM_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(SAM_ResNet, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.res_1_1 = ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_1_2 = ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_1_3 = ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_1_channel = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

        self.res_2_1 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_2_channel = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_3_1 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3_2 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3_3 = ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3_channel = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_4_1 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_2 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_3 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_4 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_5 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_6 = ResnetBlock_dilated(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4_channel = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.res_5_1 = ResnetBlock_dilated(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_5_2 = ResnetBlock_dilated(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_5_3 = ResnetBlock_dilated(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_5_channel = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        # self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ConvLSTM_layer_1 = MyLSTM(input_size=(62,78), input_dim=128, hidden_dim=[128], 
                            kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')

        self.s_attention_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_ird_1 = nn.ReLU(inplace=True)
        self.softmax_1 = nn.Softmax()

        self.conv6_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(128, affine=False)

        self.conv8_1 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        # self.norm8 = nn.InstanceNorm2d(3, affine=False)

        self.conv9_1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1 = nn.Tanh()


    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        # input_small = self.downsample(input)
    
        e1 = self.conv1_1(input)
        e1 = self.norm1(e1)
        e1 = self.relu1_1(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())

        res_1_1 = self.res_1_1(e1)
        res_1_2 = self.res_1_2(res_1_1)
        res_1_3 = self.res_1_3(res_1_2)
        res_1_3 = self.res_1_channel(res_1_3)
        # print("res_1_3 size is :", res_1_3.size())
        
        res_2_1 = self.res_2_1(res_1_3)
        res_2_1 = self.max1(res_2_1)
        # print("res_2_1 size is :", res_2_1.size())

        res_3_1 = self.res_3_1(res_2_1)
        res_3_2 = self.res_3_2(res_3_1)
        res_3_3 = self.res_3_3(res_3_2)
        res_3_3 = self.res_3_channel(res_3_3)
        # res_3_3 = self.max1(res_3_3)
        # print("res_3_3 size is :", res_3_3.size())

        res_4_1 = self.res_4_1(res_3_3)
        res_4_2 = self.res_4_2(res_4_1)
        res_4_3 = self.res_4_3(res_4_2)
        res_4_4 = self.res_4_4(res_4_3)
        res_4_5 = self.res_4_5(res_4_4)
        res_4_6 = self.res_4_6(res_4_5)
        res_4_6 = self.res_4_channel(res_4_6)
        # res_4_6 = self.max1(res_4_6)
        # print("res_4_6 size is :", res_4_6.size())

        res_5_1 = self.res_5_1(res_4_6)
        res_5_2 = self.res_5_2(res_5_1)
        res_5_3 = self.res_5_3(res_5_2)
        res_5_3 = self.res_5_channel(res_5_3)
        # print("res_5_3 size is :", res_5_3.size())

        s_attention_map = self.s_attention_conv(res_5_3) # spatial attention mechnism
        s_attention_map = self.relu_ird_1(s_attention_map) 
        s_attention_map = self.softmax_1(s_attention_map)
        s_attention_map = s_attention_map.repeat(1, 128, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d2_h = res_5_3.mul(s_attention_map)

        d2_h_sequence_unit = torch.unsqueeze(d2_h, dim=1)
        # print("shape of d2_h_sequence_unit:", d2_h_sequence_unit.size())
        d2_h_sequence_2 = d2_h_sequence_unit
        for n_sequence in range(4):
            # e5_h_sequence_1.append(e5_h_sequence_unit)
            d2_h_sequence_2 = torch.cat([d2_h_sequence_2, d2_h_sequence_unit], dim=1)

        # print("shape of e5_h_sequence_1:", e5_h_sequence_1.size())
        # print("shape of d2_h_sequence_2:", d2_h_sequence_2.size())
        layer_output_list, last_state_list = self.ConvLSTM_layer_1(d2_h_sequence_2)
        # print("layer_output_list is:", layer_output_list.size()) # list has no shape
        d2_h_refine = layer_output_list[0]
        # print("shape of d2_h_refine:", d2_h_refine.size())
        d2_h_refine = d2_h_refine[:, -1, :, :, :]
        # print("shape of d2_h_refine:", d2_h_refine.size())

        d3_h = self.conv6_1(d2_h_refine)
        d3_h = self.norm6(d3_h)
        d3_h = self.relu6_1(d3_h)
        # print("d3_h size is :", d3_h.size())

        d4_h = self.conv8_1(d3_h)
        d4_h = self.relu8_1(d4_h)
        # print("d4_h size is :", d4_h.size())
        
        d5_h = self.conv9_1(d4_h)
        d5_h = self.tanh9_1(d5_h)
        # print("d5_h size is :", d5_h.size())

        d5_out = torch.nn.functional.upsample_bilinear(d5_h, size=[120, 160])
        d5_upsp = self.upsp_output(d5_out)
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
              
        return d5_upsp, res_5_3, res_4_6

class GazeGAN_CSC(nn.Module): # GazeGAN_CSC model, which utilizes local and global U-Net equipped with cross-scale Center-Surround-Connections
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(GazeGAN_CSC, self).__init__()   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock

        # encoder of generator
        self.en_conv1 = encoderconv_2(input_nc, 64)
        self.en_conv2 = encoderconv_2(64, 128)
        # self.en_conv3 = encoderconv_2(128, 256) # local only
        self.en_conv3 = encoderconv_2(128+128, 256) # global + local
        self.en_conv4 = encoderconv_2(256, 512)
        self.en_conv5 = encoderconv_2(512, 1024)
        self.en_conv6 = encoderconv_2(1024, 1024)
        # self.en_conv7 = encoderconv_2(1024, 1024)
        # self.en_conv8 = encoderconv_2(1024, 1024)
        
        self.res_1 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_3 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_4 = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        self.de_conv1 = decoderconv_3(1024, 1024)
        self.de_conv2 = decoderconv_2(1024+1024, 512)
        self.de_conv3 = decoderconv_2(512+512, 256)
        self.de_conv4 = decoderconv_2(256+256+256, 128)
        # self.de_conv5 = decoderconv_2(128+128, 64) # local only
        self.de_conv5 = decoderconv_2(128+256+128, 64) # global + local
        self.de_conv6 = decoderconv_2(64+64, output_nc)

        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsp_output_csc = nn.Upsample(scale_factor=4, mode='bilinear')

        self.de_conv1_csc = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.relude_conv1_csc = nn.ReLU(inplace=True)
        self.norm_de_conv1_csc = nn.InstanceNorm2d(256, affine=False)
        self.de_conv2_csc = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relude_conv2_csc = nn.ReLU(inplace=True)
        self.norm_de_conv2_csc = nn.InstanceNorm2d(128, affine=False)

        # bottle-neck layer
        self.dimr_conv1 = dimredconv(output_nc, output_nc)
        # 2X downsampling model for input image
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.en_conv1_g = encoderconv_2(input_nc, 128)
        self.en_conv2_g = encoderconv_2(128, 256)
        self.en_conv3_g = encoderconv_2(256, 512)
        self.en_conv4_g = encoderconv_2(512, 1024)
        self.en_conv5_g = encoderconv_2(1024, 1024)
        self.res_g = ResnetBlock(1024, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.de_conv1_g = decoderconv_3(1024, 1024)
        self.de_conv2_g = decoderconv_2(1024+1024, 512)
        self.de_conv3_g = decoderconv_2(512+512, 256)
        self.de_conv4_g = decoderconv_2(256+256+256, 128)
        self.de_conv5_g = decoderconv_2(128+128+128, output_nc)

        self.de_conv1_g_csc = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.relude_conv1_g_csc = nn.ReLU(inplace=True)
        self.norm_de_conv1_g_csc = nn.InstanceNorm2d(256, affine=False)
        self.de_conv2_g_csc = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relude_conv2_g_csc = nn.ReLU(inplace=True)
        self.norm_de_conv2_g_csc = nn.InstanceNorm2d(128, affine=False)

        self.s_attention_conv_1 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_attention_1 = nn.ReLU(inplace=True)
        self.softmax_1 = nn.Softmax()

        self.s_attention_conv_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_attention_2 = nn.ReLU(inplace=True)
        self.softmax_2 = nn.Softmax()

        self.s_attention_conv_3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_attention_3 = nn.ReLU(inplace=True)
        self.softmax_3 = nn.Softmax()

        self.s_attention_conv_4 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_attention_4 = nn.ReLU(inplace=True)
        self.softmax_4 = nn.Softmax()

    def forward(self, input):
        input_ds = self.downsample(input)
        # print("input :", input.size())
        # print("input_ds :", input_ds.size())


        # global U-Net
        e1_g = self.en_conv1_g(input_ds)
        e2_g = self.en_conv2_g(e1_g)
        e3_g = self.en_conv3_g(e2_g)
        e4_g = self.en_conv4_g(e3_g)
        e5_g = self.en_conv5_g(e4_g)

        res1_g = self.res_g(e5_g)
        res2_g = self.res_g(res1_g)
        res3_g = self.res_g(res2_g)
        res4_g = self.res_g(res3_g)
        # print("res1_g :", res1_g.size())

        d1_g = self.de_conv1_g(res4_g)
        # print("d1_g and e4_g are :", d1_g.size(), e4_g.size())
        d1_g_csc = self.de_conv1_g_csc(d1_g)
        d1_g_csc = self.upsp_output_csc(d1_g_csc)
        # print("d1_g_csc are :", d1_g_csc.size()) 

        d2_g = self.de_conv2_g(torch.cat([d1_g, e4_g], dim=1))
        d2_g_csc = self.de_conv2_g_csc(d2_g)
        d2_g_csc = self.upsp_output_csc(d2_g_csc)
        # print("d2_g and e3_g are :", d2_g.size(), e3_g.size())

        # d3_g = self.de_conv3_g(torch.cat([d2_g, e3_g], dim=1))
        d3_g = self.de_conv3_g(torch.cat([d2_g, e3_g], dim=1))

        d1_g_csc = d1_g_csc + d3_g
        # d1_g_csc = self.norm_de_conv1_g_csc(d1_g_csc)
        d1_g_csc = self.relude_conv1_g_csc(d1_g_csc)
        s_attention_map_1 = self.s_attention_conv_1(d1_g_csc) # spatial attention mechnism
        s_attention_map_1 = self.relu_attention_1(s_attention_map_1) 
        s_attention_map_1 = self.softmax_1(s_attention_map_1)
        s_attention_map_1 = s_attention_map_1.repeat(1, 256, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d1_g_csc = d1_g_csc.mul(s_attention_map_1)

        # print("d3_g and e2_g are :", d3_g.size(), e2_g.size())

        d4_g = self.de_conv4_g(torch.cat([d3_g, e2_g, d1_g_csc], dim=1))
        # print("d4_g and e1_g are :", d4_g.size(), e1_g.size())
        d2_g_csc = d2_g_csc + d4_g
        # d2_g_csc = self.norm_de_conv2_g_csc(d2_g_csc)
        d2_g_csc = self.relude_conv2_g_csc(d2_g_csc)
        s_attention_map_2 = self.s_attention_conv_2(d2_g_csc) # spatial attention mechnism
        s_attention_map_2 = self.relu_attention_2(s_attention_map_2) 
        s_attention_map_2 = self.softmax_2(s_attention_map_2)
        s_attention_map_2 = s_attention_map_2.repeat(1, 128, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d2_g_csc = d2_g_csc.mul(s_attention_map_2)

        d5_g = self.de_conv5_g(torch.cat([d4_g, e1_g, d2_g_csc], dim=1))
        # print("d5_g are :", d5_g.size()) 
        d6_g = self.dimr_conv1(d5_g) # d6_g is a small output saliency map



        # local U-Net
        e1 = self.en_conv1(input)
        # print("size of input is :", input.size())
        # print("size of e1 is :", e1.size())
        e2_local = self.en_conv2(e1)

        # e2 = torch.add(d4_g, e2_local) # pooling the global features into local UNet
        # e2 = d4_g + e2_local # pooling the global features into local UNet
        e2 = torch.cat([e2_local, d4_g], dim=1)

        # print("size of e2 is :", e2.size())
        e3 = self.en_conv3(e2)
        # print("size of e3 is :", e3.size())
        e4 = self.en_conv4(e3)
        # print("size of e4 is :", e4.size())
        e5 = self.en_conv5(e4)
        # print("size of e5 is :", e5.size())
        e6 = self.en_conv6(e5)
        # print("size of e6 is :", e6.size())
        # e7 = self.en_conv7(e6)
        # print("size of e7 is :", e7.size())
        # e8 = self.en_conv8(e7)
        # print("size of e8 is :", e8.size())

        res1 = self.res_1(e6)
        res2 = self.res_2(res1)
        res3 = self.res_2(res2)
        res4 = self.res_2(res3)

        # print("res1 :", res1.size())
        # print("res2 :", res2.size())
        # print("res3 :", res3.size())
        # print("res4 :", res4.size())

        # d1 = self.de_conv1(e6)
        d1 = self.de_conv1(res4)
        # print("d1 and e5 are :", d1.size(), e5.size())
        d1_csc = self.de_conv1_csc(d1)
        d1_csc = self.upsp_output_csc(d1_csc)
        # print("d1_csc are :", d1_csc.size()) 

        d2 = self.de_conv2(torch.cat([d1, e5], dim=1))
        # d2 = self.de_conv2(d1)
        # print("d2 and e4 are :", d2.size(), e4.size())
        d2_csc = self.de_conv2_csc(d2)
        d2_csc = self.upsp_output_csc(d2_csc)
        # print("d2_g and e3_g are :", d2_g.size(), e3_g.size())

        d3 = self.de_conv3(torch.cat([d2, e4], dim=1))
        # d3 = self.de_conv3(d2)

        d1_csc = d1_csc + d3
        # d1_csc = self.norm_de_conv1_csc(d1_csc)
        d1_csc = self.relude_conv1_csc(d1_csc) # nonlinear operation
        s_attention_map_3 = self.s_attention_conv_3(d1_csc) # spatial attention mechnism
        s_attention_map_3 = self.relu_attention_3(s_attention_map_3) 
        s_attention_map_3 = self.softmax_3(s_attention_map_3)
        s_attention_map_3 = s_attention_map_3.repeat(1, 256, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d1_csc = d1_csc.mul(s_attention_map_3)

        # print("d3 and e3 are :", d3.size(), e3.size())

        d4 = self.de_conv4(torch.cat([d3, e3, d1_csc], dim=1))

        # d4 = self.de_conv4(d3)
        d2_csc = d2_csc + d4
        # d2_csc = self.norm_de_conv2_csc(d2_csc)
        d2_csc = self.relude_conv2_csc(d2_csc)
        s_attention_map_4 = self.s_attention_conv_4(d2_csc) # spatial attention mechnism
        s_attention_map_4 = self.relu_attention_4(s_attention_map_4) 
        s_attention_map_4 = self.softmax_4(s_attention_map_4)
        s_attention_map_4 = s_attention_map_4.repeat(1, 128, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d2_csc = d2_csc.mul(s_attention_map_4)

        # print("d3 and e3 are :", d3.size(), e3.size())
        # print("d4 and e2 are :", d4.size(), e2.size())
        
        d5 = self.de_conv5(torch.cat([d4, e2, d2_csc], dim=1))
        # d5 = self.de_conv5(d4)
        # print("d5 and e1 are :", d5.size(), e1.size())

        d6 = self.de_conv6(torch.cat([d5, e1], dim=1))
        # print("d6 is :", d6.size())
        
        # d6 = self.de_conv6(d5)
        
        d7 = self.dimr_conv1(d6)
        # print("d7 is :", d7.size())
        
        
        out = d7 # the real final output
        # out = torch.squeeze(e1, 0)

        # out = e1
        # out = out[0:2, :, :]
        # print("out is :", out, out.size())
        '''
        out1 = d5
        # out = out1[0:1, 0:3, :, :] # this is right
        out = torch.mean(out1, 1) # mean across 64 channel direction
        out = torch.unsqueeze(out, 0)
        print("out1 size is :", out1.size())
        print("out size is :", out.size())
        print("d7 size is :", d7.size())
        '''
        # return out
        return out, res4, d4

class SalGAN_BCE(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(SalGAN_BCE, self).__init__()   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock

        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        


        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2) # change
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1,
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.relu6_2 = nn.ReLU(inplace=True)
        self.conv6_3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1) # equal to upssampling
        self.norm6 = nn.InstanceNorm2d(512, affine=False)
        self.relu6_3 = nn.ReLU(inplace=True)

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) 
        self.relu7_2 = nn.ReLU(inplace=True)
        self.conv7_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) # equal to upssampling
        self.norm7 = nn.InstanceNorm2d(256, affine=False)
        self.relu7_3 = nn.ReLU(inplace=True)

        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1) 
        self.relu8_2 = nn.ReLU(inplace=True)
        self.conv8_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) # equal to upssampling
        self.norm8 = nn.InstanceNorm2d(128, affine=False)
        self.relu8_3 = nn.ReLU(inplace=True)

        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) 
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) # equal to upssampling
        self.norm9 = nn.InstanceNorm2d(64, affine=False)
        self.relu9_2 = nn.ReLU(inplace=True)

        self.conv10_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1) 
        self.relu10_1 = nn.ReLU(inplace=True)
        self.conv10_2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1) # equal to upssampling
        self.norm10 = nn.InstanceNorm2d(3, affine=False)
        self.relu10_2 = nn.ReLU(inplace=True)
        
        self.conv11_1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh11_1 = nn.Tanh()
        # self.upsm_6 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        # input_ds = self.downsample(input)
        # print("input :", input.size())
        # print("input_ds :", input_ds.size())
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())

        e6 = self.conv6_1(e5)
        e6 = self.relu6_1(e6)
        e6 = self.conv6_2(e6)
        e6 = self.relu6_2(e6)
        e6 = self.conv6_3(e6)
        e6 = self.norm6(e6)
        e6 = self.relu6_3(e6)
        # print("e6 size is :", e6.size())

        e7 = self.conv7_1(e6)
        e7 = self.relu7_1(e7)
        e7 = self.conv7_2(e7)
        e7 = self.relu7_2(e7)
        e7 = self.conv7_3(e7)
        e7 = self.norm7(e7)
        e7 = self.relu7_3(e7)
        # print("e7 size is :", e7.size())

        e8 = self.conv8_1(e7)
        e8 = self.relu8_1(e8)
        e8 = self.conv8_2(e8)
        e8 = self.relu8_2(e8)
        e8 = self.conv8_3(e8)
        e8 = self.norm8(e8)
        e8 = self.relu8_3(e8)
        # print("e8 size is :", e8.size())

        e9 = self.conv9_1(e8)
        e9 = self.relu9_1(e9)
        e9 = self.conv9_2(e9)
        e9 = self.norm9(e9)
        e9 = self.relu9_2(e9)
        # print("e9 size is :", e9.size())

        e10 = self.conv10_1(e9)
        e10 = self.relu10_1(e10)
        e10 = self.conv10_2(e10)
        # e10 = self.norm10(e10)
        e10 = self.relu10_2(e10)
        # print("e10 size is :", e10.size())

        e11 = self.conv11_1(e10)
        e11 = self.tanh11_1(e11)
        
        e11_out = torch.nn.functional.upsample_bilinear(e11, size=[480, 640])
        return e11_out, e5, e9
        # return e11_out, e11_out, e11_out

class DCN_Inception(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(DCN_Inception, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.conv1_1_h = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1_h = nn.ReLU(inplace=True)
        self.conv1_2_h = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm1_h = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2_h = nn.ReLU(inplace=True)
        self.max1_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1_h = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1_h = nn.ReLU(inplace=True)
        self.conv2_2_h = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2_h = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2_h = nn.ReLU(inplace=True)
        self.max2_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1_h = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1_h = nn.ReLU(inplace=True)
        self.conv3_2_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2_h = nn.ReLU(inplace=True)
        self.conv3_3_h = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3_h = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3_h = nn.ReLU(inplace=True)
        self.max3_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1_h = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1_h = nn.ReLU(inplace=True)
        self.conv4_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2_h = nn.ReLU(inplace=True)
        self.conv4_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4_h = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3_h = nn.ReLU(inplace=True)
        self.max4_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1_h = nn.ReLU(inplace=True)
        self.conv5_2_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2_h = nn.ReLU(inplace=True)
        self.conv5_3_h = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm5_h = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3_h = nn.ReLU(inplace=True)
        self.max5_h = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512 + 512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)
        
        self.conv7_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu7_1_h = nn.ReLU(inplace=True)
        self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        self.onemulone_ird_1_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_1 = SELayer(384, reduction=8)
        self.onemulone_ird_1_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_1 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_1 = nn.ReLU(inplace=True)

        self.onemulone_ird_2_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_2_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_2_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_2 = SELayer(384, reduction=8)
        self.onemulone_ird_2_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_2 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_2 = nn.ReLU(inplace=True)

        self.onemulone_ird_3_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_3_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_3_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_3 = SELayer(384, reduction=8)
        self.onemulone_ird_3_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_3 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_3 = nn.ReLU(inplace=True)

        self.onemulone_ird_4_A = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_B = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_4_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_4_C = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.se_4 = SELayer(384, reduction=8)
        self.onemulone_ird_4_D = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
        self.norm_ird_4 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_4 = nn.ReLU(inplace=True)

        self.onemulone_ird_5_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_5_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_5_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_5 = SELayer(384, reduction=8)
        self.onemulone_ird_5_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_5 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_5 = nn.ReLU(inplace=True)

        self.onemulone_ird_6_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_6_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_6_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_6 = SELayer(384, reduction=8)
        self.onemulone_ird_6_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_6 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_6 = nn.ReLU(inplace=True)

        self.onemulone_ird_7_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_7_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_7_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_7 = SELayer(384, reduction=8)
        self.onemulone_ird_7_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_7 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_7 = nn.ReLU(inplace=True)

        self.onemulone_ird_8_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_8_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_8_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_8 = SELayer(384, reduction=8)
        self.onemulone_ird_8_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_8 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_8 = nn.ReLU(inplace=True)

        self.onemulone_ird_9_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_9_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_9_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_9 = SELayer(384, reduction=8)
        self.onemulone_ird_9_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_9 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_9 = nn.ReLU(inplace=True)

        self.onemulone_ird_10_A = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_B = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_10_C = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_10_C = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.se_10 = SELayer(384, reduction=8)
        self.onemulone_ird_10_D = nn.Conv2d(384, 512, kernel_size=1, stride=1, padding=0)
        self.norm_ird_10 = nn.InstanceNorm2d(512, affine=False)
        self.relu_ird_10 = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)

        self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=512, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')

  
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        # input_small = self.downsample(input)
 
        e1_h = self.conv1_1_h(input)
        e1_h = self.norm1_h(e1_h)
        e1_h = self.relu1_2_h(e1_h)
        e1_h = self.max1_h(e1_h)
        # print("e1_h size is :", e1_h.size())
        
        e2_h = self.conv2_1_h(e1_h)
        e2_h = self.norm2_h(e2_h)
        e2_h = self.relu2_2_h(e2_h)
        e2_h = self.max2_h(e2_h)
        # print("e2_h size is :", e2_h.size())
        
        e3_h = self.conv3_1_h(e2_h)
        e3_h = self.norm3_h(e3_h)
        e3_h = self.relu3_3_h(e3_h)
        e3_h = self.max3_h(e3_h)
        # print("e3_h size is :", e3_h.size())

        ird_1_A = self.onemulone_ird_1_A(e3_h)
        ird_1_B = self.onemulone_ird_1_B(e3_h)
        ird_1_B = self.dcn_ird_1_B(ird_1_B)
        ird_1_C = self.onemulone_ird_1_C(e3_h)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_C = self.dcn_ird_1_C(ird_1_C)
        ird_1_concat_ori = torch.cat([ird_1_A, ird_1_B, ird_1_C], dim=1)
        # print("size ofird_1_concat_ori is:", ird_1_concat_ori.size())
        ird_1_concat_se = self.se_1(ird_1_concat_ori)
        # print("size ofird_1_concat_se is:", ird_1_concat_se.size())
        ird_1_concat_residual = self.onemulone_ird_1_D(ird_1_concat_se)
        # ird_1_concat_residual = self.norm_ird_1(ird_1_concat_residual)
        # ird_1_concat_residual = self.relu_ird_1(ird_1_concat_residual)
        ird_1 = e3_h + ird_1_concat_residual
        # print("ird_1 size is:", ird_1.size())
        ird_1 = self.relu(ird_1)

        ird_2_A = self.onemulone_ird_2_A(ird_1)
        ird_2_B = self.onemulone_ird_2_B(ird_1)
        ird_2_B = self.dcn_ird_2_B(ird_2_B)
        ird_2_C = self.onemulone_ird_2_C(ird_1)
        ird_2_C = self.dcn_ird_2_C(ird_2_C) # two 3X3 equals one 5X5 deformable conv
        ird_2_C = self.dcn_ird_2_C(ird_2_C)
        ird_2_concat_ori = torch.cat([ird_2_A, ird_2_B, ird_2_C], dim=1)
        # print("size ofird_2_concat_ori is:", ird_2_concat_ori.size())
        ird_2_concat_se = self.se_2(ird_2_concat_ori)
        # print("size ofird_2_concat_se is:", ird_2_concat_se.size())
        ird_2_concat_residual = self.onemulone_ird_2_D(ird_2_concat_se)
        # ird_2_concat_residual = self.norm_ird_2(ird_2_concat_residual)
        # ird_2_concat_residual = self.relu_ird_2(ird_2_concat_residual)
        ird_2 = ird_1 + ird_2_concat_residual
        # print("ird_2 size is:", ird_2.size()) 
        ird_2 = self.relu(ird_2)      

        ird_3_A = self.onemulone_ird_3_A(ird_2)
        ird_3_B = self.onemulone_ird_3_B(ird_2)
        ird_3_B = self.dcn_ird_3_B(ird_3_B)
        ird_3_C = self.onemulone_ird_3_C(ird_2)
        ird_3_C = self.dcn_ird_3_C(ird_3_C) # two 3X3 equals one 5X5 deformable conv
        ird_3_C = self.dcn_ird_3_C(ird_3_C)
        ird_3_concat_ori = torch.cat([ird_3_A, ird_3_B, ird_3_C], dim=1)
        # print("size ofird_3_concat_ori is:", ird_3_concat_ori.size())
        ird_3_concat_se = self.se_3(ird_3_concat_ori)
        # print("size ofird_3_concat_se is:", ird_3_concat_se.size())
        ird_3_concat_residual = self.onemulone_ird_3_D(ird_3_concat_se)
        # ird_3_concat_residual = self.norm_ird_3(ird_3_concat_residual)
        # ird_3_concat_residual = self.relu_ird_3(ird_3_concat_residual)
        ird_3 = ird_2 + ird_3_concat_residual
        # print("ird_3 size is:", ird_3.size())
        ird_3 = self.relu(ird_3)

        ird_4_A = self.onemulone_ird_4_A(ird_3)
        ird_4_B = self.onemulone_ird_4_B(ird_3)
        ird_4_B = self.dcn_ird_4_B(ird_4_B)
        ird_4_C = self.onemulone_ird_4_C(ird_3)
        ird_4_C = self.dcn_ird_4_C(ird_4_C) # two 3X3 equals one 5X5 deformable conv
        ird_4_C = self.dcn_ird_4_C(ird_4_C)
        ird_4_concat_ori = torch.cat([ird_4_A, ird_4_B, ird_4_C], dim=1)
        # print("size ofird_4_concat_ori is:", ird_4_concat_ori.size())
        ird_4_concat_se = self.se_4(ird_4_concat_ori)
        # print("size ofird_4_concat_se is:", ird_4_concat_se.size())
        ird_4_concat_residual = self.onemulone_ird_4_D(ird_4_concat_se)
        # ird_4_concat_residual = self.norm_ird_4(ird_4_concat_residual)
        # ird_4_concat_residual = self.relu_ird_4(ird_4_concat_residual)
        ird_4 = ird_3 + ird_4_concat_residual
        # print("ird_4 size is:", ird_4.size())
        ird_4 = self.relu(ird_4)

        e4_h = self.conv4_1_h(ird_4)
        # e4_h = self.relu4_1_h(e4_h)
        # e4_h = self.conv4_2_h(e4_h)
        # e4_h = self.relu4_2_h(e4_h)
        # e4_h = self.conv4_3_h(e4_h)
        e4_h = self.norm4_h(e4_h)
        e4_h = self.relu4_3_h(e4_h)
        e4_h = self.max4_h(e4_h)
        # print("e4_h size is :", e4_h.size())

        ird_5_A = self.onemulone_ird_5_A(e4_h)
        ird_5_B = self.onemulone_ird_5_B(e4_h)
        ird_5_B = self.dcn_ird_5_B(ird_5_B)
        ird_5_C = self.onemulone_ird_5_C(e4_h)
        ird_5_C = self.dcn_ird_5_C(ird_5_C) # two 3X3 equals one 5X5 deformable conv
        ird_5_C = self.dcn_ird_5_C(ird_5_C)
        ird_5_concat_ori = torch.cat([ird_5_A, ird_5_B, ird_5_C], dim=1)
        # print("size ofird_5_concat_ori is:", ird_5_concat_ori.size())
        ird_5_concat_se = self.se_5(ird_5_concat_ori)
        # print("size ofird_5_concat_se is:", ird_5_concat_se.size())
        ird_5_concat_residual = self.onemulone_ird_5_D(ird_5_concat_se)
        # ird_5_concat_residual = self.norm_ird_5(ird_5_concat_residual)
        # ird_5_concat_residual = self.relu_ird_5(ird_5_concat_residual)
        ird_5 = e4_h + ird_5_concat_residual
        # print("ird_5 size is:", ird_5.size())
        ird_5 = self.relu(ird_5)

        ird_6_A = self.onemulone_ird_6_A(ird_5)
        ird_6_B = self.onemulone_ird_6_B(ird_5)
        ird_6_B = self.dcn_ird_6_B(ird_6_B)
        ird_6_C = self.onemulone_ird_6_C(ird_5)
        ird_6_C = self.dcn_ird_6_C(ird_6_C) # two 3X3 equals one 5X5 deformable conv
        ird_6_C = self.dcn_ird_6_C(ird_6_C)
        ird_6_concat_ori = torch.cat([ird_6_A, ird_6_B, ird_6_C], dim=1)
        # print("size ofird_6_concat_ori is:", ird_6_concat_ori.size())
        ird_6_concat_se = self.se_6(ird_6_concat_ori)
        # print("size ofird_6_concat_se is:", ird_6_concat_se.size())
        ird_6_concat_residual = self.onemulone_ird_6_D(ird_6_concat_se)
        # ird_6_concat_residual = self.norm_ird_6(ird_6_concat_residual)
        # ird_6_concat_residual = self.relu_ird_6(ird_6_concat_residual)
        ird_6 = ird_5 + ird_6_concat_residual
        # print("ird_6 size is:", ird_6.size())
        ird_6 = self.relu(ird_6)

        ird_7_A = self.onemulone_ird_7_A(ird_6)
        ird_7_B = self.onemulone_ird_7_B(ird_6)
        ird_7_B = self.dcn_ird_7_B(ird_7_B)
        ird_7_C = self.onemulone_ird_7_C(ird_6)
        ird_7_C = self.dcn_ird_7_C(ird_7_C) # two 3X3 equals one 5X5 deformable conv
        ird_7_C = self.dcn_ird_7_C(ird_7_C)
        ird_7_concat_ori = torch.cat([ird_7_A, ird_7_B, ird_7_C], dim=1)
        # print("size ofird_7_concat_ori is:", ird_7_concat_ori.size())
        ird_7_concat_se = self.se_7(ird_7_concat_ori)
        # print("size ofird_7_concat_se is:", ird_7_concat_se.size())
        ird_7_concat_residual = self.onemulone_ird_7_D(ird_7_concat_se)
        # ird_7_concat_residual = self.norm_ird_7(ird_7_concat_residual)
        # ird_7_concat_residual = self.relu_ird_7(ird_7_concat_residual)
        ird_7 = ird_6 + ird_7_concat_residual
        # print("ird_7 size is:", ird_7.size())
        ird_7 = self.relu(ird_7)

        ird_8_A = self.onemulone_ird_8_A(ird_7)
        ird_8_B = self.onemulone_ird_8_B(ird_7)
        ird_8_B = self.dcn_ird_8_B(ird_8_B)
        ird_8_C = self.onemulone_ird_8_C(ird_7)
        ird_8_C = self.dcn_ird_8_C(ird_8_C) # two 3X3 equals one 5X5 deformable conv
        ird_8_C = self.dcn_ird_8_C(ird_8_C)
        ird_8_concat_ori = torch.cat([ird_8_A, ird_8_B, ird_8_C], dim=1)
        # print("size ofird_8_concat_ori is:", ird_8_concat_ori.size())
        ird_8_concat_se = self.se_8(ird_8_concat_ori)
        # print("size ofird_8_concat_se is:", ird_8_concat_se.size())
        ird_8_concat_residual = self.onemulone_ird_8_D(ird_8_concat_se)
        # ird_8_concat_residual = self.norm_ird_8(ird_8_concat_residual)
        # ird_8_concat_residual = self.relu_ird_8(ird_8_concat_residual)
        ird_8 = ird_7 + ird_8_concat_residual
        # print("ird_8 size is:", ird_8.size())
        ird_8 = self.relu(ird_8)

        e5_h = self.conv5_1_h(ird_8)
        # e5_h = self.relu5_1_h(e5_h)
        # e5_h = self.conv5_2_h(e5_h)
        # e5_h = self.relu5_2_h(e5_h)
        # e5_h = self.conv5_3_h(e5_h)
        e5_h = self.norm5_h(e5_h)
        e5_h = self.relu5_3_h(e5_h)
        e5_h = self.max5_h(e5_h)
        # print("e5_h size is :", e5_h.size())

        # e5_h_concat = torch.cat([e5_upsp, e5_h], dim=1)
        # e5_h_concat = torch.cat([e5_upsp, e5_upsp], dim=1) # defense try XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # print("shape of e5_h_concat:", e5_h_concat.size())

        # e5_sd = torch.cat([e5_s, e5_d], dim=1)
        e5_sd = self.channel_spatial_gate_1(e5_h)
        # print("e5_sd size is :", e5_sd.size())

        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_sd)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())
        
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        # print("d2_h size is :", d2_h.size())

  
        d3_h = self.conv8_1_h(d2_h)
        # d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
        d4_h_upsp = self.upsp_output(d4_out)

        return d4_h_upsp, e5_h, e5_sd
        # return d4_out

class DeepGaze_only_VGG(nn.Module): # using VGG-19 rather than VGG-16 
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(DeepGaze_only_VGG, self).__init__()   

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock

        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.upsp = nn.Upsample(scale_factor=8, mode='bilinear')

        ## VGG-19 feature extractor
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2) # change
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1,
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1*1 Conv serving as Readout network
        # self.rdot_1 = nn.Conv2d(512*5, 16, kernel_size=1, stride=1, padding=0)
        self.rdot_1 = nn.Conv2d(512*5, 512, kernel_size=1, stride=1, padding=0)
        self.relu_rdot_1 = nn.ReLU(inplace=True)

        # self.rdot_2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.rdot_2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.relu_rdot_2 = nn.ReLU(inplace=True)

        # self.rdot_3 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
        self.rdot_3 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        self.relu_rdot_3 = nn.ReLU(inplace=True)

        self.rdot_4 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        # self.relu_rdot_4 = nn.ReLU(inplace=True) 
        self.tanh_rdot_4 = nn.Tanh() 
        self.softmax_rdot_4 = nn.Softmax()       


    def forward(self, input):
        # input_ds = self.downsample(input)
        # print("input :", input.size())
        # print("input_ds :", input_ds.size())
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        # input_small = self.downsample(input)
    
        e1 = self.conv1_1(input)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.conv3_4(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_4(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.relu4_3(e4)
        e4 = self.conv4_4(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_4(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5_1 = self.conv5_1(e4)
        e5_2 = self.relu5_1(e5_1)
        e5_3 = self.conv5_2(e5_2)
        e5_4 = self.relu5_2(e5_3)
        e5_5 = self.conv5_3(e5_4)
        e5_6 = self.relu5_3(e5_5)
        e5_7 = self.conv5_4(e5_6)
        e5_8 = self.norm5(e5_7)
        e5_9 = self.relu5_4(e5_8)
        e5_10 = self.max5(e5_9)
        # print("e5 size is :", e5_10.size())

        e5_concat = torch.cat([e5_1, e5_2, e5_4, e5_5, e5_9], dim=1)
        e5_concat = self.upsp(e5_concat)

        rdot_1 = self.rdot_1(e5_concat)
        rdot_1 = self.relu_rdot_1(rdot_1)
        # print("rdot_1 size is :", rdot_1.size())

        rdot_2 = self.rdot_2(rdot_1)
        rdot_2 = self.relu_rdot_2(rdot_2)
        # print("rdot_2 size is :", rdot_2.size())

        rdot_3 = self.rdot_3(rdot_2)
        rdot_3 = self.relu_rdot_3(rdot_3)
        # print("rdot_3 size is :", rdot_3.size())

        rdot_4 = self.rdot_4(rdot_3)
        # rdot_4 = self.softmax_rdot_4(rdot_4)
        rdot_4 = self.tanh_rdot_4(rdot_4)
        # print("rdot_4 size is :", rdot_4.size())
        
        rdot_4_out = torch.nn.functional.upsample_bilinear(rdot_4, size=[480, 640])
        return rdot_4_out, e5_10, rdot_2

class DCN_SAM_VGG(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(DCN_SAM_VGG, self).__init__()
        
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # for resblock
        activation = nn.ReLU(True) # for resblock
        padding_type='reflect' # for resblock
        
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
        # self.dcn_ird_2_B = ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        '''
        self.conv1_1 = ModulatedDeformConvPack(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 =  ModulatedDeformConvPack(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = ModulatedDeformConvPack(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = ModulatedDeformConvPack(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = ModulatedDeformConvPack(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = ModulatedDeformConvPack(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = ModulatedDeformConvPack(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ModulatedDeformConvPack(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=1, no_bias=True, use_flag=True).cuda()
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2,  padding=0) # change

        self.conv5_1 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True, use_flag=True).cuda()
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True, use_flag=True).cuda()
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True, use_flag=True).cuda()
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        '''

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=False)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2,  padding=0) # change
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1,
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1) # change padding=1
        # self.conv5_3 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True, use_flag=True).cuda()
        self.norm5 = nn.InstanceNorm2d(512, affine=False)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.res_1 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_2 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_3 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        # self.res_4 = ResnetBlock(512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        
        # self.conv6_1 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True, use_flag=True).cuda()
        self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.norm6 = nn.InstanceNorm2d(512, affine=False)

        self.conv6_2 = ModulatedDeformConvPack(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        # self.conv6_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.norm6_2 = nn.InstanceNorm2d(512, affine=False)

        self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv6_1_h = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu6_1_h = nn.ReLU(inplace=True)
        self.norm6_h = nn.InstanceNorm2d(512, affine=False)

        self.conv6_1_h_2 = ModulatedDeformConvPack(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.relu6_1_h_2 = nn.ReLU(inplace=True)
        self.norm6_1_h_2 = nn.InstanceNorm2d(128, affine=False)
        
        # self.conv7_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.relu7_1_h = nn.ReLU(inplace=True)
        # self.norm7_h = nn.InstanceNorm2d(128, affine=False)
        
        self.conv8_1_h = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu8_1_h = nn.ReLU(inplace=True)
        self.norm8_h = nn.InstanceNorm2d(3, affine=False)
        
        self.conv9_1_h = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh9_1_h = nn.Tanh()

        # self.ConvLSTM_layer_1 = MyLSTM(input_size=(16,20), input_dim=1024, hidden_dim=[1024], 
                           # kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        self.ConvLSTM_layer_1 = MyLSTM(input_size=(32,40), input_dim=128, hidden_dim=[128], 
                            kernel_size=(3,3), num_layers=1, batch_first=True, bias=True, return_all_layers=False)

        # self.channel_spatial_gate_1 = Spatial_Channel_Gate_Layer(dim_in=1024, dim_redu=256)
        
        self.upsp_output = nn.Upsample(scale_factor=4, mode='bilinear')

        self.s_attention_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0) # spatial attention
        self.relu_ird_1 = nn.ReLU(inplace=True)
        self.softmax_1 = nn.Softmax()
            
        
    def forward(self, input):   
        # print("original input", input.size())
        input = torch.nn.functional.upsample_bilinear(input, size=[512, 640]) 
        # print("original input", input.size())
        input_small = self.downsample(input)
    
        e1 = self.conv1_1(input_small)
        e1 = self.relu1_1(e1)
        e1 = self.conv1_2(e1)
        e1 = self.norm1(e1)
        e1 = self.relu1_2(e1)
        e1 = self.max1(e1)
        # print("e1 size is :", e1.size())
        
        e2 = self.conv2_1(e1)
        e2 = self.relu2_1(e2)
        e2 = self.conv2_2(e2)
        e2 = self.norm2(e2)
        e2 = self.relu2_2(e2)
        e2 = self.max2(e2)
        # print("e2 size is :", e2.size())
        
        e3 = self.conv3_1(e2)
        e3 = self.relu3_1(e3)
        e3 = self.conv3_2(e3)
        e3 = self.relu3_2(e3)
        e3 = self.conv3_3(e3)
        e3 = self.norm3(e3)
        e3 = self.relu3_3(e3)
        e3 = self.max3(e3)
        # print("e3 size is :", e3.size())
        
        e4 = self.conv4_1(e3)
        e4 = self.relu4_1(e4)
        e4 = self.conv4_2(e4)
        e4 = self.relu4_2(e4)
        e4 = self.conv4_3(e4)
        e4 = self.norm4(e4)
        e4 = self.relu4_3(e4)
        e4 = self.max4(e4)
        # print("e4 size is :", e4.size())
        
        e5 = self.conv5_1(e4)
        e5 = self.relu5_1(e5)
        e5 = self.conv5_2(e5)
        e5 = self.relu5_2(e5)
        e5 = self.conv5_3(e5)
        e5 = self.norm5(e5)
        e5 = self.relu5_3(e5)
        e5 = self.max5(e5)
        # print("e5 size is :", e5.size())


        e5_upsp = self.conv6_1(e5)
        e5_upsp = self.norm6(e5_upsp)
        e5_upsp = self.relu6_1(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())

        e5_upsp = self.conv6_2(e5_upsp)
        e5_upsp = self.norm6_2(e5_upsp)
        e5_upsp = self.relu6_2(e5_upsp)
        # e5_upsp = self.upsp(e5)
        # print("e5_upsp size is :", e5_upsp.size())


        # d1_h = self.conv6_1_h(e5_h)
        d1_h = self.conv6_1_h(e5_upsp)
        # d1_h = self.conv6_1_h(e5_h_concat)
        # d1_h = self.conv6_1_h(e5_h_concat_refine)
        d1_h = self.norm6_h(d1_h)
        d1_h = self.relu6_1_h(d1_h)
        # print("d1_h size is :", d1_h.size())

        d1_h = self.conv6_1_h_2(d1_h)
        d1_h = self.norm6_1_h_2(d1_h)
        d1_h = self.relu6_1_h_2(d1_h)
        # e5_upsp = self.upsp(e5)
        # print("d1_h size is :", d1_h.size())

        '''
        d2_h = self.conv7_1_h(d1_h)
        # d2 = self.norm7(d2)
        d2_h = self.relu7_1_h(d2_h)
        print("d2_h size is :", d2_h.size())
        '''
        
        s_attention_map = self.s_attention_conv(d1_h) # spatial attention mechnism
        s_attention_map = self.relu_ird_1(s_attention_map) 
        s_attention_map = self.softmax_1(s_attention_map)
        s_attention_map = s_attention_map.repeat(1, 128, 1, 1) # extend from 1X1XWXH to 1XNXWXH
        d2_h = d1_h.mul(s_attention_map)

        d2_h_sequence_unit = torch.unsqueeze(d2_h, dim=1)
        # print("shape of d2_h_sequence_unit:", d2_h_sequence_unit.size())


        # e5_h_sequence_1 = []
        d2_h_sequence_2 = d2_h_sequence_unit

        for n_sequence in range(4):
            # e5_h_sequence_1.append(e5_h_sequence_unit)
            d2_h_sequence_2 = torch.cat([d2_h_sequence_2, d2_h_sequence_unit], dim=1)


        # print("shape of e5_h_sequence_1:", e5_h_sequence_1.size())
        # print("shape of d2_h_sequence_2:", d2_h_sequence_2.size())

        layer_output_list, last_state_list = self.ConvLSTM_layer_1(d2_h_sequence_2)
        # print("layer_output_list is:", layer_output_list.size()) # list has no shape
        d2_h_refine = layer_output_list[0]
        # print("shape of d2_h_refine:", d2_h_refine.size())
        d2_h_refine = d2_h_refine[:, -1, :, :, :]
        # print("shape of d2_h_refine:", d2_h_refine.size())
               
        # d3_h = self.conv8_1_h(d2_h)
        d3_h = self.conv8_1_h(d2_h_refine)
        # d2 = self.norm7(d2)
        d3_h = self.relu8_1_h(d3_h)
        # print("d3_h size is :", d3_h.size())
        
        d4_h = self.conv9_1_h(d3_h)
        d4_h = self.tanh9_1_h(d4_h)
        # print("d4_h size is :", d4_h.size())

        d4_out = torch.nn.functional.upsample_bilinear(d4_h, size=[120, 160])
        d4_upsp = self.upsp_output(d4_out)
        # d4_h_upsp = self.upsp_output(d4_h)
        # print("d4_h_upsp size is :", d4_h_upsp.size())
        # d4_h_upsp = torch.nn.functional.upsample_bilinear(d4_h_upsp, size=[480, 640])
        
        return d4_upsp, e5_upsp, d2_h
        # return d4_out

###################################################################################
######## basic modules
class encoderconv_2(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(encoderconv_2, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1), # output_shape = (image_shape-filter_shape+2*padding)/stride + 1, image_shape is odd number
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.LeakyReLU(0.2), 
        )
    def forward(self, input):
        return self.conv(input)

class decoderconv_2(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(decoderconv_2, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.conv = nn.Sequential( 
            nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1, output_padding=0), 
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.ReLU(True), 
        )
    def forward(self, input):
        return self.conv(input)

class decoderconv_3(nn.Module): # basic conv of encoder
    def __init__(self, in_ch, ou_ch):
        super(decoderconv_3, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.conv = nn.Sequential( 
            nn.ConvTranspose2d(in_ch, ou_ch, kernel_size=4, stride=2, padding=1, output_padding=(1,0)), 
            # nn.BatchNorm2d(ou_ch),
            norm_layer(ou_ch),
            nn.ReLU(True), 
        )
    def forward(self, input):
        return self.conv(input)

class dimredconv(nn.Module): # dim-reduction layer, i.e. bottleneck layer
    def __init__(self, in_ch, ou_ch):
        super(dimredconv, self).__init__()

        self.conv = nn.Sequential( 
            nn.Conv2d(in_ch, ou_ch, kernel_size=3, stride=1, padding=1), # output_shape = (image_shape-filter_shape+2*padding)/stride + 1, image_shape is odd number
            # nn.BatchNorm2d(ou_nc),
            nn.Tanh(), 
        )
    def forward(self, input):
        return self.conv(input)

class ResnetBlock(nn.Module): # Define a resnet block
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        # return out, out, out
        return out

class ResnetBlock_dilated(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_dilated, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        '''
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        '''
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=2),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        '''
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        '''
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=2),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Spatial_Channel_Gate_Layer(nn.Module):
    def __init__(self, dim_in, dim_redu):
        super(Spatial_Channel_Gate_Layer, self).__init__()
        self.onemulone_ird_1_A = nn.Conv2d(dim_in, dim_redu, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.onemulone_ird_1_B = nn.Conv2d(dim_in, dim_redu, kernel_size=1, stride=1, padding=0) # 1X1 conv, keep the resolution of feature map unchanged, but reduce the channel
        self.dcn_ird_1_B = ModulatedDeformConvPack(dim_redu, dim_redu, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.onemulone_ird_1_C = nn.Conv2d(dim_in, dim_redu, kernel_size=1, stride=1, padding=0)
        self.dcn_ird_1_C_1 = ModulatedDeformConvPack(dim_redu, dim_redu, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.dcn_ird_1_C_2 = ModulatedDeformConvPack(dim_redu, dim_redu, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        self.s_attention_conv = nn.Conv2d(3*dim_redu, 1, kernel_size=1, stride=1, padding=0)
        self.norm_ird_1 = nn.InstanceNorm2d(256, affine=False)
        self.relu_ird_1 = nn.ReLU(inplace=True)
        self.softmax_1 = nn.Softmax()
        self.expanddim = dim_in

        self.channel_gate = SELayer(dim_in, reduction=8)

    def forward(self, inputx):

        s_gate_A = self.onemulone_ird_1_A(inputx)
        s_gate_B = self.onemulone_ird_1_B(inputx)
        s_gate_B = self.dcn_ird_1_B(s_gate_B)
        s_gate_C = self.onemulone_ird_1_C(inputx)
        s_gate_C = self.dcn_ird_1_C_1(s_gate_C)
        s_gate_C = self.dcn_ird_1_C_2(s_gate_C)
        s_gate_concat = torch.cat([s_gate_A, s_gate_B, s_gate_C], dim=1)
        s_attention_map = self.s_attention_conv(s_gate_concat)
        # s_attention_map = self.norm_ird_1(s_attention_map)
        s_attention_map = self.relu_ird_1(s_attention_map) 
        s_attention_map = self.softmax_1(s_attention_map)
        s_attention_map = s_attention_map.repeat(1, self.expanddim, 1, 1) # extend from 1X1XWXH to 1XNXWXH
       
        input_channel_gate = self.channel_gate(inputx) # after channel-attention-gate
        outx = input_channel_gate.mul(s_attention_map) # after spatial-attention-gate

        return outx

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MyLSTMCell(nn.Module): # We should not use class name as "ConvLSTMCell", otherwise there will be error: has no weight

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(MyLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        
        self.conv_lstm = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        '''
        self.conv_lstm = ModulatedDeformConvPack(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=1,
                              deformable_groups=1,
                              no_bias=True)
        # ModulatedDeformConvPack(128, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2, no_bias=True).cuda()
        '''
    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv_lstm(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        # return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
          #      Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return ((torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                (torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class MyLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(MyLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(MyLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
            # layer_output_list = layer_output_list
            # last_state_list   = last_state_list

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

