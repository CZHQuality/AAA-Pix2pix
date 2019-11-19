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

# This part is the loss function
class CCLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(CCLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # calculating the mean value of tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number
        # print("map_gtd_mean is :", map_gtd_mean)

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 
        map_gtd_std = torch.std(map_gtd) # calculate the standard deviation
        map_gtd_std = map_gtd_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization
        map_gtd = (map_gtd - map_gtd_mean) / (map_gtd_std + self.epsilon) # normalization

        map_pred_mean = torch.mean(map_pred) # re-calculating the mean value of normalized tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_gtd_mean = torch.mean(map_gtd) # re-calculating the mean value of normalized tensor
        map_gtd_mean = map_gtd_mean.item() # change the tensor into a number

        CC_1 = torch.sum( (map_pred - map_pred_mean) * (map_gtd - map_gtd_mean) )
        CC_2 = torch.rsqrt(torch.sum(torch.pow(map_pred - map_pred_mean, 2))) * torch.rsqrt(torch.sum(torch.pow(map_gtd - map_gtd_mean, 2))) + self.epsilon
        CC = CC_1 * CC_2
        # print("CC loss is :", CC)
        CC = -CC # the bigger CC, the better



        # we put the L1 loss with CC together for avoiding building a new class
        # L1_loss =  torch.mean( torch.abs(map_pred - map_gtd) )
        # print("CC and L1 are :", CC, L1_loss)
        # CC = CC + L1_loss

        return CC


class KLLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(KLLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()
        
        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are :", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        # print("min2 and max2 are :", min2, max2)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (torch.sum(map_pred) + self.epsilon)# normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (torch.sum(map_gtd) + self.epsilon) # normalization step to make sure that the map_gtd sum to 1
        # print("map_pred is :", map_pred)
        # print("map_gtd is :", map_gtd)


        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        # print("KL 1 is :", KL)
        KL = map_gtd * KL
        # print("KL 2 is :", KL)
        KL = torch.sum(KL)
        # print("KL 3 is :", KL)
        # print("KL loss is :", KL)

        return KL

class NSSLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(NSSLoss, self).__init__()        
        # self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
        self.epsilon = 1e-8 # the parameter to make sure the denominator non-zero


    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth fixation map
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

        min1 = torch.min(map_pred)
        max1 = torch.max(map_pred)
        # print("min1 and max1 are (saliecny map):", min1, max1)
        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
      
        min2 = torch.min(map_gtd)
        max2 = torch.max(map_gtd)
        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon) # min-max normalization for keeping KL loss non-NAN
        
        map_gtd_id_1 = torch.gt(map_gtd, 0.5)
        map_gtd_id_0 = torch.lt(map_gtd, 0.5)
        map_gtd_id_00 = torch.eq(map_gtd, 0.5)
        map_gtd[map_gtd_id_1] = 1.0
        map_gtd[map_gtd_id_0] = 0.0
        map_gtd[map_gtd_id_00] = 0.0

        map_pred_mean = torch.mean(map_pred) # calculating the mean value of tensor
        map_pred_mean = map_pred_mean.item() # change the tensor into a number

        map_pred_std = torch.std(map_pred) # calculate the standard deviation
        map_pred_std = map_pred_std.item() # change the tensor into a number 

        map_pred = (map_pred - map_pred_mean) / (map_pred_std + self.epsilon) # normalization

        NSS = map_pred * map_gtd
        # print("early NSS is :", NSS)
        # NSS = NSS # should not add anythin, because there are a lot of 0.00XXX in map1_NSS due to float format
        # id = torch.nonzero(NSS)
        id = torch.gt(NSS, 0.1) # find out the id of NSS > 0.1
        bignss = NSS[id]
        # print(bignss)
        if(len(bignss) == 0): # NSS[id] is empty 
            id = torch.gt(NSS, -0.00000001) # decrease the threshold, because must set it as tensor not inter
            bignss = NSS[id]
        # NSS = torch.sum(NSS[id])
        # NSS = torch.mean(NSS)
        NSS = torch.mean(bignss)
        
        NSS = -NSS # the bigger NSS the better
        return NSS 
        # return 0 # if return, error : TypeError: mean(): argument 'input' (position 1) must be Tensor, not float

'''
class VGGLoss(nn.Module): # The traditional VGG loss
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = KLLoss(gpu_ids) # KL/CC is better than L1 loss
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
'''

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
# def _ssim(X, Y, win, data_range=1.0, size_average=True, full=False):
    """ Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    # C1 = pow()
    # print("K1 * data_range:", K1*data_range)
    # print("K1 ** data_range:", K1**2)
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
# def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=1.0, size_average=True, full=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
# def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=1.0, size_average=True, full=False, weights=None):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    # def __init__(self, gpu_ids, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
    def __init__(self, gpu_ids, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3):
        """ class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        # notice we use 1-(original SSIM score) as output
        return 1 - ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    # def __init__(self, gpu_ids, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
    def __init__(self, gpu_ids, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3, weights=None):
        """ class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        # notice we use 1-(original SSIM score) as output
        return 1 - ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights)