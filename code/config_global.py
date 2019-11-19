###################################################
# Global parameters of this Toolbox
# epsilons = [ .0001] # step size of gradient update
epsilons = [ .0002]
# epsilons = [ .005] # better for feature-space attack
iter_num = 51 # The total amount of iterations. Notice that, when iter_num==1, the iterative attack methods degrade into normal single-step attack methods.  
start_idx = 150 # the start idx of source images to be attacked
# end_idx = 160 # the end idx of source images to be attacked
end_idx = 151 # the end idx of source images to be attacked

# pretrained_model = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel/PretrainedModel/MyGazeGAN/latest_net_G.pth" # path of pretrained models
pretrained_model_GazeGAN_1 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_GazeGAN_1.pth" # path of pretrained models
pretrained_model_SALICON_2 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_SALICON_2.pth" # path of pretrained models
pretrained_model_Globalpix2pix = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_Globalpix2pix.pth" # path of pretrained models
pretrained_model_GazeGAN_2 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_GazeGAN_2.pth" # path of pretrained models
pretrained_model_DCN_LSTM_1 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_DCov_LSTM_1.pth" # path of pretrained models
pretrained_model_DCN_2 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_DConv_2_no_LSTM.pth" # path of pretrained models
pretrained_model_SAM_VGG_1 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_SAM_VGG_1.pth" # path of pretrained models
pretrained_model_SAM_VGG_2 = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_SAM_VGG_2.pth" # path of pretrained models
pretrained_model_SAM_ResNet = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_SAM_ResNet.pth" # path of pretrained models
pretrained_model_GazeGAN_CSC = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_GazeGAN_CSC.pth" # path of pretrained models
pretrained_model_SalGAN_BCE = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_SalGAN_BCE.pth" # path of pretrained models
pretrained_model_DCN_Inception = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_DCN_Inception.pth" # path of pretrained models
pretrained_model_DeepGaze_only_VGG = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_DeepGaze_only_VGG.pth" # path of pretrained models
pretrained_model_DCN_SAM_VGG = "/home/che-z/docker-czh/MyAdvExm/PretrainedModel_Ensemble/latest_net_G_DCN_SAM_VGG.pth" # path of pretrained models

path_root = '/home/che-z/docker-czh/datasets/SALICON/' # path of source images to be attacked. This is for SALICON dataset
path_root_A = path_root + 'images/val/' # source image
path_root_B = path_root + 'maps/val/' # dense saliency map of source image
path_root_C = path_root + 'fixations_img/val/' # discrete fixation map of source image

Used_attack_method_idx = 5 # 1, 2, ..., N. Each index corresponds to one attack method as shown in "Attack_methods_library.py". This is critical to update the 2nd momentum

non_target_attack = False
target_attack = False
universal_attack = True

feature_space_attack = False
image_space_attack = True
use_L1_perceptual = True
use_L2_perceptual = False
use_SSIM_perceptual = False

if(image_space_attack==True):
    root_sv_path = "/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/Ensemble_2/test_24/" # path to save the generated adversarial examples    
if(feature_space_attack==True and image_space_attack==False): # for hybrid, we still save into the above path
    root_sv_path = "/home/che-z/docker-czh/MyAdvExm/Visualization_blackbox/Feature_Single/test_5/"

Unit_Width = 640
Unit_Height = 480

# select the loss function by changing the weights
gpu_ids = '0'
if(image_space_attack==True):
    wei_L1 = 0.0 # 1.0
    wei_CC = 0.0 # 10.0
    wei_KL = 10.0 # 10.0
    wei_NSS = 0.0 # 0.0   NSS is not a good loss function here
    wei_Repre = 0.0 # 1.0  1.0e3 
if(feature_space_attack==True):
    wei_L1 = 0.0 # 1.0
    wei_CC = 0.0 # 10.0
    wei_KL = 0.0 # 10.0
    wei_NSS = 0.0 # 0.0   NSS is not a good loss function here
    wei_Repre = 10.0 # 1.0  1.0e3 

# wei_VGG = 0.0 # 
wei_percep = 1.0 * 1e-4 # for L1 perceptual loss, the perceptually constraint between original source image and the iteratively updated adv. example
wei_percep_MSE = 1.0 * 1e-2 # for L2 perceptual loss, the perceptually constraint between original source image and the iteratively updated adv. example
wei_percep_SSIM = 10.0 # for SSIM perceptual loss, the perceptually constraint between original source image and the iteratively updated adv. example
B_CONST_vec = [1.0 * 1e-4, 1.0 * 1e-2, 1.0 * 1e-2 * (1/2), 1.0 * 1e-2 * (1/4), 1.0 * 1e-2 * (1/8), 1.0 * 1e-2 * (1/16), 1.0 * 1e-2 * (1/32), 1.0 * 1e-2 * (1/64), 1.0 * 1e-2 * (1/128)]
B_CONST = B_CONST_vec[1] # initial weighting hyperparameter to balance the attack performance loss and perceptual constraint loss

Serial_Minibatch_Attack = True