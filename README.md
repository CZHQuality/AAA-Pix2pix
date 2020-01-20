# SMBEA: Serial-Mini-Batch-Ensemble-Attack

**SMBEA** is a new **Black-Box** Adversarial Attack against various **Pixel-to-Pixel** Tasks, such as **Saliency Detection, Depth Estimation, Image Translation, Style Transfer, etc.** This code repository is an Open-Source Toolbox based on Pytorch Platform.

A **preliminary version** of this repository has been accepted by
**AAAI2020**: ‘‘***A New Ensemble Adversarial Attack Powered by Long-term Gradient Memories***’’


We provide 3 visualizations (GIF format) for your reference.

Each GIF contains two parts:

Part-I: In the begining still frames: 
the upper-left region is the original clean image, the bottom-left region is the ground-truth output of the clean image, the upper-right region is the guide image, the bottom-right region is the ground-truth output of the guide image

Part-II: In the following dynamic frames: 
the upper-left region is the crafted adversarial example, the upper-right region is the normalized perturbation (obtained by elemen-wise subtraction of clean image and adversarial example, and normalized by min-max normalization for better obvervation).
The bottom regions are the outputs of two black-box target models on the crafted adversarial example.
The timestamp denotes the iterations.

**Visualizations on LSUN'17 Dataset：**

![image](https://github.com/CZHQuality/AAA-Pix2pix/blob/master/Visualizations/1_Our_Attack_LSUN17.gif)

**Visualizations on Cityspaces Dataset：**

![image](https://github.com/CZHQuality/AAA-Pix2pix/blob/master/Visualizations/2_Our_Attack_Cityspaces.gif)

**Visualizations on Google Satellite Dataset：**

![image](https://github.com/CZHQuality/AAA-Pix2pix/blob/master/Visualizations/3_Our_Attack_Google.gif)



**Requirements**

1. Pytorch 3.5.2

2. NVIDIA GPU (at least 16GB memory for ensemble attacks!!)

3. You have to compile the Deformable Convolution Lib by yourself: **https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch** (Required)

4. If you want to train/design/enhance the victim models from scratch to defend different attacks in our literature, please visit **https://github.com/CZHQuality/Sal-CFS-GAN** for more details about model training. These two repositories support each other.



**Complete version (including feature-space ensemble, long-term gradient auto-update) of our code will be released after our journal version is accepted, thanks!!!**

