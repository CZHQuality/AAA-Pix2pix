# SMBEA: Serial-Mini-Batch-Ensemble-Attack against Pix2pix Tasks
**AAAI2020 paper**: **‘‘A New Ensemble Adversarial Attack Powered by Long-term Gradient Memories’’**


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

**Visualizations on Google Satellite Images：**

![image](https://github.com/CZHQuality/AAA-Pix2pix/blob/master/Visualizations/3_Our_Attack_Google.gif)



**Requirements**

Pytorch 3.5.2

NVIDIA GPU (at least 16GB memory for ensemble attacks!!)

You have to compile the Deformable Convolution Lib by your self: **https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch**



**Completed version of our code will be released after our journal version is accepted, thanks!!!**



**Citation:**

@InProceedings{SMBEA,

author={Z. Che and A. Borji and G. Zhai and S. Ling and J. Li and P. L. Callet},

booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},

title={A New Ensemble Adversarial Attack Powered by Long-term Gradient Memories},

year={2020},

}
