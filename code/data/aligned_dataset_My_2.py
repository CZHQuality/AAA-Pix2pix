# We change the file path of input dataset (when using distorted dataset, and adversarial examples)
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        # dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A) # opt.phase can control the train set or test set
        # self.A_paths = sorted(make_dataset(self.dir_A))
        
        # dir_A = 'maps/train/' if self.opt.label_nc == 0 else '_label'
        # dir_A = 'images/train/' if self.opt.label_nc == 0 else '_label' # in our task, sourece domain A is the Image
        # dir_A = 'image/img/ContrastChange_2/' if self.opt.label_nc == 0 else '_label' # in our task, source domain A is the Image
        dir_A = 'ScaleDatabase/Compression_2/' if self.opt.label_nc == 0 else '_label' # in our task, source domain A is the Image
        self.dir_A = os.path.join(opt.dataroot, dir_A)
        # self.dir_A = os.path.join(dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        # if opt.isTrain or opt.use_encoded_image:
          #  dir_B = '_B' if self.opt.label_nc == 0 else '_img'
          #  self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  # opt.phase can control the train set or test set 
          #  self.B_paths = sorted(make_dataset(self.dir_B))
        
        if opt.isTrain or opt.use_encoded_image:
            # dir_B = 'images/train/' if self.opt.label_nc == 0 else '_img'
            # dir_B = 'maps/train/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            dir_B = 'map/ContrastChange_2/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # self.dir_B = os.path.join(dir_B)
            self.dir_B = os.path.join(opt.dataroot, dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

            # dir_C = 'fixations_img/train/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            # self.dir_C = os.path.join(opt.dataroot, dir_C)  
            dir_C = 'fixation_img/ContrastChange_2/' if self.opt.label_nc == 0 else '_img' # in our task, sourece domain B is the Saliency Map
            self.dir_C = os.path.join(opt.dataroot, dir_C)  
            self.C_paths = sorted(make_dataset(self.dir_C))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path) 
        A = A.resize((640, 480),Image.ANTIALIAS)       
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        # B_tensor = inst_tensor = feat_tensor = 0
        B_tensor = C_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            B = B.resize((640, 480),Image.ANTIALIAS)
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

            C_path = self.C_paths[index]   
            C = Image.open(C_path).convert('RGB')
            C = C.resize((640, 480),Image.ANTIALIAS)
            transform_C = get_transform(self.opt, params)      
            C_tensor = transform_C(C)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        # input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,  
          #            'feat': feat_tensor, 'path': A_path}
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,  'fixpts': C_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'