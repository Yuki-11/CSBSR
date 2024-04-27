import os
import numpy as np
import cv2
from PIL import Image
from copy import copy
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from torch.utils.data import DataLoader, Dataset
from model.data.transforms.transforms import ToTensor
from model.data.blur.blur import set_blur, conv_kernel2d


class CrackDataSet(Dataset):
    def __init__(self, cfg, image_dir , seg_dir, transforms=None, sr_transforms=None):
        print("image_dir", image_dir)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.blur_flag = cfg.BLUR.FLAG
        self.blur_kernel_size = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.blur_isotropic = cfg.BLUR.ISOTROPIC

    def __getitem__(self, i):
        
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath)) #Image.open(fpath)
        spath = os.path.join(self.seg_dir, fname)

        seg_target = np.array(Image.open(spath))[:, :, np.newaxis] # HxWxC
        img, seg_target = self.img_transforms(img, seg_target)
        sr_target = copy(img)
        
        if self.blur_flag:
            blur_kernel = set_blur(self.blur_kernel_size, mode="gaus", isotropic=self.blur_isotropic).to("cpu")
            img = conv_kernel2d(img, blur_kernel).to("cpu")
            blur_kernel = blur_kernel.view(1, *blur_kernel.shape)
        else:
            blur_kernel = torch.zeros((1, self.blur_kernel_size, self.blur_kernel_size))
            center = self.blur_kernel_size // 2
            blur_kernel[0, center, center] = 1

        # print('dataset', blur_kernel.size())
        
        img = self.sr_transforms(img)

        return img, sr_target, seg_target, blur_kernel

    def __len__(self):
        # print(self.fnames)
        return len(self.fnames)


class CrackDataSetTest(Dataset):
    def __init__(self, cfg, image_dir , seg_dir, blur_dir, blur_name, batch_size, transforms=None, sr_transforms=None):
        from model.data.samplers.patch_sampler import SplitPatch
        self.gt_image_dir = image_dir
        self.gt_seg_dir = seg_dir
        self.gt_blur_dir = os.path.join(blur_dir, blur_name, 'kernels')
        self.input_image_dir = os.path.join(blur_dir, blur_name, 'lr_images')
        self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        patch_sizeh, patch_sizew = [int(i / self.scale_factor) for i in cfg.INPUT.IMAGE_SIZE]
        self.split_img_patch = SplitPatch(batch_size, 3, patch_sizeh, patch_sizew)
        self.seg_ch = cfg.MODEL.NUM_CLASSES

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.gt_image_dir, fname)
        sr_target = np.array(Image.open(fpath)) #Image.open(fpath)
        spath = os.path.join(self.gt_seg_dir, fname)
        seg_target = np.array(Image.open(spath))[:, :, np.newaxis] # HxWxC
        sr_target, seg_target = self.img_transforms(sr_target, seg_target)
        
        fname = fname.replace('jpg', 'png')
        kpath = os.path.join(self.gt_blur_dir, fname)
        blur_kernel = np.array(Image.open(kpath))[:, :, np.newaxis]
        blur_kernel, _ = self.img_transforms(blur_kernel, None)
        blur_kernel = blur_kernel / torch.sum(blur_kernel)
        # print(blur_kernel.shape)

        if self.scale_factor != 1:
            fpath = os.path.join(self.input_image_dir, fname)
            img = np.array(Image.open(fpath))
            img, _ = self.img_transforms(img, None)
        else:
            img = sr_target.detach()

            
        img, img_unfold_shape = self.split_img_patch(img) # img.shape = [patches, ch, H, W]
        img_unfold_shape[[5, 6]] = img_unfold_shape[[5, 6]] * self.scale_factor # Considering the effect of upsampling when combining patch images
        seg_unfold_shape = img_unfold_shape.copy()
        seg_unfold_shape[[1, 4]] = self.seg_ch # initialize segmentation channel 
        
        num_patch = img_unfold_shape[2] * img_unfold_shape[3]
        blur_kernel = blur_kernel.expand(num_patch, *blur_kernel.shape[1:])

        # if self.cfg.BLUR:
        #     # print("Adding blur...")
        #     # blur_fname = fname.replace('.jpg', '') + "_blured.jpg"
        #     blur_fpath = os.path.join(self.blur_dir, 'images', fname)
        #     img = np.array(Image.open(blur_fpath))
        #     img, _ = self.img_transforms(img, None)

        #     # blur_fname = fname.replace('.jpg', '') + "_kernel.jpg"
        #     blur_fpath = os.path.join(self.blur_dir, 'kernels', fname)
        #     blur_kernel = np.array(Image.open(blur_fpath))[:, :, np.newaxis] # HxWxC
        #     blur_kernel, _ = self.img_transforms(blur_kernel, None)
            
        # # else:
        # #     blur_kernel = torch.zeros((1, self.blur_kernel_size, self.blur_kernel_size))
        # #     center = self.blur_kernel_size // 2
        # #     blur_kernel[0, center, center] = 1
        # img = self.sr_transforms(img)
        # img, img_unfold_shape = self.split_img_patch(img)
        # seg_unfold_shape = img_unfold_shape.copy()
        # seg_unfold_shape[[1, 4]] = self.seg_ch # initialize segmentation channel 
        
    
        return img, sr_target, seg_target, blur_kernel, fname, img_unfold_shape, seg_unfold_shape

    def __len__(self):
        return len(self.fnames)
    

class TTICrackDataSetTest(Dataset):
    def __init__(self, cfg, image_dir, batch_size, transforms=None):
        from model.data.samplers.patch_sampler import SplitPatch
        self.image_dir = image_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.png')]
        self.img_transforms = transforms
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.split_img_patch = SplitPatch(batch_size, 3, *cfg.INPUT.IMAGE_SIZE)
        self.seg_ch = cfg.MODEL.NUM_CLASSES

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath)) #Image.open(fpath)
        img, _ = self.img_transforms(img, None)
        # print(img.shape)
        # print(blur_kernel.shape)
            
        img, img_unfold_shape = self.split_img_patch(img) # img.shape = [patches, ch, H, W]
        img_unfold_shape[[5, 6]] = img_unfold_shape[[5, 6]] * self.scale_factor # Considering the effect of upsampling when combining patch images
        # print(img_unfold_shape)
        seg_unfold_shape = img_unfold_shape.copy()
        seg_unfold_shape[[1, 4]] = self.seg_ch # initialize segmentation channel 

        return img, fname, img_unfold_shape, seg_unfold_shape

    def __len__(self):
        return len(self.fnames)



class SRPretrainDataSet(Dataset):
    def __init__(self, cfg, image_dir, transforms=None, sr_transforms=None):
        print("image_dir", image_dir)
        self.image_dir = image_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.png')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms
        self.blur_flag = cfg.BLUR.FLAG
        self.blur_kernel_size = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.blur_isotropic = cfg.BLUR.ISOTROPIC

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath))   # Image.open(fpath)

        img, _ = self.img_transforms(img, None)
        sr_target = copy(img)
        
        # range_gaus_deterioration_ratio=0.2
        if self.blur_flag:
            blur_kernel = set_blur(self.blur_kernel_size, mode="gaus", isotropic=self.blur_isotropic, ).to("cpu") # mode="gaus",
            # print(blur_kernel.shape)
            img = conv_kernel2d(img, blur_kernel).to("cpu")
            blur_kernel = blur_kernel.view(1, *blur_kernel.shape)

        else:
            blur_kernel = torch.zeros((1, self.blur_kernel_size, self.blur_kernel_size))
            center = self.blur_kernel_size // 2
            blur_kernel[0, center, center] = 1
        
        img = self.sr_transforms(img)
        # print('fin dataset')
        return img, sr_target, blur_kernel

    def __len__(self):
        return len(self.fnames)
