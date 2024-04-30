##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from blur import set_blur, conv_kernel2d
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import os
from pathlib import Path
import sys
from PIL import Image
import torch
import random
import numpy as np
from tqdm import tqdm


mode = "gaus" # "all_rand"
rdr = [0.5, 1.0]
a = 0.2
b = 4.0
rgdr = (a, b) # 0.2, 1.3, 2.6, 4.0
rgdr2 = (a, b) # 0.2, 1.3, 2.6, 4.0
kernel_size = 21
scale_factor = 4

class FactorResize(object):
    def __init__(self, factor):
        self.factor = factor
        self.interpolation = InterpolationMode.BICUBIC

    def __call__(self, image):
        height, width = image.shape[-2:]
        transform = transforms.Resize((int(height/self.factor), int(width/self.factor)), self.interpolation)
        image = transform(image)

        return image


def make_test_blur(fname, dataset_dir, output_dir, mode, rdr, rgdr, kernel_size):
    fpath = os.path.join(dataset_dir, fname)
    hr_img = transforms.functional.to_tensor(Image.open(fpath))
    blur_kernel = set_blur(size=kernel_size, mode=mode,
                        range_deterioration_ratio=rdr,
                        range_gaus_deterioration_ratio=rgdr,
                        range_gaus_deterioration_ratio2=rgdr2,
                        isotropic=False,
                        device="cpu"
                        )

    fname = fname.replace('jpg', 'png')
    save_img(blur_kernel, 'gray', fname, output_dir + 'kernels/')

    hr_img_blur = conv_kernel2d(hr_img, blur_kernel)
    sr_transforms = FactorResize(scale_factor)
    lr_img_blur = torch.clamp(sr_transforms(hr_img_blur), min=0, max=1)
    # print(torch.max(lr_img_blur))
    save_img(hr_img_blur, 'RGB', fname, output_dir + 'hr_images/')
    save_img(lr_img_blur, 'RGB', fname, output_dir + 'lr_images/')

def save_img(img, mode, fname, output_dir):
    # fname = fname.replace('.jpg', f'_{add_fname}.jpg')
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    fpath = os.path.join(output_dir, fname)

    if mode == 'RGB':
        img = transforms.ToPILImage(mode=mode)(img)  

    else:
        img = img / torch.max(img)
        img = transforms.ToPILImage(mode='L')(img)  
        
    img.save(fpath)
    # 



if __name__ == '__main__':
    num_seed = 5
    torch.manual_seed(num_seed)
    random.seed(num_seed)
    np.random.seed(num_seed)
    dataset_dir = sys.argv[1]
    output_dir = sys.argv[2]

    fnames = [path.name for path in Path(dataset_dir).glob('*.png')]
    assert len(fnames) != 0
    for fname in tqdm(fnames):
        make_test_blur(fname, dataset_dir, output_dir, mode,
                        rdr, rgdr, kernel_size)
