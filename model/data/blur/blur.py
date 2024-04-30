##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from skimage.draw import disk
import torch.nn.functional as F
import torch
import cv2
import math
import os
from pathlib import Path
import sys
import torchvision.transforms as transforms
import random
from tqdm import tqdm

class Blur(object):
    def __init__(self, size=21, device="cuda", range_deterioration_ratio=(0, 1), range_deterioration_ratio2=None):
        self.size = size
        self.device = device
        self.range_deterioration_ratio = self.get_range_deterioration_ratio(range_deterioration_ratio)
        if range_deterioration_ratio2 != None:
            self.range_deterioration_ratio2 = self.get_range_deterioration_ratio(range_deterioration_ratio2)
        else:
            self.range_deterioration_ratio2 = None

    def get_range_deterioration_ratio(self, range_deterioration_ratio):
        if type(range_deterioration_ratio) in (int, float):
            range_deterioration_ratio = (range_deterioration_ratio, range_deterioration_ratio)
        else:
            assert type(range_deterioration_ratio) in (list, tuple) and len(range_deterioration_ratio) == 2, "range_deterioration_ratio is only (int or float or Tuple[float, int])"

        return range_deterioration_ratio

    def get_deterioration(self):
        b, a = self.range_deterioration_ratio
        deterioration = round(self.size * ((b - a) * np.random.rand() + a))
        if deterioration < 1: deterioration = 1
        return deterioration


class MotionBlur(Blur):
    def __init__(self, size=21, device="cuda", range_deterioration_ratio=(0, 1)):
        super().__init__(size, device, range_deterioration_ratio)

    def make(self):
        kernel_motion_blur = np.zeros((self.size, self.size))
        eps_arr = np.random.rand(1)
        deg = 180 * eps_arr[0]
        kernel_zeros = np.zeros((self.size, self.size))
        
        # len_lines = np.array([i for i in range(1, self.size+1, 2)])
        # len_line = np.random.choice(len_lines, 1).item()
        len_line = self.even2odd(self.get_deterioration())
        # print(len_line)
        # mask_mat = np.ones(len_line)

        if 0 <= deg < 45:
            kernel_motion_blur[int((self.size-1)/2), :] = np.ones(self.size) 
        if 45 <= deg < 90:
            kernel_motion_blur = np.eye(self.size)[::-1] 
            deg -= 45
        if 90 <= deg < 135:
            kernel_motion_blur[:, int((self.size-1)/2)] = np.ones(self.size) 
            deg -= 90
        if 135 <= deg < 180:
            kernel_motion_blur = np.eye(self.size)   
            deg -= 135

        h, w = kernel_motion_blur.shape
        mat = cv2.getRotationMatrix2D((int((w-1)/2), int((h-1)/2)), deg, 1.0)
        kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, mat, (w, h), flags=cv2.INTER_LINEAR)

        margin = round((self.size - len_line) / 2)
        kernel_zeros[margin:margin+len_line, margin:margin+len_line] = kernel_motion_blur[margin:margin+len_line, margin:margin+len_line]
        kernel_motion_blur = kernel_zeros
        
        kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur)
        
        return torch.FloatTensor(kernel_motion_blur).to(self.device)

    def even2odd(self, num):
        if num % 2 == 0:
            if np.random.randint(2):
                return num -1
            else:
                return num + 1
        else:
            return num
    

class DiskBlur(Blur):
    def __init__(self, size=21, device="cuda", range_deterioration_ratio=(0, 1)):
        super().__init__(size, device, range_deterioration_ratio)

    def make(self):
        # ratio_R = torch.rand(1).item()
        kernel = np.zeros((self.size, self.size), dtype=np.float32)
        circleCenterCoord = int(self.size / 2)
        # circleRadius = int(circleCenterCoord * ratio_R) + 0.5 # +1
        circleRadius = self.get_deterioration() / 2 + 0.5

        rr, cc = disk((circleCenterCoord, circleCenterCoord), circleRadius)
        # print(circleCenterCoord, circleRadius, rr, cc)
        kernel[rr,cc]=1

#         if(dim == 3 or dim == 5):
#             kernel = Adjust(kernel, dim)

        kernel = kernel / np.sum(kernel)
        return torch.FloatTensor(kernel).to(self.device)


class GaussianBlur(Blur):
    def __init__(self, size=21, range_theta=(0, 180), isotropic=True, 
                device="cuda", range_deterioration_ratio=(0, 4), range_deterioration_ratio2=None):
        super().__init__(size, device, range_deterioration_ratio, range_deterioration_ratio2)
        self.range_theta = range_theta
        self.isotropic = isotropic

    def make(self):
        theta = ((self.range_theta[1] - self.range_theta[0]) * torch.rand(1).item() + self.range_theta[0]) * np.pi / 180
        
#         kernel_radius = self.kernel_size // 2
        circleCenterCoord = int(self.size / 2)
        kernel_radius = int(circleCenterCoord)
        kernel_range = np.linspace(-kernel_radius, kernel_radius, self.size).reshape((1, -1))
        
        horizontal_range = np.tile(kernel_range, (self.size, 1))
        vertical_range = np.tile(kernel_range.T, (1, self.size))
        
        # sigma = [(i[1] - i[0]) * torch.rand(1).item() + i[0] for i in self.range_sigma]
        
        sigma = self.get_deterioration()

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        cos_theta_2 = cos_theta ** 2
        sin_theta_2 = sin_theta ** 2

        sigma_x_2 = 2.0 * (sigma[0] ** 2)
        if self.isotropic:
            sigma_y_2 = sigma_x_2
        else:
            sigma_y_2 = 2.0 * (sigma[1] ** 2)
#         print(sigma_x_2, sigma_y_2)
#         print(horizontal_range.shape, vertical_range.shape)

        a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
        b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
        c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

        gaussian = lambda x,y: np.exp(- ( a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2)))
        
        kernel = gaussian(horizontal_range, vertical_range)
        kernel = kernel / kernel.sum()

#         kernel = kernel / torh.sum(kernel)

        return torch.FloatTensor(kernel).to(self.device)

    def get_deterioration(self):
        deterioration = []
        a, b = self.range_deterioration_ratio
        deterioration.append((b - a) * np.random.rand() + a)
        if self.range_deterioration_ratio2 == None:
            deterioration.append((b - a) * np.random.rand() + a)
        else:
            a, b = self.range_deterioration_ratio2
            deterioration.append((b - a) * np.random.rand() + a)            
        return deterioration


def conv_kernel2d(img, kernel, device="cuda", add_minibatch=True):
    img = img.to(device)
    kernel = kernel.to(device)

    size_img = np.array(img.shape)
    size_kernel = kernel.shape[-1]
    # print(kernel.shape)
    if add_minibatch:
        img = img.view(1, *size_img)
        kernel = kernel.view(1, size_kernel, size_kernel).repeat(size_img[0], 1, 1)
        kernel = kernel.view(size_img[0], 1, size_kernel, size_kernel)
        groups = size_img[0]
    else:
        groups = size_img[1]

    padding = int((size_kernel - 1) / 2)
    com_img = F.conv2d(img, kernel, stride=1, padding=padding, groups=groups).squeeze()

    return com_img

def kernel_compound(kernel_a, kernel_b, device="cuda"):
    com_kernel = conv_kernel2d(kernel_a.unsqueeze(0), kernel_b)
    # print('test')
    return com_kernel / torch.sum(com_kernel)

def set_blur(size=21, device="cuda", mode="all_rand", range_deterioration_ratio=(0.1, 1.0), range_gaus_deterioration_ratio=(0.2, 4), 
            range_gaus_deterioration_ratio2=None, isotropic=True):
    """
    -input-
    mode:{"all_rand", "<ker1>-<ker2>", "<ker>"}
    
    """
    # print(mode)
    kernels = {"motion":MotionBlur(size, device="cuda", 
                range_deterioration_ratio=range_deterioration_ratio),
                "gaus":GaussianBlur(size, device="cuda",
                    range_deterioration_ratio=range_gaus_deterioration_ratio, 
                    range_deterioration_ratio2=range_gaus_deterioration_ratio2, 
                    isotropic=isotropic),
                "disk":DiskBlur(size, device="cuda", 
                range_deterioration_ratio=range_deterioration_ratio)
                }
    if mode == "all_rand":
        if np.random.randint(2):
            misc_kernel = kernels["gaus"].make()
        else:
            misc_kernel = kernels["disk"].make()

        return kernel_compound(kernels["motion"].make(), misc_kernel)

    elif "-" in mode:
        ker1, ker2 = mode.split("-")
        return kernel_compound(kernels[ker1].make(), kernels[ker2].make())

    else:
        # print(mode)
        return kernels[mode].make()


def make_blur(fname, output_dir):
    mode = "gaus" # "all_rand"
    rdr = [0.5, 1.0]
    a = 4
    b = 4
    rgdr = (0.2, 4.0) # 0.2, 1.3, 2.6, 4.0
    rgdr2 = (0.2, 4.0) # 0.2, 1.3, 2.6, 4.0
    kernel_size = 21
    scale_factor = 4

    blur_kernel = set_blur(size=kernel_size, mode=mode,
                        range_deterioration_ratio=rdr,
                        range_gaus_deterioration_ratio=rgdr,
                        range_gaus_deterioration_ratio2=rgdr2,
                        isotropic=False,
                        device="cpu"
                        )

    save_img(blur_kernel, 'gray', fname, output_dir)

def save_img(img, mode, fname, output_dir):
    # fname = fname.replace('.jpg', f'_{add_fname}.jpg')
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    fpath = os.path.join(output_dir, fname)
    print(torch.max(img))
    img = img / torch.max(img)
    # img = torch.clamp(img * 50, 0, 1)
    img = transforms.ToPILImage(mode='L')(img)  
    print(fpath)
    img.save(fpath)
    # 



if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    output_dir = sys.argv[1]
    num = int(sys.argv[2])
    for i in range(num):
        make_blur(f"{i}.png", output_dir)
