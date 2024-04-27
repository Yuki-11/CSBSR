##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
## This code refers to the following codes
## [1] https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py#L4 (MIT License)
## [2] https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py (MIT License)
## [3] https://github.com/PatRyg99/HausdorffLoss (No License)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import math
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import glob
from tqdm import tqdm

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt


class BaseMetrix:
    def __init__(self, input_order='BHWC', test_y_channel=True, standardized = True):
        if input_order not in ['BHWC', 'BCHW']:
            raise ValueError(
                f'Wrong input_order {input_order}. Supported input_orders are '
                '"BHWC" and "BCHW"')
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.standardized = standardized

    def pre_process(self, img1, img2, crop_border):
        assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
        
        if self.standardized:
            img1, img2 = list(map(lambda x: np.round(255 * x).astype(np.int8), [img1, img2]))
        img1 = reorder_image(img1, input_order=self.input_order).astype(np.float32)
        img2 = reorder_image(img2, input_order=self.input_order).astype(np.float32)

        if crop_border != 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if self.test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        return img1, img2


class IoU:
    """Intersection over Union
    output and target have range [0, 1]"""

    def __init__(self, th=0.5):
        self.name = "IoU"
        self.th = 0.5

    def __call__(self, output, target):
        smooth = 1e-5

        if torch.is_tensor(output):
            output = output.data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        output = output > self.th
        target = target > self.th
        intersection = (output & target).sum(axis=(2,3))
        union = (output | target).sum(axis=(2,3))

        return (intersection + smooth) / (union + smooth)


# https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py#L4

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2, [1,2,3]) # batch dim retained
        # print("mseeeeeeee", mse)  
        return (10 * torch.log10(1 / mse.to('cpu')).detach().numpy().copy())
        # return (20 * torch.log10(255.0 / torch.sqrt(mse)).to('cpu')).detach().numpy().copy()


class PSNR_old(BaseMetrix):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """
    def __init__(self, input_order='BHWC', test_y_channel=True, standardized=True):
        super().__init__(input_order, test_y_channel, standardized)
        self.name = "PSNR"

    def __call__(self, img1, img2, crop_border=0):
        img1, img2 = self.pre_process(img1.to("cpu").detach().numpy().copy(),
                                    img2.to("cpu").detach().numpy().copy(),
                                    crop_border)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))


# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # print("ssimmmmm", ssim_map.mean(1).mean(1).mean(1))
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average).to('cpu').detach().numpy().copy()

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


# https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_metric.py
class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray):

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.max(distances[indexes])

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = pred.byte()
        target = target.byte()
        result_hd = np.empty((pred.shape[0], 2))

        for batch_idx in range(pred.shape[0]):
            result_hd[batch_idx, 0] = self.hd_distance(pred[batch_idx].cpu().numpy(), target[batch_idx].cpu().numpy())
            result_hd[batch_idx, 1] = self.hd_distance(target[batch_idx].cpu().numpy(), pred[batch_idx].cpu().numpy())

        result_hd = np.max(result_hd, axis=1)    
        # print(result_hd)

        return result_hd  # np.max(right_hd, left_hd)



def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.
    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].
    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def _convert_input_type_range(img):
    """Convert the type and range of the input image.
    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img

def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    # img = _convert_input_type_range(img)
    if y_only:
        # out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
        weight = np.array([65.481, 128.553, 24.966])
        out_img = np.einsum('bhwc,c->bhw', img, weight) + 16.0
        out_img = out_img.reshape(*out_img.shape, 1) # bhwc
        # print(out_img.shape)
    else:
        weight = np.array([[65.481, -37.797, 112.0], 
                           [128.553, -74.203, -93.786],
                           [24.966, 112.0, -18.214]])
        bias = np.array([16, 128, 128]).reshape(1, 1, 1, 3)
        out_img = np.einsum(img, weight, 'bhwc,cj->bhwj') + bias
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def reorder_image(img, input_order='BHWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (b, h, w), return (b, h, w, 1);
    If the input_order is (b, c, h, w), return (b, h, w, c);
    If the input_order is (b, h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'BHWC' or 'BCHW'.
            If the input image shape is (b, h, w), input_order will not have
            effects. Default: 'BHWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['BHWC', 'BCHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'BHWC' and 'BCHW'")
    if len(img.shape) != 4:
        additional_dim = [1 for _ in range(4 - len(img.shape))]
        img = img.reshape(*additional_dim, *img.shape)
    if input_order == 'BCHW':
        img = img.transpose(0, 2, 3, 1)
    
    return img

def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 4 and img.shape[3] == 3:
        img = rgb2ycbcr(img, y_only=True)
        # img = img[..., None]
    return img * 255.


def main():
    test_path = sys.argv[1]
    gt_path = sys.argv[2]

    psnr = PSNR(standardized=False)
    test_files = glob.glob(test_path + '*')
    gt_files = glob.glob(gt_path + '*')
    sum_psnr = 0
    sum_ssim = 0
    length = len(test_files)
    for (test_file, gt_file) in tqdm(zip(test_files, gt_files)):
        # assert img1.shape == img2.shape
        test_image = np.array(Image.open(test_file))
        gt_image = np.array(Image.open(gt_file))
        # psnr = calculate_psnr(test_image, gt_image, 4)
        # ssim = calculate_ssim(test_image, gt_image, 4)
        #print(psnr)
        sum_psnr += psnr(test_image, gt_image)
        # sum_ssim += ssim
    print(test_path)
    print("averaged psnr {}".format(sum_psnr/length))
    # print("averaged ssim {}".format(sum_ssim/length))


if __name__ == '__main__':
    main()