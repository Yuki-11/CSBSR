##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import numpy as np
import torchvision.transforms as transforms
import os
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


class MetaWeight(nn.Module):
    def __init__(self, cfg, amp, bias):
        from ..data.blur import blur
        super().__init__()
        self.offset = bias
        var = cfg.SOLVER.ORIENTED_WEIGHT_GAUS
        range_deterioration_ratio = (var, var)
        self.gaus = blur.GaussianBlur(range_deterioration_ratio=range_deterioration_ratio, size=cfg.BLUR.KERNEL_SIZE).make()
        self.gaus /= torch.max(self.gaus) # adjust max value to 1
        self.gaus = self.gaus.expand(1, 1, *self.gaus.shape)
        self.amp = amp


class CrackOrientedWeight(MetaWeight):
    def __init__(self, cfg, amp, bias):
        super().__init__(cfg, amp, bias)

    def __call__(self, gt):
        # print(gt.shape, self.gaus.shape)
        conv_map = blur.conv_kernel2d(gt, self.gaus, add_minibatch=False)
        weight = self.amp * conv_map + self.offset
        return weight.view(weight.shape[0], 1, *weight.shape[1:])

class CrackOrientedExpWeight(nn.Module):
    def __init__(self, cfg, amp, bias, _lambda=2):
        super().__init__()
        self.offset = bias
        self.amp = amp
        self._lambda = _lambda

    def __call__(self, gt):
        # print(gt.shape, self.gaus.shape)
        gt_sdm_npy = compute_sdm(gt.cpu().numpy(), gt.shape)
        gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(gt.device.index)
        weight = torch.exp(-self.amp * gt_sdm)
        return self._lambda * weight


class SegmentFailerOrientedWeight(MetaWeight):
    def __init__(self, cfg, amp, bias):
        super().__init__(cfg, amp, bias)

    def __call__(self, pred, gt):
        _pred, gt = pred.detach().to('cuda'), gt.to('cuda')
        # print(self.gaus.device, _pred.device, gt.device)
        conv_map = blur.conv_kernel2d(torch.abs(_pred - gt), self.gaus, add_minibatch=False)
        weight = self.amp * conv_map + self.offset
        return weight.view(weight.shape[0], 1, *weight.shape[1:])


class SegmentFailerOrientedExpWeight(nn.Module):
    def __init__(self, cfg, amp, bias, _lambda=1.0):
        super().__init__()
        self.offset = bias
        self.amp = amp
        self._lambda = _lambda

    def __call__(self, pred, gt):
        _pred, gt = pred.detach().to('cuda'), gt.to('cuda')
        weight = torch.exp(self.amp * torch.abs(_pred - gt))
        return self._lambda * weight

# print(self.gaus)
# print(torch.max(self.gaus))
# kernel = transforms.ToPILImage(mode='L')(self.gaus / torch.max(self.gaus))
# fpath = os.path.join("output/gaus.png")
# kernel.save(fpath)


def compute_sdm(img_gt, out_shape, norm=False):
    """
    compute the (normalized) signed distance map of binary mask
    input: segmentation, shape = (batch_size, c, x, y)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    if norm:
        normalize sdm to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)
    out_shape = img_gt.shape

    sdm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            negdis = distance(negmask)
            if norm:
                negdis = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))
            sdm[b][0] = negdis

    return sdm


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from config import cfg
    coew = CrackOrientedExpWeight(cfg, 2, 0)
    segment_targets = torch.zeros([1, 1, 7, 7])
    print(segment_targets.shape)
    segment_targets[0, 0, 2:5, 2:5] = 1
    print(segment_targets)
    print("====")
    print(coew(segment_targets))


