##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Kernelized Back-Projection Networks for Blind Super Resolution
## Tomoki Yoshida, Yuki Kondo, Takahiro Maeda, Kazutoshi Akita, Norimichi Ukita
## 
## This code is based on https://github.com/Yuki-11/KBPN, and is licensed Apache LICENSE.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.utils.boundary_loss import BoundaryLoss
from model.utils.oriented_weight import CrackOrientedWeight, SegmentFailerOrientedWeight, CrackOrientedExpWeight, SegmentFailerOrientedExpWeight

class KBPNLoss(nn.Module):  
    def __init__(self, cfg, sr_transforms, reduction="mean"):
        super(KBPNLoss, self).__init__()

        self.hr_loss = nn.L1Loss(reduction='none').to('cuda')
        self.lr_loss = nn.L1Loss(reduction='none').to('cuda')
        self.kernel_loss = nn.MSELoss(reduction='none').to('cuda')

        self.reduction = reduction
        self.dim_reduction = (1, 2, 3)
        self.scale_factor = cfg.MODEL.SCALE_FACTOR 
        
        self.get_pseudo_lr = Get_pseudo_lr(cfg, sr_transforms)
        self.weight = cfg.SOLVER.SR_LOSS_FUNC_SR_WEIGHT
        self.kernel_pretrain_iter = cfg.SOLVER.SR_KERNEL_MODULE_PRETRAIN_ITER
        self.only_kernel_loss = cfg.SOLVER.ONLY_KERNEL_LOSS_FOR_PRETRAIN
        self.weight_iter = cfg.SOLVER.ORIENTED_WEIGHT_ITER
        self.w_co_sr = CrackOrientedExpWeight(cfg, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_BIAS)
        # self.w_co_sr = CrackOrientedWeight(cfg, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_BIAS)
        self.w_sfo_sr = SegmentFailerOrientedExpWeight(cfg, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_BIAS)
        # self.w_sfo_sr = SegmentFailerOrientedWeight(cfg, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_BIAS)

    def forward(self, hr_pred, hr_target, lr_target, kernel_pred, gt_kernel, segment_preds, segment_targets, iter):
        # print(1, kernel_pred.size(), gt_kernel.size())
        hr_loss = self.hr_loss(hr_pred, hr_target)
        
        lr_pred, kernel_pred = self.get_pseudo_lr(hr_pred, kernel_pred)
        # print(2, kernel_pred.size(), gt_kernel.size())
        lr_loss = self.lr_loss(lr_pred, lr_target)
        kernel_loss = self.kernel_loss(kernel_pred, gt_kernel)
        if iter > self.weight_iter and self.weight_iter != -1:
            hr_loss, lr_loss = self.multiple_weight(hr_loss, lr_loss, segment_preds, segment_targets)
        # print(lr_loss.size(), kernel_loss.size())
        if self.only_kernel_loss and (self.kernel_pretrain_iter[0] <= iter < self.kernel_pretrain_iter[1]):
            loss = kernel_loss
        else:
            loss = self.weight[0] * hr_loss.mean(self.dim_reduction) + self.weight[1] * lr_loss.mean(self.dim_reduction) +\
                   self.weight[2] * kernel_loss.mean(self.dim_reduction)

        return loss, kernel_pred

    def multiple_weight(self, hr_loss, lr_loss, segment_preds, segment_targets):
        if self.w_co_sr.amp != 0:
            w_co = self.w_co_sr(segment_targets)
            # print('debug:', w_co.shape, hr_loss.shape)
            hr_loss = w_co * hr_loss
            lr_loss = F.interpolate(w_co, scale_factor=1/self.scale_factor, mode='bilinear') * lr_loss
        if self.w_sfo_sr.amp != 0:
            # print('sfow')
            w_co = self.w_sfo_sr(segment_preds, segment_targets)
            # print(self.w_co_sr(segment_targets).shape, sr_loss.shape)
            hr_loss = w_co * hr_loss
            lr_loss = F.interpolate(w_co, scale_factor=1/self.scale_factor, mode='bilinear') * lr_loss

        return hr_loss, lr_loss

class Get_pseudo_lr(nn.Module):
    def __init__(self, cfg, sr_transforms):
        super(Get_pseudo_lr, self).__init__()
        self.ksize = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.pad = (self.ksize-1) // 2
        self.interpolate = 'bicubic'
        self.down_scale = 1 / self.scale_factor
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.sr_transforms = sr_transforms

    def __call__(self, sr_t, kernel):
        kernel = self.GAP(kernel)
        #kernel = F.softmax(kernel, dim=1)
        kernel = kernel / ((kernel.sum(dim=1)).view(kernel.shape[0],1,1,1))
        weight = kernel.view(kernel.shape[0], 1, self.ksize, self.ksize)
        out_ch = sr_t.shape[1]
        for k in range(sr_t.shape[0]):
            tmp_weight = weight[k].expand(out_ch, 1, self.ksize, self.ksize) # .flip([2,3])
            
            tmp = F.conv2d(sr_t[k].view(1, sr_t.shape[1], sr_t.shape[2], sr_t.shape[3]), tmp_weight, padding=self.pad, groups=out_ch) # 
            tmp = self.sr_transforms(tmp)
            # tmp = F.interpolate(tmp, scale_factor=self.down_scale, mode=self.interpolate, recompute_scale_factor=True, align_corners=False)
            
            if k == 0:
                lrs = tmp
            else:
                lrs = torch.cat((lrs, tmp), dim=0)
        
        return lrs, weight