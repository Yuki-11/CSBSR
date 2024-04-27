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

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.environ['HOME'], 'CSBSR'))
# print(sys.path)
from model.utils.boundary_loss import BoundaryLoss

class BoundaryComboLoss(nn.Module):  
    def __init__(self, per_epoch, resume_iter=0, smooth=10 ** -8, reduction='none', pos_weight=[1, 1], loss_weight=[1, 1], alpha_min=0.01, decrease_ratio=1, out_map=False):
        super(BoundaryComboLoss, self).__init__()
        # per_epoch = int(per_epoch / 2)
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = per_epoch 
        self.iter = resume_iter % per_epoch

        print('xxx:', resume_iter, per_epoch, resume_iter // per_epoch)
        self.alpha =  1.0 - (resume_iter // per_epoch) * 0.01 * self.decrease_ratio
        self.alpha = self.alpha_min if self.alpha <= self.alpha_min else self.alpha
        print('alpha:', self.alpha)

        print(self.iter, '\n\n')
        
        self.wbce_dice_loss = BCE_DiceLoss(smooth=smooth, pos_weight=pos_weight, loss_weight=loss_weight, out_map=out_map).to('cuda') 
        self.bd_loss = BoundaryLoss(out_map=out_map).to('cuda')

    def forward(self, predict, target, iter_cnt=True):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        wbce_dice_loss = self.wbce_dice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)
        loss = self.alpha * wbce_dice_loss + (1 - self.alpha) * bd_loss
        # print(wbce_dice_loss, bd_loss)
        
        # if iter_cnt:
        #     if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
        #         self.alpha -= 0.01 * self.decrease_ratio
        #         self.iter = 1
        #     else: 
        #         self.iter += 1
        #     print(self.iter, iter_cnt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    def update_alpha(self):
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            self.iter = 1
        else: 
            self.iter += 1
        # print('test')
        # print(self.iter)


class GeneralizedBoundaryComboLoss(nn.Module):  
    def __init__(self, per_epoch, smooth=10 ** -8, reduction='none', pos_weight=[1, 1], loss_weight=[1, 1], alpha_min=0.01, decrease_ratio=1, resume_iter=0):
        super(GeneralizedBoundaryComboLoss, self).__init__()
        # per_epoch = int(per_epoch / 2)
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight
        print('xxx:', resume_iter, per_epoch, resume_iter // per_epoch)
        self.alpha =  1.0 - (resume_iter // per_epoch) * 0.01 * decrease_ratio
        print('alpha:', self.alpha)
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = per_epoch 
        self.iter = resume_iter % per_epoch + 1
        
        self.wbce_dice_loss = BCE_DiceLoss(smooth=smooth, pos_weight=pos_weight, loss_weight=loss_weight, gdice=True).to('cuda') 
        self.bd_loss = BoundaryLoss().to('cuda')

    def forward(self, predict, target, iter_cnt=True):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        wbce_dice_loss = self.wbce_dice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)
        loss = self.alpha*wbce_dice_loss + (1 - self.alpha) * bd_loss
        
        # if iter_cnt:
        #     if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
        #         self.alpha -= 0.01 * self.decrease_ratio
        #         self.iter = 1
        #     else: 
        #         self.iter += 1

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    def update_alpha(self):
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            self.iter = 1
        else: 
            self.iter += 1


class Boundary_GDiceLoss(nn.Module):  
    def __init__(self, per_epoch, resume_iter=0, smooth=10 ** -8, reduction='none', alpha_min=0.01, decrease_ratio=1):
        super(Boundary_GDiceLoss, self).__init__()
        # per_epoch = int(per_epoch / 2)
        self.smooth = smooth
        self.reduction = reduction
        print('xxx:', resume_iter, per_epoch, resume_iter // per_epoch)
        self.alpha =  1.0 - (resume_iter // per_epoch) * 0.01 * decrease_ratio
        print('alpha:', self.alpha)
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = int(per_epoch) 
        self.iter = resume_iter % per_epoch + 1
        
        self.gdice_loss = GDiceLoss().to('cuda') 
        self.bd_loss = BoundaryLoss().to('cuda')

    def forward(self, predict, target, iter_cnt=True):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        gdice_loss = self.gdice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)

        loss = self.alpha*gdice_loss + (1 - self.alpha) * bd_loss
        
        # if iter_cnt:
        #     if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
        #         self.alpha -= 0.01 * self.decrease_ratio
        #         self.iter = 1
        #     else: 
        #         self.iter += 1

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
    def update_alpha(self):
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            self.iter = 1
        else: 
            self.iter += 1

class WeightedBCELoss(nn.Module):
    def __init__(self, smooth=10 ** -8, reduction='mean', pos_weight=[1, 1]):
        super(WeightedBCELoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], f"predict & target batch size don't match. predict.shape={predict.shape}"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)

        loss =  - (self.pos_weight[0]*target*torch.log(predict+self.smooth) + self.pos_weight[1]*(1-target)*torch.log(1-predict+self.smooth))/sum(self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean(dim=(1,2,3))
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (torch.einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * torch.einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (torch.einsum("bcxy->bc", net_output) + torch.einsum("bcxy->bc", y_onehot))
        gdc: torch.Tensor = 1 - 2 * (torch.einsum("bc->b", intersection) + self.smooth) / (torch.einsum("bc->b", union) + self.smooth)
        # gdc = divided.mean()

        return gdc
    

# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction

    paper:
        Milletari, F., Navab, N., Ahmadi, S.A.: V-Net: Fully Convolutional Neural Networks for 
        Volumetric Medical Image Segmentation. In: 2016 Fourth International Conference on 3D 
        Vision (3DV). pp. 565–571. IEEE (oct 2016)
    """
    def __init__(self, smooth=1e-6, p=2, reduction='none', out_map=False):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.out_map = out_map

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        if predict.shape[1] != target.shape[1]:
            t_shape = target.shape
            _target = target.detach().expand(t_shape[0], predict.shape[1], *t_shape[2:])
        else:
            _target = target

        if self.out_map:
            num =  2 * torch.sum(torch.mul(predict, _target), dim=1) + self.smooth
            den = torch.sum(predict.pow(self.p) + _target.pow(self.p)) + self.smooth
            loss_map = 1 / _target.numel() - num / den
            return loss_map
        else:
            predict = predict.contiguous().view(predict.shape[0], -1)
            _target = _target.contiguous().view(_target.shape[0], -1)
            num =  2 * torch.sum(torch.mul(predict, _target), dim=1) + self.smooth
            den = torch.sum(predict.pow(self.p) + _target.pow(self.p), dim=1) + self.smooth
            loss = 1 - num / den
        

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index

#     def forward(self, predict, target):
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.softmax(predict, dim=1)

#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss

#         return total_loss/target.shape[1]

class BCE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='none', pos_weight=[1, 1], loss_weight = [1, 1], gdice=False, out_map=False):
        super(BCE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        if out_map:
            self.bce_loss = WeightedBCELoss(pos_weight=pos_weight, reduction='none').to('cuda')
        else:
            self.bce_loss = WeightedBCELoss(pos_weight=pos_weight).to('cuda')
        if gdice:
            self.dice_loss = GDiceLoss().to('cuda')
        else:
            self.dice_loss = BinaryDiceLoss(out_map=out_map).to('cuda')
        self.loss_weight = loss_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        # print(f'WCE:{self.bce_loss(predict, target)}\nDice:{self.dice_loss(predict, target)}')
        loss = (self.loss_weight[0] * self.bce_loss(predict, target) + self.loss_weight[1] * self.dice_loss(predict, target)) / sum(self.loss_weight)
        
        # print("loss", loss.mean())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



if __name__ == '__main__':
    output_map = False
    loss_bce = WeightedBCELoss(reduction='none')
    loss_dice = BinaryDiceLoss(reduction='none')
    t = torch.zeros([1, 1, 5, 5])
    print(t)
    t[:, :, 1:4, 1:4] = 1
    y = torch.zeros([1, 1, 5, 5]) + 0.5
    print(y)
    print(loss_bce(t, y))
    print('==============')
    print(loss_dice(t, y, output_map=output_map))
    print('==============')
    print(torch.sum(loss_dice(t, y, output_map=output_map)))