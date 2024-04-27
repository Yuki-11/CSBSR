##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from copy import copy

import torch
import torch.nn as nn

from .dbpn import *
from .unet import *
from .kbpn import KBPN
from model.modeling.pspnet_pytorch.pspnet import PSPNet, PSPNet_BlurSkip
from model.modeling.srcnet import SrcNetSR, SegNet
from model.modeling.crackformer import CrackFormer
from model.modeling.hrnet_ocr.nets.hrnet import HRNet_W48_OCR, HRNet_W48_ASPOCR, HRNet_W48_OCR_B
from model.modeling.DSRL.deeplab import DeepLab, DeepLabx4
from model.modeling.hrnet_ocr.tools.set_config import set_configer

from model.utils.loss_functions import BinaryDiceLoss, WeightedBCELoss, BoundaryComboLoss, Boundary_GDiceLoss, GeneralizedBoundaryComboLoss, BCE_DiceLoss
from model.utils.CrackFormerLoss.lossFunctions import cross_entropy_loss_RCF
from model.utils.sr_loss_functions import KBPNLoss
from model.modeling.DSRL.utils.fa_loss import FALoss
from model.utils.oriented_weight import CrackOrientedWeight, SegmentFailerOrientedWeight, CrackOrientedExpWeight, SegmentFailerOrientedExpWeight
from model.utils.misc import fix_model_state_dict, chop_forward
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


_MODEL = {
    2: Net_2,
    4: Net_4,
    6: Net_6,
    7: Net_7,
    8: Net_8,
    10: Net_10,
}


# ================================================= Abstract Class ===========================================================


class MetaSRModel(nn.Module):
    def __init__(self, cfg, mro_name):
        next_class = mro_name[mro_name.index('MetaSRModel') + 1]
        if next_class != nn.Module.__name__:
            super().__init__(cfg, mro_name)
        else:
            super().__init__()
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.sr_model = self.set_sr_model(cfg)
        self.norm_method = cfg.SOLVER.NORM_SR_OUTPUT
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD

    def forward_sr(self, iter, x, sr_targets=None, kernel_targets=None):
        x, sr_targets, kernel_targets = self.mount_cuda([x, sr_targets, kernel_targets])
        # Super Resolution task
        if self.sr_model is None:
            sr_preds = sr_targets.clone()
            kernel_preds = torch.zeros(kernel_targets.shape).to("cuda")
        elif self.sr_model == 'bicubic':
            upsampling_size =  [ i * self.scale_factor for i in x.size()[2:]]
            transform = transforms.Resize(upsampling_size, InterpolationMode.BICUBIC)
            sr_preds = transform(x)
            kernel_preds = torch.zeros(kernel_targets.shape).to("cuda")
        elif self.sr_model.__class__.__name__ == "KBPN":
            sr_preds, kernel_preds = self.sr_model(x, iter, kernel_targets)
        else:
            sr_preds = self.sr_model(x)
            kernel_preds = torch.zeros(kernel_targets.shape).to("cuda")
            
        return x, sr_targets, kernel_targets, sr_preds, kernel_preds

    def set_sr_model(self, cfg):
        if cfg.MODEL.SR_SEG_INV:
            num_channels = 1
        else:
            num_channels = 3

        if cfg.MODEL.SCALE_FACTOR == 1: 
            sr_model = None
        elif cfg.MODEL.SR == 'bicubic':
            sr_model = 'bicubic'
        elif cfg.MODEL.SR == 'SrcNetSR':
            sr_model = SrcNetSR(cfg)
        elif cfg.MODEL.SR == 'DBPN':
            sr_model = _MODEL[cfg.MODEL.NUM_STAGES](cfg.MODEL.SCALE_FACTOR, num_channels)
            if not cfg.MODEL.SR_SCRATCH:
                pretrained_model_path = os.path.join('weights', "pretrain", 'DBPN_pretrain_x{}_stage{}.pth'.format(cfg.MODEL.SCALE_FACTOR, cfg.MODEL.NUM_STAGES))
                m_key, u_key = sr_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path), addition_word='sr_model.'), strict=False)
                assert len(u_key) == 0, (f'unexpected_keys are exist.\n {u_key}')
                print('DBPN pretrained model was loaded from {}'.format(pretrained_model_path))
        elif cfg.MODEL.SR == 'KBPN':
            sr_model = KBPN(cfg, cfg.MODEL.NUM_STAGES, num_channels)
            if not cfg.MODEL.SR_SCRATCH:
                if cfg.BLUR.KERNEL_SIZE == cfg.BLUR.KERNEL_SIZE_OUTPUT:
                    pretrained_model_path = os.path.join('weights', "pretrain", 'KBPN_pretrain_x{}_stage{}.pth'.format(cfg.MODEL.SCALE_FACTOR, cfg.MODEL.NUM_STAGES))
                else:
                    pretrained_model_path = os.path.join('weights', "pretrain", 'KBPN_pretrain_x{}_stage{}_bicubic{}.pth'.format(cfg.MODEL.SCALE_FACTOR, cfg.MODEL.NUM_STAGES, cfg.BLUR.KERNEL_SIZE))
                m_key, u_key = sr_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path), addition_word='sr_model.'), strict=False)
                assert len(u_key) == 0, (f'unexpected_keys are exist.\n {u_key}')
                print('KBPN pretrained model was loaded from {}'.format(pretrained_model_path))
        elif cfg.MODEL.SR == "DSRL":
            sr_model = "DSRL"
        else:
            raise NotImplementedError(cfg.MODEL.SR)

        return sr_model

    def mount_cuda(self, var):
        for i in range(len(var)):
            if var[i] != None:
                var[i] = var[i].to("cuda")

        return var

    def norm_sr(self, sr_images):
        if self.norm_method == "all":
            num_channels = 3
            _mean = torch.empty(sr_images.shape).to('cuda')
            _std = torch.empty(sr_images.shape).to('cuda')
            for i in range(num_channels):
                _mean[:, i, :, :] = self.mean[i]
                _std[:, i, :, :] = self.std[i]
            sr_images_norm = (sr_images - _mean) / _std

        elif self.norm_method == "instance":
            norm = nn.InstanceNorm2d(3)
            sr_images_norm  = norm(sr_images)
        else:
            return sr_images
            
        return sr_images_norm

    def clip_sr(self, sr_preds):
        sr_preds[sr_preds>1] = 1 # clipping
        sr_preds[sr_preds<0] = 0 # clipping
        return sr_preds


class MetaSRLossCalc(nn.Module):
    def __init__(self, cfg, sr_transforms, num_train_ds, resume_iter, mro_name):
        next_class = mro_name[mro_name.index('MetaSRLossCalc') + 1]
        if next_class == MetaSSLossCalc.__name__:
            super().__init__(cfg, num_train_ds, resume_iter, mro_name)
        elif next_class != nn.Module.__name__:
            super().__init__(cfg, mro_name)
        else:
            super().__init__()
        self.sr_loss_fn = self.set_sr_loss(cfg, sr_transforms)

    def calc_sr_loss(self, x, sr_preds, sr_targets, iter, kernel_preds=None, kernel_targets=None, segment_preds=None, segment_targets=None):
        # print('calc1')
        # Super Resolution task
        if self.sr_model is None:
            sr_loss = None
        elif self.sr_model == 'bicubic':
            sr_loss = None
        elif self.sr_loss_fn.__class__.__name__ == "KBPNLoss":
            sr_loss, kernel_preds = self.sr_loss_fn(sr_preds, sr_targets, x, kernel_preds, kernel_targets, segment_preds, segment_targets, iter)
        else:
            sr_loss = self.sr_loss_fn(sr_preds, sr_targets).mean(dim=(1,2,3))
        
        return sr_loss, kernel_preds

    def set_sr_loss(self, cfg, sr_transforms):
        print(cfg.SOLVER.SR_LOSS_FUNC)
        if cfg.SOLVER.SR_LOSS_FUNC == 'KBPN':
            sr_loss_fn = KBPNLoss(cfg, sr_transforms, reduction="mean")
        elif cfg.SOLVER.SR_LOSS_FUNC == "L1":
            sr_loss_fn = nn.L1Loss(reduction='none')
        elif cfg.SOLVER.SR_LOSS_FUNC == "L2":
            sr_loss_fn = nn.MSELoss(reduction='none')
        elif cfg.SOLVER.SR_LOSS_FUNC == None:
            sr_loss_fn = None
        else:
            raise NotImplementedError(cfg.SOLVER.SR_LOSS_FUNC)

        return sr_loss_fn


class MetaSSModel(nn.Module):
    def __init__(self, cfg, mro_name):
        next_class = mro_name[mro_name.index('MetaSSModel') + 1]
        if next_class != nn.Module.__name__:
            super().__init__(cfg, mro_name)
        else:
            super().__init__()
        self.segmentation_model = self.set_ss_model(cfg)

    def forward_ss(self, sr_preds):
        # Semantic segmentation task
        if self.seg_model_name in ['PSPNet', 'HRNet_OCR', 'CrackFormer']:
            segment_preds, aux_segment_preds = self.segmentation_model(sr_preds)
            return segment_preds, aux_segment_preds
        else:
            segment_preds = self.segmentation_model(sr_preds)     
            return segment_preds, None

    def set_ss_model(self, cfg):
        num_classes = cfg.MODEL.NUM_CLASSES
        if cfg.MODEL.DETECTOR_TYPE == 'u-net16':
            self.seg_model_name = 'u-net16'
            segmentation_model = UNet16(num_classes=num_classes, pretrained=True, up_sampling_method=cfg.MODEL.UP_SAMPLE_METHOD)        
        elif cfg.MODEL.DETECTOR_TYPE == 'PSPNet':
            # Main loss and auxiliary loss conform to segmentation loss function
            self.seg_model_name = 'PSPNet'
            segmentation_model = PSPNet(n_classes=num_classes, pretrained=True)
        elif cfg.MODEL.DETECTOR_TYPE == 'PSPNet_BlurSkip':
            # Main loss and auxiliary loss conform to segmentation loss function
            self.seg_model_name = 'PSPNet_BlurSkip'
            segmentation_model = PSPNet_BlurSkip(cfg.BLUR.KERNEL_SIZE_OUTPUT ** 2, n_classes=num_classes, pretrained=True)
        elif cfg.MODEL.DETECTOR_TYPE == 'PSPNet_BlurSkip_origin':
            # Main loss and auxiliary loss conform to segmentation loss function
            self.seg_model_name = 'PSPNet_BlurSkip'
            segmentation_model = PSPNet_BlurSkip(cfg.BLUR.KERNEL_SIZE_OUTPUT ** 2, n_classes=num_classes, pretrained=True,  modify_blur_skip=False)
        elif cfg.MODEL.DETECTOR_TYPE == 'PSPNet_BlurSkipReduct':
            # Main loss and auxiliary loss conform to segmentation loss function
            self.seg_model_name = 'PSPNet_BlurSkipReduct'
            segmentation_model = PSPNet_BlurSkip(cfg.BLUR.KERNEL_SIZE ** 2, n_classes=num_classes, pretrained=True)
        elif cfg.MODEL.DETECTOR_TYPE == 'SegNet':
            self.seg_model_name = 'SegNet'
            segmentation_model = SegNet()
        elif cfg.MODEL.DETECTOR_TYPE == 'HRNet_OCR':
            # Main loss and auxiliary loss conform to segmentation loss function
            self.seg_model_name = 'HRNet_OCR'
            configer = set_configer("./model/modeling/hrnet_ocr/config/H_48_D_4_composite.json")
            segmentation_model = HRNet_W48_OCR(configer)
        elif cfg.MODEL.DETECTOR_TYPE == 'CrackFormer':
            self.seg_model_name = 'CrackFormer'
            segmentation_model = CrackFormer()
        elif cfg.MODEL.DETECTOR_TYPE == 'DSRL':
            self.seg_model_name = 'DSRL'
            segmentation_model = 'DSRL'
        else:
            raise NotImplementedError(cfg.MODEL.DETECTOR_TYPE)

        return segmentation_model


class MetaSSLossCalc(nn.Module):
    def __init__(self, cfg, num_train_ds, resume_iter, mro_name):
        super().__init__(cfg, mro_name)
        self.ss_loss_fn = self.set_ss_loss(cfg, num_train_ds, resume_iter)
        self.aux_weight = cfg.SOLVER.SEG_AUX_LOSS_WEIGHT
        self.main_weight = cfg.SOLVER.SEG_MAIN_LOSS_WEIGHT
        self.seg_model_name = cfg.MODEL.DETECTOR_TYPE
        self.iter_cnt = True

    def calc_ss_loss(self, segment_preds, segment_targets, aux_segment_preds):
        segment_targets = segment_targets.to('cuda')
        # Semantic segmentation task
        # print(self.seg_model_name)
        if aux_segment_preds != None:
            if "Boundary" in self.ss_loss_fn.__class__.__name__:
                if self.seg_model_name == "CrackFormer":
                    aux_segment_loss = self.ss_loss_fn(aux_segment_preds, segment_targets, iter_cnt=False) * aux_segment_preds.shape[1]
                else:
                    aux_segment_loss = self.ss_loss_fn(aux_segment_preds, segment_targets, iter_cnt=False)
                segment_loss = self.main_weight * self.ss_loss_fn(segment_preds, segment_targets, iter_cnt=self.iter_cnt) + self.aux_weight * aux_segment_loss
            else:
                aux_segment_loss = self.ss_loss_fn(aux_segment_preds, segment_targets)
                segment_loss = self.main_weight * self.ss_loss_fn(segment_preds, segment_targets) + self.aux_weight * aux_segment_loss
        else:
            if "Boundary" in self.ss_loss_fn.__class__.__name__:
                segment_loss = self.ss_loss_fn(segment_preds, segment_targets, iter_cnt=self.iter_cnt)
            else:
                segment_loss = self.ss_loss_fn(segment_preds, segment_targets)

        return segment_loss

    def set_ss_loss(self, cfg, num_train_ds, resume_iter):
        seg_rsm_iter = resume_iter - (cfg.SOLVER.SR_PRETRAIN_ITER[1] - 1)  if resume_iter > (cfg.SOLVER.SR_PRETRAIN_ITER[1] - 1) else 0
        out_map = False if cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SS_AMP == 0 and cfg.SOLVER.INTERM_SSLOSSWEGHT4SR == False  else True
        if cfg.SOLVER.SEG_LOSS_FUNC == "BCE":
            ss_loss_fn = nn.BCELoss().to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "WeightedBCE":
            pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
            ss_loss_fn = WeightedBCELoss(pos_weight=pos_weight).to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "Dice":
            ss_loss_fn = BinaryDiceLoss().to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "Combo":
            pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
            loss_weight = cfg.SOLVER.WB_AND_D_WEIGHT
            ss_loss_fn = BCE_DiceLoss(pos_weight=pos_weight, loss_weight = loss_weight).to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "BoundaryCombo": # old: Boundary
            pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
            loss_weight = cfg.SOLVER.WB_AND_D_WEIGHT
            per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1   
            ss_loss_fn = BoundaryComboLoss(per_epoch, pos_weight=pos_weight, loss_weight=loss_weight, 
                                            decrease_ratio=cfg.SOLVER.BOUNDARY_DEC_RATIO,
                                            resume_iter = seg_rsm_iter, out_map=out_map).to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "Boundary_GDice":
            per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1
            ss_loss_fn = Boundary_GDiceLoss(per_epoch, resume_iter=seg_rsm_iter, 
                                            decrease_ratio=cfg.SOLVER.BOUNDARY_DEC_RATIO).to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "GeneralizedBoundaryCombo":
            pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
            loss_weight = cfg.SOLVER.WB_AND_D_WEIGHT
            per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1
            ss_loss_fn = GeneralizedBoundaryComboLoss(per_epoch, pos_weight=pos_weight, loss_weight=loss_weight,
                                            decrease_ratio=cfg.SOLVER.BOUNDARY_DEC_RATIO,
                                            resume_iter = seg_rsm_iter).to('cuda')
        elif cfg.SOLVER.SEG_LOSS_FUNC == "CrackFormerLoss":
            ss_loss_fn = cross_entropy_loss_RCF
        else:
            raise NotImplementedError(cfg.SOLVER.SEG_LOSS_FUNC)

        return ss_loss_fn


# ================================================= Derived class ===========================================================


class JointModelWithLoss(MetaSRLossCalc, MetaSSLossCalc, MetaSRModel, MetaSSModel):
    def __init__(self, cfg, num_train_ds, resume_iter, sr_transforms):
        mro_name = [k.__name__ for k in JointModelWithLoss.__mro__]
        super().__init__(cfg, sr_transforms, num_train_ds, resume_iter, mro_name)
        self.oriented_w_iter = cfg.SOLVER.ORIENTED_WEIGHT_ITER
        self.w_co_sr = CrackOrientedExpWeight(cfg, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.CRACK_ORIENTED_WEIGHT4SR_BIAS)
        self.w_sfo_sr = SegmentFailerOrientedExpWeight(cfg, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_AMP, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_BIAS)
        self.w_sfo_ss = SegmentFailerOrientedExpWeight(cfg, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SS_AMP, cfg.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SS_BIAS)
        self.w_ssloss_sr = cfg.SOLVER.INTERM_SSLOSSWEGHT4SR
        self.blur_ksize = cfg.BLUR.KERNEL_SIZE  # Estimate kernel_dim by networks. If you use upsampling, KERNEL_SIZE_OUTPUT is different from KERNEL_SIZE and the 
                                                # required kernel size is KERNEL_SIZE_OUTPUT.
        self.gap = nn.AdaptiveAvgPool2d(1)
        if cfg.MODEL.DETECTOR_TYPE == 'DSRL' and  cfg.MODEL.SR == 'DSRL':
            self.upsample = cfg.MODEL.DSRL_UPSAMPLE
            if self.upsample == 'deconv':
                self.parallel_model = DeepLabx4(num_classes=1)
                fname = 'DSRLx4'
                print('This training model is DSRLx4 that is original model.')
            else:
                self.parallel_model = DeepLab(num_classes=1)
                fname = 'DSRL'
            
            if not cfg.MODEL.SR_SCRATCH:
                pretrained_model_path = os.path.join('weights', f'{fname}.pth')
                m_key, u_key = self.parallel_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path), addition_word='parallel_model.'), strict=False)
                assert len(u_key) == 0, (f'unexpected_keys are exist.\n {u_key}')
                print('DSRL pretrained model was loaded from {}'.format(pretrained_model_path))
            self.fa_loss = FALoss()

        if self.seg_model_name in ['PSPNet_BlurSkip', 'PSPNet_BlurSkip_origin', 'PSPNet_BlurSkipReduct']:
            if hasattr(self, 'module'):
                for param in self.module.sr_model.parameters():
                    param.requires_grad = False
                for param in self.module.segmentation_model.parameters():
                    param.requires_grad = False
                for param in self.module.segmentation_model.blur_skip.parameters():
                    param.requires_grad = True
            else:
                for param in self.sr_model.parameters():
                    param.requires_grad = False
                for param in self.segmentation_model.parameters():
                    param.requires_grad = False
                for param in self.segmentation_model.blur_skip.parameters():
                    param.requires_grad = True

            print('+++++++ Fixed all parameters except BlurSkip. +++++++') 

    def forward(self, iter, x, sr_targets=None, segment_targets=None, kernel_targets=None):
        if self.sr_model == "DSRL":
            x, sr_targets, kernel_targets = self.mount_cuda([x, sr_targets, kernel_targets])
            if segment_targets != None:
                segment_targets = self.mount_cuda([segment_targets])[0]
            segment_preds, sr_preds, fea_seg, fea_sr = self.parallel_model(x)
            # print(f'segment_preds: {segment_preds.shape}, sr_preds: {sr_preds.shape}, fea_seg:{fea_seg.shape}, fea_sr:{fea_sr.shape}')
            kernel_preds = torch.zeros(kernel_targets.shape).to("cuda")

            # Interpolation of SR for changes in image size due to the design nature of transposed convolution
            # if segment_preds.shape[2] != sr_preds.shape[2]:
            #     h, w = segment_preds.shape[2:]
            #     sr_preds = F.interpolate(sr_preds, size=(h, w), mode='bicubic', align_corners=True)
            #     fea_sr = F.interpolate(fea_sr, size=(h, w), mode='bicubic', align_corners=True)

            if self.upsample != 'deconv':
                h, w = sr_targets.shape[2:]
                sr_preds = F.interpolate(sr_preds, size=(h, w), mode=self.upsample, align_corners=True)
                segment_preds = F.interpolate(segment_preds, size=(h, w), mode=self.upsample, align_corners=True)

            fa_loss = self.fa_loss(fea_seg, fea_sr)
            sr_loss, _ = self.calc_sr_loss(x, sr_preds, sr_targets, iter, None, None, None, None)
            if segment_targets != None:
                segment_loss = self.calc_ss_loss(segment_preds, segment_targets, None)
            else:
                segment_loss = 0

            return segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds, fa_loss
        else:
            x, sr_targets, kernel_targets, sr_preds, kernel_preds = self.forward_sr(iter, x, sr_targets, kernel_targets)
            if self.seg_model_name == 'PSPNet_BlurSkip':
                segment_preds, aux_segment_preds = self.forward_ss_with_blurkernel(self.norm_sr(sr_preds), kernel_preds)
            elif self.seg_model_name == 'PSPNet_BlurSkip_origin':
                segment_preds, aux_segment_preds = self.forward_ss_with_blurkernel(self.norm_sr(sr_preds), kernel_preds)
            elif self.seg_model_name == 'PSPNet_BlurSkipReduct':
                kernel_preds_2d = self.gap(kernel_preds.clone()).view(kernel_preds.shape[0], 1, *kernel_targets.shape[2:])
                kernel_preds_reduct = F.interpolate(kernel_preds_2d, size=self.blur_ksize, mode='bicubic', align_corners=True)
                kernel_preds_reduct_vec = kernel_preds_reduct.view(-1, self.blur_ksize ** 2, 1, 1)
                segment_preds, aux_segment_preds = self.forward_ss_with_blurkernel(self.norm_sr(sr_preds), kernel_preds_reduct_vec)
            else:
                segment_preds, aux_segment_preds = self.forward_ss(self.norm_sr(sr_preds))

            sr_loss, kernel_preds = self.calc_sr_loss(x, sr_preds, sr_targets, iter, kernel_preds, kernel_targets, segment_preds, segment_targets)
            segment_loss = self.calc_ss_loss(segment_preds, segment_targets, aux_segment_preds)
            sr_loss, segment_loss = self.multiple_weight(sr_loss, segment_loss, segment_preds, segment_targets, iter)

            return segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds

    def forward_ss_with_blurkernel(self, sr_preds, kernel_preds):
        segment_preds, aux_segment_preds = self.segmentation_model(sr_preds, kernel_preds)
        return segment_preds, aux_segment_preds

    def multiple_weight(self, sr_loss, segment_loss, segment_preds, segment_targets, iter):
        if self.oriented_w_iter <= iter:
            
            if self.sr_loss_fn.__class__.__name__ != "KBPNLoss":    # In KBPN loss, the weight calculation is performed internally.
                if self.w_co_sr.amp != 0:
                    # print(self.w_co_sr(segment_targets).shape, sr_loss.shape)
                    sr_loss = self.w_co_sr(segment_targets) * sr_loss
                if self.w_sfo_sr.amp != 0:
                    sr_loss = self.w_sfo_sr(segment_preds, segment_targets) * sr_loss
            # if self.w_co_ss.amp != 0:
            #     segment_loss = self.w_co_ss(segment_targets) * segment_loss
            if self.w_sfo_ss.amp != 0:
                segment_loss = self.w_sfo_ss(segment_preds, segment_targets) * segment_loss
            if self.w_ssloss_sr:
                w_segment_loss = segment_loss.detach()
                sr_loss = w_segment_loss * sr_loss
        return sr_loss, segment_loss


class JointModel(MetaSRModel, MetaSSModel):
    def __init__(self, cfg):
        mro_name = [k.__name__ for k in JointModel.__mro__]
        super().__init__(cfg, mro_name)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.ksize = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.blur_ksize = cfg.BLUR.KERNEL_SIZE  # Estimate kernel_dim by networks. If you use upsampling, KERNEL_SIZE_OUTPUT is different from KERNEL_SIZE and the 
                                                # required kernel size is KERNEL_SIZE_OUTPUT.
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        if cfg.MODEL.DETECTOR_TYPE == 'DSRL' and  cfg.MODEL.SR == 'DSRL':
            self.upsample = cfg.MODEL.DSRL_UPSAMPLE
            if self.upsample == 'deconv':
                self.parallel_model = DeepLabx4(num_classes=1)
                fname = 'DSRLx4'
                print('This training model is DSRLx4 that is original model.')
            else:
                self.parallel_model = DeepLab(num_classes=1)
                fname = 'DSRL'
            
            if not cfg.MODEL.SR_SCRATCH:
                pretrained_model_path = os.path.join('weights', f'{fname}.pth')
                m_key, u_key = self.parallel_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path), addition_word='parallel_model.'), strict=False)
                assert len(u_key) == 0, (f'unexpected_keys are exist.\n {u_key}')
                print('DSRL pretrained model was loaded from {}'.format(pretrained_model_path))
                
    def forward(self, x, damy_kernel, sr_targets=None):
        iter = -1
        if self.sr_model == "DSRL":
            x, sr_targets, kernel_targets = self.mount_cuda([x, sr_targets, damy_kernel])
            segment_preds, sr_preds, fea_seg, fea_sr = self.parallel_model(x)
            kernel_preds = torch.zeros(kernel_targets.shape).to("cuda")
            if self.upsample != 'deconv':
                h, w = [a * self.scale_factor for a in x.shape[2:]]
                sr_preds = F.interpolate(sr_preds, size=(h, w), mode=self.upsample, align_corners=True)
                segment_preds = F.interpolate(segment_preds, size=(h, w), mode=self.upsample, align_corners=True)

        else:
            x, _, _, sr_preds, kernel_preds = self.forward_sr(iter, x, sr_targets=sr_targets, kernel_targets=damy_kernel)
            
            sr_preds = self.clip_sr(sr_preds)
            if self.seg_model_name == 'PSPNet_BlurSkip':
                segment_preds, aux_segment_preds = self.forward_ss_with_blurkernel(self.norm_sr(sr_preds), kernel_preds)
            elif self.seg_model_name == 'PSPNet_BlurSkipReduct':
                kernel_preds_2d = self.GAP(kernel_preds.clone()).view(kernel_preds.shape[0], 1, self.ksize, self.ksize)
                kernel_preds_reduct = F.interpolate(kernel_preds_2d, size=self.blur_ksize, mode='bicubic', align_corners=True)
                kernel_preds_reduct_vec = kernel_preds_reduct.view(-1, self.blur_ksize ** 2, 1, 1)
                segment_preds, aux_segment_preds = self.forward_ss_with_blurkernel(self.norm_sr(sr_preds), kernel_preds_reduct_vec)
            else:
                segment_preds, aux_segment_preds = self.forward_ss(self.norm_sr(sr_preds))

            if self.sr_model.__class__.__name__ == "KBPN":
                kernel_preds = self.GAP(kernel_preds)
                kernel_preds = kernel_preds / ((kernel_preds.sum(dim=1)).view(kernel_preds.shape[0],1,1,1))
                kernel_preds = kernel_preds.view(kernel_preds.shape[0], 1, self.ksize, self.ksize)

        return sr_preds, segment_preds, kernel_preds

    def forward_ss_with_blurkernel(self, sr_preds, kernel_preds):
        segment_preds, aux_segment_preds = self.segmentation_model(sr_preds, kernel_preds)
        return segment_preds, aux_segment_preds


class JointInvModelWithLoss(MetaSSLossCalc, MetaSRLossCalc, MetaSRModel, MetaSSModel):
    def __init__(self, cfg, num_train_ds, resume_iter, sr_transforms):
        super().__init__(cfg, num_train_ds, resume_iter)
        self.sr_transforms = sr_transforms

    def forward(self, iter, x, sr_targets=None, segment_targets=None, kernel_targets=None):
        lr_segment_targets = self.sr_transforms(segment_targets)

        lr_segment_preds = self.forward_ss(x)
        _, _, _, segment_preds, kernel_preds = self.forward_sr(iter, lr_segment_preds)
        segment_loss = self.calc_ss_loss(lr_segment_preds, lr_segment_targets)
        sr_loss = self.calc_sr_loss(lr_segment_preds, segment_preds, segment_targets, kernel_preds)

        return segment_loss, sr_loss, segment_preds, lr_segment_preds, kernel_preds


class JointInvModel(MetaSRModel, MetaSSModel):
    def __init__(self, cfg):
        mro_name = [k.__name__ for k in JointInvModel.__mro__]
        super().__init__(cfg, mro_name)
        # MetaSRModel.__init__(self, cfg)
        # MetaSSModel.__init__(self, cfg)

    def forward(self, x):
        iter = -1
        lr_segment_preds = self.forward_ss(x)
        _, _, _, segment_preds, kernel_preds = self.forward_sr(iter, lr_segment_preds)
        segment_preds = self.clip_sr(segment_preds)
            
        return lr_segment_preds, segment_preds, kernel_preds


class SRModelWithLoss(MetaSRLossCalc, MetaSRModel):
    def __init__(self, cfg, sr_transforms, num_train_ds=None, resume_iter=None):
        mro_name = [k.__name__ for k in SRModelWithLoss.__mro__]
        super().__init__(cfg, sr_transforms, num_train_ds, resume_iter, mro_name)
        # super(MetaSRModel, self).__init__(cfg)
        # super(MetaSRLossCalc, self).__init__()
        # MetaSRModel.__init__(self, cfg)
        # print(self.sr_model)
        # MetaSRLossCalc.__init__(self, cfg)
        # print(self.sr_model)

    def forward(self, iter, x, sr_targets=None, kernel_targets=None):
        # print('start forward')
        # print(sr_targets.size(), kernel_targets.size())
        x, sr_targets, kernel_targets, sr_preds, kernel_preds = self.forward_sr(iter, x, sr_targets, kernel_targets)
        sr_loss, kernel_preds = self.calc_sr_loss(x, sr_preds, sr_targets, iter, kernel_preds, kernel_targets)
        # print('fin forward')
        return sr_loss, sr_preds, kernel_preds


class SRModel(MetaSRModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, iter, x, sr_targets=None, kernel_targets=None):
        x, sr_targets, kernel_targets, sr_preds, kernel_preds = self.forward_sr(iter, x, sr_targets, kernel_targets)
        
        return sr_preds, kernel_preds

