##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from yacs.config import CfgNode as CN
# from ..data.transforms.transforms import *

_C = CN()
_C.DEVICE = 'cuda'

_C.MODEL = CN()
_C.MODEL.SCALE_FACTOR = 4
_C.MODEL.DETECTOR_TYPE = 'u-net16' # 'PSPNet'
_C.MODEL.SR = 'DBPN'
_C.MODEL.UP_SAMPLE_METHOD = "deconv" # "pixel_shuffle"
_C.MODEL.DETECTOR_DBPN_NUM_STAGES = 4
_C.MODEL.OPTIMIZER = 'Adam' # SGD
_C.MODEL.NUM_CLASSES = 1  
_C.MODEL.NUM_STAGES = 4
_C.MODEL.SR_SEG_INV = False
_C.MODEL.JOINT_LEARNING = True
_C.MODEL.SR_RESIDUAL_LEARNING = True
_C.MODEL.KBPN_KERNEL_SFT = True
_C.MODEL.SR_PIXEL_SHUFFLE = False
_C.MODEL.SR_SCRATCH = False
_C.MODEL.DSRL_UPSAMPLE = 'bilinear'
_C.MODEL.SUM_LR_ERROR_POS = 'HR'
_C.MODEL.ZERO_PAD_KERNEL = False

_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 300000
_C.SOLVER.TRAIN_DATASET_RATIO = 0.95
_C.SOLVER.SR_PRETRAIN_ITER = [1, 150001] # [start, stop]
_C.SOLVER.SR_SR_MODULE_PRETRAIN_ITER = [1, 50001]
_C.SOLVER.SR_KERNEL_MODULE_PRETRAIN_ITER = [50001, 100000]
_C.SOLVER.ONLY_KERNEL_LOSS_FOR_PRETRAIN = False

_C.SOLVER.SEG_PRETRAIN_ITER = [0, 0]
_C.SOLVER.BATCH_SIZE = 8 # default 16

_C.SOLVER.TASK_LOSS_WEIGHT = 0.5 # -1: increase
_C.SOLVER.INCRESE_TASK_W_ITER = [30000, 170000]
_C.SOLVER.SEG_LOSS_FUNC = "Dice"    # Dice or BCE or WeightedBCE or Boundary or WBCE&Dice or GDC_Boundary
_C.SOLVER.BOUNDARY_DEC_RATIO = 1.0
_C.SOLVER.WB_AND_D_WEIGHT = [1, 1] # [WBCE ratio, Dice ratio]
_C.SOLVER.BCELOSS_WEIGHT = [20, 1] # [True ratio, False ratio]
_C.SOLVER.SEG_AUX_LOSS_WEIGHT = 0.4
_C.SOLVER.SEG_MAIN_LOSS_WEIGHT = 1.0
_C.SOLVER.DSRL_FA_WEIGHT = 0.5
_C.SOLVER.DSRL_SR_WEIGHT = 0.5
_C.SOLVER.DSRL_SEG_WEIGHT = 1.0

_C.SOLVER.ORIENTED_WEIGHT_GAUS = 2
_C.SOLVER.ORIENTED_WEIGHT_ITER = -1
_C.SOLVER.CRACK_ORIENTED_WEIGHT4SR_AMP = 0.0 # 2
_C.SOLVER.CRACK_ORIENTED_WEIGHT4SR_BIAS = 1.0
_C.SOLVER.CRACK_ORIENTED_WEIGHT4SS_AMP = 0.0
_C.SOLVER.CRACK_ORIENTED_WEIGHT4SS_BIAS = 1.0
_C.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_AMP = 0.0 # 1.5
_C.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SR_BIAS = 1.0
_C.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SS_AMP = 0.0
_C.SOLVER.SEG_FAIL_ORIENTED_WEIGHT4SS_BIAS = 1.0
_C.SOLVER.INTERM_SSLOSSWEGHT4SR = False

_C.SOLVER.SR_LOSS_FUNC = "L1"    # L1 or Boundary
_C.SOLVER.SR_LOSS_FUNC_SR_WEIGHT = [0.4, 0.4, 0,2] # [HR, LR, kernel] loss weight

_C.SOLVER.LR_LOSS_FUNC = "L1"
_C.SOLVER.ALPHA_MIN = 0.01
_C.SOLVER.DECREASE_RATIO = 1.0
_C.SOLVER.SYNC_BATCHNORM = True
_C.SOLVER.NORM_SR_OUTPUT = "all"
_C.SOLVER.LR = 1e-3
_C.SOLVER.LR_STEPS = []
_C.SOLVER.SCHEDULER = True
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 1.0
_C.SOLVER.WARMUP_ITERS = 5000
_C.SOLVER.DOWNSCALE_INTERPOLATION = 'bicubic'


_C.BLUR = CN()
_C.BLUR.FLAG = True
_C.BLUR.KERNEL_SIZE = 21 # Estimate kernel_dim by networks. If you use upsampling, KERNEL_SIZE_OUTPUT is different from KERNEL_SIZE and the required kernel size is KERNEL_SIZE_OUTPUT.
_C.BLUR.KERNEL_SIZE_OUTPUT = 21 # kernel_dim
_C.BLUR.ISOTROPIC = False

_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = [448, 448] # H x W
_C.INPUT.MEAN = [0.4741, 0.4937, 0.5048]
_C.INPUT.STD = [0.1621, 0.1532, 0.1523]

_C.DATASET = CN()
_C.DATASET.ONLY_IMAGES = False
_C.DATASET.DATA_AUGMENTATION = [["ConvertFromInts", None],
                                ["RandomMirror", None],
                                # [PhotometricDistort, None],
                                # [Normalize(cfg), None],
                                ["ToTensor", None],
                                # [RandomGrayscale, {'p':0.25},
                                ["RandomVerticalFlip", {'p':0.3}],
                                ["RandomResizedCrop", {'scale': (1.0, 1.0), 'ratio': (1.0, 1.0)}], # scale, ratio
                                ]
_C.DATASET.TRAIN_IMAGE_DIR = 'datasets/crack_segmentation_dataset/train/images'
_C.DATASET.TRAIN_MASK_DIR = 'datasets/crack_segmentation_dataset/train/masks'
_C.DATASET.TEST_IMAGE_DIR = 'datasets/crack_segmentation_dataset/test_blured/gt/images'
_C.DATASET.TEST_MASK_DIR = 'datasets/crack_segmentation_dataset/test_blured/gt/masks'
_C.DATASET.TEST_BLURED_DIR = 'datasets/crack_segmentation_dataset/test_blured/'
_C.DATASET.TEST_BLURED_NAME = '02_40'


_C.OUTPUT_DIR = 'output/CSSR_SR-SS'
_C.SEED = 1121

_C.BASE_NET = 'weights/vgg16_reducedfc.pth'
