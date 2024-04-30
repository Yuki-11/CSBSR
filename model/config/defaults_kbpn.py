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

_C = CN()
cfg = _C

_C.MODEL = CN()
_C.MODEL.KERNEL_SFT = True

_C.TRAINER = CN()
# _C.TRAINER.UPSCALE_FACTOR = 4
# _C.TRAINER.BATCH_SIZE = 1
# _C.TRAINER.PRETRAINED_ITER = 100
# _C.TRAINER.PRETRAIN = False
# _C.TRAINER.EPOCHS = 100
# _C.TRAINER.SNAPSHOTS = 25
# _C.TRAINER.START_ITER = 1
# _C.TRAINER.LR = 0.0001
# _C.TRAINER.GPU_MODE = True
# _C.TRAINER.THREADS = 8
# _C.TRAINER.SEED = 123
# _C.TRAINER.GPUS = 1
_C.TRAINER.FEATURE_EXTRACTOR = 'VGG'
_C.TRAINER.USE_FEATURE_EXTRACTOR = False

_C.TRAINER.W1 = 1.0  # weight of MSE(SR, HR) loss
_C.TRAINER.W2 = 0.1  # weight of perceptual loss (= VGG loss)
_C.TRAINER.W3 = 0.001  # weight of adversatial loss
_C.TRAINER.W4 = 10.0   # weight of style loss
_C.TRAINER.W5 = 100.0  # weight of MSE(predicted kernel, GT kernel) loss
#_C.TRAINER.SUM1_IN_KLOSS = False
_C.TRAINER.WLR = 1.0  # weight of MSE(pseudo LR, input LR) loss
_C.TRAINER.W_INTER_SR = 0.05 # for intermediate SR supervision 
_C.TRAINER.W_INTER_LR = 0.05 # for intermediate LR supervision
_C.TRAINER.W_K_REGU = 0.1

_C.TRAINER.INTER_SUPERVISION = False
_C.TRAINER.INTER_SR = False
_C.TRAINER.INTER_LR = False
_C.TRAINER.LOSS_CALCULATION_STAGE = [] # for intermediate supervision
_C.TRAINER.COMPLEX_INTER_SUPERVISION = False
_C.TRAINER.GAN_TRAINING = False

_C.TRAINER.SAVE_FOLDER = 'weights/'
_C.TRAINER.PREFIX = 'PIRM_VGG'
_C.TRAINER.MOTION_BLUR_MODE = 'length'
_C.TRAINER.MOTION_LIMITATION = True

_C.TRAINER.SAVE_OPTIMIZER = True
_C.TRAINER.SAVE_DISCRIMINATOR = False

_C.TRAINER.DEBUG = False
_C.TRAINER.ID = 0
_C.TRAINER.KERNEL_REGULARIZATION = False



_C.TRAINER.KERNEL_PRETRAIN = False
_C.TRAINER.SR_PRETRAIN = False

_C.TRAINER.PCA_COMPONENTS = '' #



_C.DATA = CN()
_C.DATA.DATA_DIR = './Dataset'
_C.DATA.DATA_AUGMENTATION = True
_C.DATA.HR_TRAIN_DATASET = 'DIV2K_train_HR'
_C.DATA.PATCH_SIZE = 60
_C.DATA.SEMI_PATCH_SIZE = 300
_C.DATA.BLUR = ['motion']
_C.DATA.DEBUG = False
_C.DATA.DO_PCA = True # needless for KBPN
_C.DATA.DO_GAP = False
_C.DATA.PCA_SAMPLE = 500 # needless for KBPN
_C.DATA.NORM_AFTER_PCA = False

_C.DATA.KERNEL_SIZE = 7 # predicted kernel size

_C.DATA.KERNEL_SIZE_FOR_LR = 21 # kernel size for producing LR image

_C.DATA.HOW_TO_DOWNSCALE = 'stride' ## Choose 'area' or 'stride'
_C.DATA.BLUR_COMBINATION = False 
_C.DATA.PCA_MIX = False ## 
_C.DATA.GAUSSIAN_SIGMA_X = [0.2, 4.0]
_C.DATA.GAUSSIAN_SIGMA_Y = [0.2, 4.0]
_C.DATA.GAUSSIAN_ISOTROPIC = True

_C.DATA.GAUSSIAN_NOISE = False
_C.DATA.GAUSSIAN_NOISE_SIGMA = 15.0
_C.DATA.GAUSSIAN_NOISE_MEAN = 0.0


_C.MODEL = CN()
_C.MODEL.MODEL_TYPE = 'KBPNS' ## mainly use 'KBPNLL'
_C.MODEL.KBPN_TYPE = 'withGAP'  ## Choice 'v1', 'v2', 'withGAP (main)'
_C.MODEL.PRETRAINED_SR = ''  ## pretrained SR model (if former kbpn(=LR-KBPN), '(model dir)/withGAP/kbpn/(filename).pth'
_C.MODEL.PRETRAINED_KERNEL_PREDICTOR = '' ## If new kbpn (= SR-KBPN), ''. If former kbpn (= LR-KBPN) '(model dir)/withGAP/predictor/(filename).pth'
_C.MODEL.LOAD_PRETRAINED = False
_C.MODEL.LOAD_PREDICTOR = '' ## If new kbpn (= SR-KBPN), ''. If former kbpn (= LR-KBPN) '(model dir)/withGAP/predictor/(filename).pth'
_C.MODEL.PRETRAINED_D = 'dnnDBPNLLPIRM_RESNET_epoch_Discriminator_499.pth'
_C.MODEL.LOAD_PRETRAINED_D = False
_C.MODEL.LOAD_OPTIMIZER = ''
_C.MODEL.WHICH_CONDITION = 'kernel'
_C.MODEL.KERNEL_NORM = False ## (True:)Normalize kernel [0, 1] / (False: ) Not
_C.MODEL.SUM_KERNEL_1 = True
_C.MODEL.USE_SR_IN_KPRED = False
_C.MODEL.USE_PRE_KERNEL = True
_C.MODEL.VERSION = 'v2' ## Choose '' (= 1st implementation) or '' (= 2nd implementation)
_C.MODEL.COMMON_K_PREDICTOR = True
_C.MODEL.SUB_IN_KERNEL_PRED = 'square'
_C.MODEL.LR_SUB_IN_UNIT = 'just_sub'
_C.MODEL.WEIGHT_INIT_MEAN = 0.0
_C.MODEL.WEIGHT_INIT_STD = 0.0001
_C.MODEL.GRADIENT_CLIP_VALUE = 1.0
_C.MODEL.USE_GT_KERNEL_FOR_UPPER_LIMIT = False
_C.MODEL.KERNEL_P_VERSION = 'v1' ## Choose 'v0', 'v1', 'v2', 'v3'
_C.MODEL.HOW_TO_USE_PRE_KERNEL = 'concat' ## Choose 'concat' or 'els'(=element-wise sum)
_C.MODEL.CONDITIONING_KERNEL = False
_C.MODEL.CONDITIONING_KERNEL_PER_FEATURE = False
_C.MODEL.WEIGHT_NORM = False
_C.MODEL.KERNEL_PRED_BN = False
_C.MODEL.KERNEL_ACT = 'leaky_relu'
_C.MODEL.KERNELIZATION = 'sum'
_C.MODEL.KERNEL_EPS = 1e-24
_C.MODEL.RESIDUAL_CONDITION_ON_SR_FEATURE = False
_C.MODEL.USE_INPUT_NORM = True
_C.MODEL.KERNEL_SFT = True

_C.MODEL.FINETUNE = CN()
_C.MODEL.FINETUNE.KERNEL_P_VERSION = ''


cfg = _C