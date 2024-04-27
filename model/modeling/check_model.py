##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from .dbpn import *
from .unet import *
from .kbpn import KBPN
from model.modeling.pspnet_pytorch.pspnet import PSPNet
from model.config import cfg
from torchsummary import summary

num_channels = 3
model = KBPN(cfg, cfg.MODEL.NUM_STAGES, num_channels)  
summary(model,(3, 448, 448))