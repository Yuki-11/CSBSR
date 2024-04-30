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
import numpy as np


class SplitPatch:
    def __init__(self, batch_size, ch, patch_sizeh, patch_sizew):
        self.kc, self.kh, self.kw = ch, patch_sizeh, patch_sizew  # kernel size
        self.sc, self.sh, self.sw = ch, patch_sizeh, patch_sizew  # stride
        self.batch_size = batch_size

    def __call__(self, x):
        patches = x.unfold(0, self.kc, self.sc).unfold(1, self.kh, self.sh).unfold(2, self.kw, self.sw)
        unfold_shape = patches.size() # [splited C, splited H, splited W, patch C, patch H, patch W]
        patches = patches.contiguous().view(-1, self.kc, self.kh, self.kw)
        unfold_shape = np.append(self.batch_size, np.array(unfold_shape))
        # print(patches.shape)
        return patches, unfold_shape


class JointPatch:
    def __init__(self):
        pass

    def __call__(self, patches, unfold_shape, batch_size=-1):
        # print(patches.shape)
        # unfold_shape[0] = int(patches.shape[0] / (unfold_shape[2] * unfold_shape[3])) # Support for variable batch size
        # print(patches, type(patches))
        unfold_shape = list(unfold_shape)
        # if not batch_size == -1:
        #     if 
        #         print()
        unfold_shape[0] = -1
            
        # print(patches.shape, unfold_shape)
        patches_orig = patches.view(*unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        patches_orig = patches_orig.view(-1, output_c, output_h, output_w)
        return patches_orig
