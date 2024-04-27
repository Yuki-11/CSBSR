##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Kernelized Back-Projection Networks for Blind Super Resolution
## Tomoki Yoshida, Yuki Kondo, Takahiro Maeda, Kazutoshi Akita, Norimichi Ukita
## 
## This code is based on https://github.com/Yuki-11/KBPN, and is licensed Apache LICENSE.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# from network.base_networks_new_0129 import UpBlock, ConvBlock, DeconvBlock, predictor_withGAP, KBlock, DownBlock, SFTlayer

import torch
import torch.nn as nn
# from collections import OrderedDict
import torch.nn.functional as F
import torchvision


class KBPN(nn.Module):
    # DenseKBPN
    def __init__(self, cfg, num_stage, num_ch, feat_ch=256, md_ch=128): # default md_ch = 64
        super(KBPN, self).__init__()
        print('init')
        scale_factor = cfg.MODEL.SCALE_FACTOR
        conv_setting = {2: [6, 2, 2],  # scale_factor:[conv_k_sz, stride, padding]
                        4: [8, 4, 2], 
                        8: [12, 8, 2]}
        conv_k_sz, stride, padding = conv_setting[scale_factor]

        self.condition_ch = cfg.BLUR.KERNEL_SIZE_OUTPUT ** 2   # original: cfg.BLUR.KERNEL_SIZE**2
        self.kernel_sft = cfg.MODEL.KBPN_KERNEL_SFT     # sub
        self.sr_pretrain_iter = cfg.SOLVER.SR_SR_MODULE_PRETRAIN_ITER
        self.kernel_pretrain_iter = cfg.SOLVER.SR_KERNEL_MODULE_PRETRAIN_ITER
        self.residual_learning = cfg.MODEL.SR_RESIDUAL_LEARNING
        # pretrain flag
        self.train_flag = {"SR": True, "kernel": True}  
        # self.train_flag = True

        # #Initial Feature_Extraction
        # self.feat = nn.Sequential(ConvBlock(num_ch, feat_ch, 3, 1, 1, activation='prelu', norm=None),
        #                         ConvBlock(feat_ch, md_ch, 1, 1, 0, activation='prelu', norm=None)
        #                         )

        vgg = list(torchvision.models.vgg16(pretrained=True).features)
        vgg = vgg[:4] + vgg[5:9]
        self.feat = nn.Sequential(*vgg)

        """
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            <remove>  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
        """

        # #Initial Kernel Prediction
        # if not cfg.TRAINER.SR_PRETRAIN:
        self.predictor = predictor_withGAP(input_ch=md_ch, output_ch=cfg.BLUR.KERNEL_SIZE ** 2, stride=1, padding=1, ksize_output=cfg.BLUR.KERNEL_SIZE_OUTPUT, weight_norm=None)
        
        # Back-projection stages
        self.num_stage = num_stage
        bps = [KernelBackProjectionStageWithSFT(md_ch, conv_k_sz, stride, padding, cfg, stage) for stage in range(1, num_stage)] +\
            [KernelBackProjectionStageWithSFT(md_ch, conv_k_sz, stride, padding, cfg, num_stage, final_stage=True)]
        self.back_projection_stages = nn.ModuleList(bps)
        #Reconstruction
        self.output_conv = ConvBlock(num_stage*md_ch, num_ch, 3, 1, 1, activation=None, normalization=None)
        if self.residual_learning:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic')

        # initialization -> functional
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, iter, kernel=None):
        self._pretrain_check(iter)

        init_f = self.feat(x)

        if self.sr_pretrain_iter[0] <= iter < self.sr_pretrain_iter[1]:
            # print(kernel.size())
            init_kernel = kernel.view(-1, self.condition_ch, 1, 1)
        else:
            init_kernel, _ = self.predictor(init_f)
            # print('init_kernel', init_kernel.shape)

        for i, stage in enumerate(self.back_projection_stages):
            # single-stage model
            if self.num_stage == 1:
                concat_h, kernel_pred = stage(init_f, x, init_f, init_kernel, iter)
            # first stage
            elif i == 0:
                l, x, init_feat, kernel, concat_h, concat_l = stage(init_f, x, init_f, init_kernel, iter)
            # final stage
            elif i + 1 == self.num_stage:
                concat_h, kernel_pred = stage(l, x, init_feat, kernel, iter, concat_h, concat_l)
            # middle stage
            else:
                l, x, init_feat, kernel, concat_h, concat_l = stage(l, x, init_feat, kernel, iter, concat_h, concat_l)

        sr = self.output_conv(concat_h)

        if self.residual_learning:
            sr = sr + self.upsample(x)
            return sr, kernel_pred
        else:
            return sr, kernel_pred

    def _pretrain_check(self, iter):
        """ 
        During the SR pre-training, the data does not pass through self.predictor in forward, 
        and the gradient request is controlled in self.back_projection_stages, 
        so this function does not require gradient control.
        """
        if (self.kernel_pretrain_iter[0] <= iter < self.kernel_pretrain_iter[1]) and self.train_flag['SR']:
            print('Train only kernel_predictor....')
            modules_bps = self._pick_sr_layers()
            modules = [self.feat, *modules_bps, self.output_conv]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            # print(self.train_flag['SR'])
            self.train_flag['SR'] = False
            # print(self.train_flag['SR'])

        if (iter >= self.kernel_pretrain_iter[1] - 1) and not self.train_flag['SR']:
            print('Finish training only kernel_predictor....')
            modules_bps = self._pick_sr_layers()
            modules = [self.feat, *modules_bps, self.output_conv]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
            self.train_flag['SR'] = True

    def _pick_sr_layers(self):
        n = len(self.back_projection_stages)
        modules_bps = []
        for i in range(n):
            for m in self.back_projection_stages[i].modules():
                in_words = lambda x:x in m.__class__.__name__
                if any(map(in_words, ("UpBlock", "DownBlock" , "SFTlayer"))):
                    modules_bps.append(m)

        modules = [self.feat, *modules_bps, self.output_conv]  
        
        return modules 

class KernelBackProjectionStageWithSFT(nn.Module):
    def __init__(self, md_ch, conv_k_sz, stride, padding, cfg, stages, final_stage=False):
        super(KernelBackProjectionStageWithSFT, self).__init__()
        up_stages = stages - 1 if stages > 1 else 1
        self.up = UpBlock(cfg, md_ch, conv_k_sz, stride, padding, up_stages)
        self.kb = KBlock(cfg, md_ch, conv_k_sz, stride, padding, stages)
        self.final_stage = final_stage
        self.num_filter_output = cfg.BLUR.KERNEL_SIZE_OUTPUT ** 2
        self.kernel_sft = cfg.MODEL.KBPN_KERNEL_SFT
        self.sum_lr_error_pos = cfg.MODEL.SUM_LR_ERROR_POS
        if not final_stage:
            self.down = DownBlock(cfg, md_ch, conv_k_sz, stride, padding, stages)
            if self.kernel_sft:
                self.sft = SFTlayer(base_feature_ch=md_ch, condition_ch=self.num_filter_output, stage=stages, weight_norm=False) # default: condition_ch = cfg.BLUR.KERNEL_SIZE**2

    def forward(self, low, x, init_feat, kernel, iter, concat_h=None, concat_l=None):
        h = self.up(low)
        pre_concat_h = torch.cat((concat_h, h), dim=1) if concat_h != None else h
        if self.sum_lr_error_pos == 'HR':
            h, kernel = self.kb(pre_concat_h, h, x, init_feat, kernel, iter)
        elif self.sum_lr_error_pos == 'LR':
            h, error_feat, kernel = self.kb(pre_concat_h, h, x, init_feat, kernel, iter)
        concat_h = torch.cat((concat_h, h), dim=1) if concat_h != None else h

        if self.final_stage:
            return concat_h, kernel
        else:
            low = self.down(concat_h)
            if self.sum_lr_error_pos == 'LR':
                low = low + error_feat
            concat_l = torch.cat((concat_l, low), dim=1) if concat_l != None else low
            low = self.sft(concat_l, kernel) if self.kernel_sft else concat_l
            return low, x, init_feat, kernel, concat_h, concat_l


class BlockBase(nn.Module):
    def __init__(self, output_dim, bias, activation, normalization):
        super(BlockBase, self).__init__()
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation
        self.normalization = normalization

    def create_block(self):
        # ## Nomalizing layer
        if self.normalization == 'batch':
            self.norm = nn.BatchNorm2d(self.output_dim)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(self.output_dim)
        elif self.normalization == 'group':
            self.norm = nn.GroupNorm(32, self.output_dim)
        elif self.normalization == 'spectral':
            self.norm = None
            self.layer = nn.utils.spectral_norm(self.layer)
        elif self.normalization == None:
            self.norm = None
        
        ### Activation layer
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU(init=0.01)
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.01, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif self.activation == None:
            self.act = None

        ### Initialize weights
        if self.activation == 'relu':
            nn.init.kaiming_normal_(self.layer.weight, nonlinearity='relu')
        elif self.activation == 'prelu' or self.activation == 'lrelu':
            nn.init.kaiming_normal_(self.layer.weight, a=0.01, nonlinearity='leaky_relu')
        elif self.activation == 'tanh':
            nn.init.xavier_normal_(self.layer.weight, gain=5/3)
        else:
            nn.init.xavier_normal_(self.layer.weight, gain=1)
        if self.bias:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.act is not None:
            x = self.act(x)      
        
        return x


class DenseBlock(BlockBase):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.Linear(input_dim, output_dim, bias=bias)

        # ## Overwrite normalizing layer for 1D version
        self.norm = normalization
        if self.norm == 'batch':
            self.norm = nn.BatchNorm1d(output_dim)
        elif self.norm == 'instance':
            self.norm = nn.InstanceNorm1d(output_dim)
        self.create_block()


class ConvBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.create_block()


class DeconvBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.create_block()


class ConvAndPixelShuffleBlock(BlockBase):
    def __init__(self, input_dim, output_dim, factor, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        self.layer = nn.Conv2d(input_dim, output_dim * factor ** 2, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)
        self.ps = nn.PixelShuffle(factor) # writing
        self.create_block()
    
    def forward(self, x):
        x = super().forward(x)
        return self.ps(x)


class predictor_withGAP(nn.Module):
    def __init__(self, input_ch, output_ch, stride, padding, ksize_output, norm=None, weight_norm=False, activation='prelu'):
        super(predictor_withGAP, self).__init__()
        '''
        self.conv1 = nn.Conv2d(3, feature_ch, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(feature_ch, feature_ch, kernel_size=3, padding=padding)
        self.conv3 = nn.Conv2d(feature_ch, feature_ch, kernel_size=3, padding=padding)
        self.conv4 = nn.Conv2d(feature_ch, feature_ch, kernel_size=3, padding=padding)
        self.GAP    = nn.AdaptiveAvgPool2d(1)
        if weight_norm:
            self.conv1 = torch.nn.utils.weight_norm(self.conv1)
            self.conv2 = torch.nn.utils.weight_norm(self.conv2)
            self.conv3 = torch.nn.utils.weight_norm(self.conv3)
            self.conv4 = torch.nn.utils.weight_norm(self.conv4)
        '''

        n = 3
        feat_ext = [ConvBlock(input_ch, input_ch, 3, 1, 1, activation=activation, normalization=norm) for i in range(n-1)] +\
                    [ConvBlock(input_ch, output_ch, 3, 1, 1, activation=activation, normalization=norm)]
        self.feat_ext = nn.Sequential(*feat_ext)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        estimate_ksize = int(output_ch ** 0.5)
        if ksize_output != estimate_ksize:
            self.upsample_flag = True
            self.upsample = nn.Upsample(size=ksize_output, mode='bicubic')
        else:
            self.upsample_flag = False
        
    def forward(self, x):
        shape = x.shape
        z = self.feat_ext(x)
        vec = self.GAP(z)
        if self.upsample_flag:
            vec = self.upscale_and_reshape(vec)
        else:
            vec = vec / ((vec.sum(dim=1)).view(vec.shape[0], 1, 1, 1))
            # vec = F.softmax(vec, dim=1)

        H = vec.expand(vec.shape[0], vec.shape[1], shape[2], shape[3])

        return H, vec

    def upscale_and_reshape(self, vec):
        ksize = int(vec.shape[1] ** 0.5)
        kernel = vec.view(vec.shape[0], 1, ksize, ksize)
        kernel = self.upsample(kernel)
        kernel = kernel / kernel.sum(dim=(2, 3), keepdim=True)
        kernel = kernel.view(kernel.shape[0], kernel.shape[2] ** 2, 1, 1)
        return kernel


class KBlock(nn.Module):
    def __init__(self, cfg, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None, weight_norm=False):
        super(KBlock, self).__init__()
        # num_filter = base_filter = 64 = channel
        ## parameters
        self.ksize = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.down_scale = 1 / self.scale_factor
        self.pad = (self.ksize - 1) // 2
        self.sr_pretrain_iter = cfg.SOLVER.SR_SR_MODULE_PRETRAIN_ITER
        self.kernel_pretrain_iter = cfg.SOLVER.SR_KERNEL_MODULE_PRETRAIN_ITER
        self.sum_lr_error_pos = cfg.MODEL.SUM_LR_ERROR_POS
        self.num_filter_ikc = cfg.BLUR.KERNEL_SIZE ** 2
        self.num_stages = num_stages

        ## build module
        self.sr_reconst = ConvBlock(num_stages * num_filter, 3, 3, 1, 1, activation=None, normalization=norm)
        self.kernel_predictor = KernelPredictorLikeIKC(cfg, kernel_ch=self.num_filter_ikc, feat_ch=num_filter, padding=1)
        
        params = 0
        for p in self.kernel_predictor.parameters():
            if p.requires_grad:
                params += p.numel()
        print('kpred:', params) 

        self.GAP = nn.AdaptiveAvgPool2d(1)

        if self.sum_lr_error_pos == 'HR':
            if cfg.MODEL.SR_PIXEL_SHUFFLE:
                self.up_conv1 = ConvAndPixelShuffleBlock(3, num_filter, self.scale_factor,  3, 1, 1, activation=activation, normalization=norm)
            else:
                self.up_conv1 = DeconvBlock(3, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)
        elif self.sum_lr_error_pos == 'LR':
            self.conv = ConvBlock(3, num_filter, 3, 1, 1, activation=None, normalization=norm)

        # pretrain flag
        self.train_flag = {"SR": True, "kernel": True}

    def forward(self, concat_h, h, input_lr, lr_f0, d_kernel, iter):
        ## reconstruct SR image
        self._pretrain_check(iter)
        sr_t = self.sr_reconst(concat_h)
        if  not (self.sr_pretrain_iter[0] <= iter < self.sr_pretrain_iter[1]):
            # print('kernel predict...')
            d_kernel = self.kernel_predictor(sr_t, d_kernel)

        ## predict pseudo LR image
        vec_ = self.GAP(d_kernel)
        vec = vec_ / ((vec_.sum(dim=1)).view(vec_.shape[0], 1, 1, 1))

        weight = vec.view(vec.shape[0], 1, self.ksize, self.ksize)
        for k in range(sr_t.shape[0]):
            tmp_weight = weight[k].expand(sr_t.shape[1], 1, self.ksize, self.ksize) #.flip([2,3])
            tmp = F.conv2d(sr_t[k, :].view(1, sr_t.shape[1], sr_t.shape[2], sr_t.shape[3]), tmp_weight, stride=self.scale_factor, padding=self.pad, groups=sr_t.shape[1])

            if k == 0:
                pseudo_lr = tmp
            else:
                pseudo_lr = torch.cat((pseudo_lr, tmp), dim=0)

        ## back-projection
        d_kernel = vec.expand(vec.shape[0], vec.shape[1], pseudo_lr.shape[2], pseudo_lr.shape[3])
        error = pseudo_lr - input_lr
        if self.sum_lr_error_pos == 'HR':
            e_h = self.up_conv1(error)
            return h + e_h, d_kernel
        elif self.sum_lr_error_pos == 'LR':
            error_feat = self.conv(error)
            return h, error_feat, d_kernel

    def _pretrain_check(self, iter):
        if (self.sr_pretrain_iter[0] <= iter < self.sr_pretrain_iter[1]) and self.train_flag["kernel"]:
            print(f'KBlock in stage{self.num_stages}: Parameters in kernel predictor don\'t require grad because SR PRETRAINING. ')
            for param in self.kernel_predictor.parameters():
                param.requires_grad = False
            self.train_flag["kernel"] = False
        
        elif (iter >= self.sr_pretrain_iter[1]) and not self.train_flag["kernel"]:
            print(f'KBlock in stage{self.num_stages}: Finish SR PRETRAINING. ')
            for param in self.kernel_predictor.parameters():
                param.requires_grad = True
            self.train_flag["kernel"] = True

        if (self.kernel_pretrain_iter[0] <= iter < self.kernel_pretrain_iter[1]) and self.train_flag["SR"]:
            print(f'KBlock in stage{self.num_stages}: Parameters in SR reconst modules don\'t require grad because kernel predictor PRETRAINING. ')
            if self.sum_lr_error_pos == 'HR':
                modules = [self.sr_reconst, self.up_conv1]
            elif self.sum_lr_error_pos == 'LR':
                modules = [self.sr_reconst, self.conv]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            self.train_flag["SR"] = False

        elif (iter >= self.kernel_pretrain_iter[1]) and not self.train_flag["SR"]:
            print(f'KBlock in stage{self.num_stages}: Finish training only kernel_predictor....')
            if self.sum_lr_error_pos == 'HR':
                modules = [self.sr_reconst, self.up_conv1]
            elif self.sum_lr_error_pos == 'LR':
                modules = [self.sr_reconst, self.conv]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = True
            self.train_flag["SR"] = True


class UpBlock(torch.nn.Module):
    def __init__(self, cfg, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        # num_filter = base_filter = 64 = channel
        scale_factor = cfg.MODEL.SCALE_FACTOR
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, normalization=norm)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)
        if cfg.MODEL.SR_PIXEL_SHUFFLE:
            self.up_conv1 = ConvAndPixelShuffleBlock(num_filter, num_filter, scale_factor, 3, 1, 1, activation=activation, normalization=norm)
            self.up_conv3 = ConvAndPixelShuffleBlock(num_filter, num_filter, scale_factor, 3, 1, 1, activation=activation, normalization=norm)
        else:
            self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)
            self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, cfg, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        scale_factor = cfg.MODEL.SCALE_FACTOR
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, normalization=norm)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)
        if cfg.MODEL.SR_PIXEL_SHUFFLE:
            self.down_conv2 = ConvAndPixelShuffleBlock(num_filter, num_filter, scale_factor, 3, 1, 1, activation=activation, normalization=norm)
        else:
            self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, normalization=norm)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


# 書き換えする
class SFTlayer(torch.nn.Module):
    def __init__(self, base_feature_ch, condition_ch, stage, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, weight_norm=False):
        super(SFTlayer, self).__init__()
        # 
        concatted_ch = stage*base_feature_ch + condition_ch

        self.SFT_scale_conv0 = torch.nn.Conv2d(concatted_ch, concatted_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.SFT_scale_conv1 = torch.nn.Conv2d(concatted_ch, stage*base_feature_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.SFT_shift_conv0 = torch.nn.Conv2d(concatted_ch, concatted_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.SFT_shift_conv1 = torch.nn.Conv2d(concatted_ch, stage*base_feature_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.sigmoid         = torch.nn.Sigmoid()
        if weight_norm:
            self.SFT_scale_conv0 = torch.nn.utils.weight_norm(self.SFT_scale_conv0)
            self.SFT_scale_conv1 = torch.nn.utils.weight_norm(self.SFT_scale_conv1)
            self.SFT_shift_conv0 = torch.nn.utils.weight_norm(self.SFT_shift_conv0)
            self.SFT_shift_conv1 = torch.nn.utils.weight_norm(self.SFT_shift_conv1)


    def forward(self, features, conditions):
        # features are got from Main SR stream, conditions are got from condition stream
        concatted = torch.cat((features, conditions), 1) 

        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(concatted), 0.1, inplace=True))
        scale = self.sigmoid(scale)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(concatted), 0.1, inplace=True))
        return features * scale  + shift


class KernelPredictorLikeIKC(torch.nn.Module):
    def __init__(self, cfg, kernel_ch, feat_ch, padding, stage=1, use_sr_img=False, use_pre_kernel=True, subtraction='square', upscale_factor=4, norm=None):
        super(KernelPredictorLikeIKC, self).__init__()
        ## When you use the 'pre-kernel' estimated in pre-stage, we concatenate present & previous kernels(dim=1)
        reduction_ch = 32
        estimate_ksize = cfg.BLUR.KERNEL_SIZE
        self.ksize_output = cfg.BLUR.KERNEL_SIZE_OUTPUT
        self.fe_SR = nn.Sequential(ConvBlock(3, kernel_ch, kernel_size=3, padding=padding, normalization=norm),
                                    ConvBlock(kernel_ch, reduction_ch, kernel_size=1, padding=0, activation='lrelu', normalization=norm),
                                    ConvBlock(reduction_ch, reduction_ch, kernel_size=3, padding=padding, activation='lrelu', normalization=norm),
                                    ConvBlock(reduction_ch, reduction_ch, kernel_size=3, padding=padding, activation='lrelu', normalization=norm),
                                    ConvBlock(reduction_ch, kernel_ch, kernel_size=3, padding=padding, activation='lrelu', normalization=norm))
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fe_kernel = nn.Sequential(ConvBlock(self.ksize_output ** 2, kernel_ch, kernel_size=3, padding=padding, activation='lrelu', normalization=norm),
                                        ConvBlock(kernel_ch, kernel_ch, kernel_size=3, padding=padding, activation='lrelu', normalization=norm))

        final_act = None # 'sigmoid', None
        # print(final_act)
        self.fe_cat = nn.Sequential(ConvBlock(2 * kernel_ch, reduction_ch, kernel_size=1, padding=0, activation='lrelu', normalization=norm),
                                    ConvBlock(reduction_ch, reduction_ch, kernel_size=3, padding=1, activation='lrelu', normalization=norm),
                                    ConvBlock(reduction_ch, kernel_ch, kernel_size=3, padding=1, activation=final_act, normalization=norm))  

        self.zero_pad_kernel = cfg.MODEL.ZERO_PAD_KERNEL
        if self.zero_pad_kernel:
            self.pad_descriminator = nn.Sequential(nn.Linear(estimate_ksize ** 2, 8),
                                                   nn.ReLU(),
                                                   nn.Dropout(0.2),
                                                   nn.Linear(8, 8),
                                                   nn.ReLU(),
                                                   nn.Dropout(0.2),
                                                   nn.Linear(8, 1),
                                                   nn.Sigmoid()
                                                   )
            self.zero_pad = nn.ZeroPad2d(int((self.ksize_output - estimate_ksize) / 2))
        
        if self.ksize_output != estimate_ksize:
            self.upsample_flag = True
            self.upsample = nn.Upsample(size=self.ksize_output, mode='bicubic')
        else:
            self.upsample_flag = False

    def forward(self, SR, pre_kernel=None):
        fsr = self.fe_SR(SR)

        fh = self.GAP(pre_kernel)
        fh = fh.expand(fh.shape[0], fh.shape[1], fsr.shape[2], fsr.shape[3])
        fh = self.fe_kernel(fh)

        fcat = torch.cat((fsr, fh), dim=1)
        delta = self.fe_cat(fcat)

        delta = self.GAP(delta)
        if self.upsample_flag:
            delta = self.upscale_and_reshape(delta)
        delta = delta.expand(pre_kernel.shape)
        # print(final_act)

        return pre_kernel + delta

    def upscale_and_reshape(self, vec):
        ksize = int(vec.shape[1] ** 0.5)
        kernel = vec.view(vec.shape[0], 1, ksize, ksize)
        if self.zero_pad_kernel:
            vec = vec.view(vec.shape[0], -1)
            p = self.pad_descriminator(vec)
            _kernel = []
            
            for kernel_i, p_i in zip(kernel, p): 
                kernel_i = kernel_i.unsqueeze(0)
                if p_i.item() >= 0.5:
                    kernel_i = self.upsample(kernel_i)
                    kernel_i = kernel_i.view(1, kernel_i.shape[2] ** 2, 1, 1)  
                else:
                    kernel_i = self.zero_pad(kernel_i)
                    kernel_i = kernel_i.view(1, self.ksize_output ** 2, 1, 1)
                _kernel.append(kernel_i)
            _kernel = torch.cat(_kernel, dim=0)
        else:
            kernel = self.upsample(kernel)
            _kernel = kernel.view(kernel.shape[0], kernel.shape[2] ** 2, 1, 1)

        return _kernel



# class kernel_convolution():
#     def __init__(self, version, ksize, scale_factor):
#         self.version = version
#         self.GAP = nn.AdaptiveAvgPool2d(1)
#         self.ksize = ksize
#         self.pad = (ksize - 1) // 2
#         self.scale_factor = scale_factor

#     def __call__(self, img, pre_kernel):
#         if self.version=='withGAP':
#             ## GAP version. Non region dependent
#             vec_kernel = self.GAP(pre_kernel)
#             vec_kernel = F.softmax(vec_kernel, dim=1)
#             weight = vec_kernel.view(vec_kernel.shape[0], 1, self.ksize, self.ksize)
#             for b in range(img.shape[0]):
#                 tmp_weight = weight[b].expand(img.shape[1], 1, self.ksize, self.ksize).flip([2,3])
#                 tmp = F.conv2d(img[b,:].view(1, img.shape[1], img.shape[2], img.shape[3]), tmp_weight, stride=self.scale_factor, padding=self.pad, groups=img.shape[1])
#                 if b == 0:
#                     pseudo_lr = tmp
#                 else:
#                     pseudo_lr = torch.cat((pseudo_lr, tmp), dim=0)
#         else:
#             Img = F.pad(img, (self.pad, self.pad, self.pad, self.pad))
#             if self.version=='v1':
#                 ## Region dependent version
#                 for b in range(pre_kernel.shape[0]):
#                     for i in range(pre_kernel.shape[2]):
#                         for j in range(pre_kernel.shape[3]):
#                             tmp_kernel = pre_kernel[b, :, i, j].view(1, 1, self.ksize, self.ksize)
#                             tmp_kernel = pre_kernel.expand(Img.shape[1], 1, self.ksize, self.ksize) #.flip([2, 3])
#                             tmp_img = Img[k, :, self.scale_factor*i:self.scale_factor*i + self.ksize, self.scale_factor*j:self.scale_factor*j + self.ksize]
#                             tmp_lr = F.conv2d(tmp_sr.view(1, SR_T.shape[1], self.ksize, self.ksize), tmp_kernel, groups=Img.shape[1])
#                             if j == 0:
#                                 row_lr = tmp_lr
#                             else:
#                                 row_lr = torch.cat((row_lr, tmp_lr),dim=3)
#                         if i == 0:
#                             lr = row_lr
#                         else:
#                             lr = torch.cat((lr, row_lr), dim=2)
#                     if b == 0:
#                         pseudo_lr = lr
#                     else:
#                         pseudo_lr = torch.cat((pseudo_lr, lr), dim=0)

#             else:
#                 for b in range(pre_kernel.shape[0]):
#                     for i in range(pre_kernel.shape[2]):
#                         for j in range(pre_kernel.shape[3]):
#                             tmp_kernel = pre_kernel[b, :, i, j].view(1, 1, self.ksize, self.ksize)
#                             tmp_kernel = pre_kernel.expand(Img.shape[1], 1, self.ksize, self.ksize).flip([2, 3])
#                             tmp_lr = F.conv2d(Img[b].view(1, Img.shape[1], Img.shape[2], Img.shape[3]), tmp_kernel, groups=Img.shape[1])
#                             if j == 0:
#                                 row_lr = tmp_lr[:, :, self.scale_factor*i, self.scale_factor*j].view(1, tmp_lr.shape[1], 1, 1)
#                             else:
#                                 row_lr = torch.cat((row_lr, tmp_lr[:, :, self.scale_factor*i, self.scale_factor*j].view(1, tmp_lr.shape[1], 1, 1)), dim=3)
#                         if i==0:
#                             lr = row_lr
#                         else:
#                             lr = torch.cat((lr, row_lr), dim=2)
#                     if b == 0:
#                         pseudo_lr = lr
#                     else:
#                         pseudo_lr = torch.cat((lrs, lr), dim=0)
            

#         return pseudo_lr


# def set_activation(key):
#     act = key
#     if key == 'relu':
#         act = torch.nn.ReLU(True)
#     elif key == 'prelu':
#         act = torch.nn.PReLU()
#     elif key == 'lrelu':
#         act = torch.nn.LeakyReLU(0.2, True)
#     elif key == 'tanh':
#         act = torch.nn.Tanh()
#     elif key == 'sigmoid':
#         act = torch.nn.Sigmoid()
#     return act
