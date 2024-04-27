# Original code: https://github.com/Lextal/pspnet-pytorch
# Reference; https://github.com/hszhao/semseg/blob/master/model/pspnet.py
# from threading import enumerate
import torch
from torch import nn
from torch.nn import functional as F

from model.modeling.pspnet_pytorch import extractors
from model.modeling.blocks import SFTLikeBlock, ConvBlock, SFTBlock



# models = {
#     'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
#     'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
#     'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
#     'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
#     'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
#     'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
#     'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
# }

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        # p = F.upsample(input=x, size=(h, w), mode='bilinear')
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',
                pretrained=True, zoom_factor=8, dropout=0.1,):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(deep_features_size, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, n_classes, kernel_size=1), 
                nn.Sigmoid()
            )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )


    def forward(self, x):
        x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # print(x_size)
        h, w = x_size[2:]
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        f, auxiliary = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        # if self.training:
        if self.zoom_factor != 1:
            auxiliary = self.aux(auxiliary)
            # auxiliary = F.upsample(auxiliary, size=(h, w), mode='bilinear', align_corners=True)
            auxiliary = F.interpolate(auxiliary, size=(h, w), mode='bilinear', align_corners=True)
        return self.final(p), auxiliary
        # else: 
        #     return self.final(p)

class PSPNet_BlurSkip(nn.Module):
    def __init__(self, blur_dim, n_classes=18, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34',
                pretrained=True, zoom_factor=8, dropout=0.1, n_layer_blurskip=2, modify_blur_skip=True):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)

        reduction_dim = blur_dim # 32
        self.GAP = nn.AdaptiveAvgPool2d(1)
        # self.conv_kernel = ConvBlock(blur_dim, reduction_dim)
        if modify_blur_skip:
            mod_list = [[SFTLikeBlock(reduction_dim + 64, 64), ConvBlock(64, 64)] for _ in range(n_layer_blurskip)]
        else:
            mod_list = [[SFTBlock(reduction_dim, 64), ConvBlock(64, 64)] for _ in range(n_layer_blurskip)]
        mod_list = sum(mod_list, [])
        
        self.blur_skip = nn.ModuleList(mod_list)

        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Sigmoid()
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(deep_features_size, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, n_classes, kernel_size=1), 
                nn.Sigmoid()
            )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )


    def forward(self, x, kernel_preds):
        x_size = x.size()
        h, w = x_size[2:]
        _kernel_preds = self.GAP(kernel_preds)
        _kernel_preds = _kernel_preds.expand(*_kernel_preds.shape[:2], h, w) 

        f, auxiliary = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        _p = p.clone()

        # kernel_preds = self.conv_kernel()
        for i, block in enumerate(self.blur_skip):
            if i % 2 == 0:  # SFTLikeBlock
                _p = block(_p, _kernel_preds)
            else:           # ConvBlock
                _p = block(_p)
        p = p + _p

        if self.zoom_factor != 1:
            auxiliary = self.aux(auxiliary)
            # auxiliary = F.upsample(auxiliary, size=(h, w), mode='bilinear', align_corners=True)
            auxiliary = F.interpolate(auxiliary, size=(h, w), mode='bilinear', align_corners=True)
        
        return self.final(p), auxiliary