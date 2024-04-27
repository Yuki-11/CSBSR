##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import math
import torch
import torch.nn as nn


class BlockBase(nn.Module):
    def __init__(self, output_dim, bias, activation, normalization):
        super(BlockBase, self).__init__()
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation
        self.normalization = normalization

    def create_block(self):
        ### Nomalizing layer
        if self.normalization =='batch':
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

        ### Overwrite normalizing layer for 1D version
        self.norm = normalization
        if self.norm =='batch':
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


class SFTLikeBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(SFTLikeBlock, self).__init__(output_dim, bias, activation, norm)
        self.conv_scale = nn.Sequential(*[ConvBlock(input_dim, input_dim, kernel_size, stride, padding, bias=bias, activation=activation, normalization=norm),
                                        ConvBlock(input_dim, output_dim, kernel_size, stride, padding, bias=bias, activation='sigmoid', normalization=norm),
                                        ])
        self.conv_shift = nn.Sequential(*[ConvBlock(input_dim, input_dim, kernel_size, stride, padding, bias=bias, activation=activation, normalization=norm),
                                        ConvBlock(input_dim, output_dim, kernel_size, stride, padding, bias=bias, activation=None, normalization=norm),
                                        ])

    def forward(self, features, conditions):
        # features are got from Main SR stream, conditions are got from condition stream
        concatted = torch.cat((features, conditions), 1) 
        scale = self.conv_scale(concatted)
        shift = self.conv_shift(concatted)
        return features * scale  + shift
    

class SFTBlock(BlockBase):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(SFTBlock, self).__init__(output_dim, bias, activation, norm)
        self.conv_scale = nn.Sequential(*[ConvBlock(input_dim, input_dim, kernel_size, stride, padding, bias=bias, activation=activation, normalization=norm),
                                        ConvBlock(input_dim, output_dim, kernel_size, stride, padding, bias=bias, activation='sigmoid', normalization=norm),
                                        ])
        self.conv_shift = nn.Sequential(*[ConvBlock(input_dim, input_dim, kernel_size, stride, padding, bias=bias, activation=activation, normalization=norm),
                                        ConvBlock(input_dim, output_dim, kernel_size, stride, padding, bias=bias, activation=None, normalization=norm),
                                        ])

    def forward(self, features, conditions):
        # features are got from Main SR stream, conditions are got from condition stream
        scale = self.conv_scale(conditions)
        shift = self.conv_shift(conditions)
        return features * scale  + shift


class DeformableConvBlock(BlockBase):
    def __init__(self, input_dim, output_dim, offset_dim=None, kernel_size=3, stride=1, padding=1, deform_groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        
        if offset_dim is None:
            offset_dim = input_dim
        
        self.layer = DeformConv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias)
        self.offset_conv = nn.Conv2d(offset_dim, deform_groups * 2 * kernel_size**2, kernel_size, stride, padding, bias=True)
            
        self.create_block()
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

    def forward(self, x, offset=None):
        if offset is None:
            offset = self.offset_conv(x)
        else:
            offset = self.offset_conv(offset)
        x = self.layer(x, offset)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)      
        return x


class ModulatedDeformableBlock(BlockBase):
    def __init__(self, input_dim, output_dim, offset_dim=None, kernel_size=3, stride=1, padding=1, deform_groups=1, bias=False, activation='relu', normalization='batch'):
        super().__init__(output_dim, bias, activation, normalization)
        if offset_dim is None:
            offset_dim = input_dim
        
        self.layer = ModulatedDeformConv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias)
        self.offset_conv = nn.Conv2d(offset_dim, deform_groups * 3 * kernel_size**2, kernel_size, stride, padding, bias=True)
            
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
        self.create_block()


    def forward(self, x, offset=None):
        if offset is None:
            o1, o2, mask = torch.chunk(self.offset_conv(x), 3, dim=1)
        else:
            o1, o2, mask = torch.chunk(self.offset_conv(offset), 3, dim=1)

        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        x = self.layer(x, offset, mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)
            
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_convs=2, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', normalization='batch'):
        super(ResidualBlock, self).__init__()

        self.num_convs = num_convs

        self.layers, self.norms, self.acts = [], [], []
        self.layers.append(nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias))
        for _ in range(num_convs - 1):
            self.layers.append(nn.Conv2d(output_dim, output_dim, kernel_size, 1, padding, bias=bias))

        #TODO: Initialization for skip_layer is not implemented
        if input_dim != output_dim or stride != 1:
            self.skip_layer = [nn.Conv2d(input_dim, output_dim, 1, stride, 0, bias=bias)]
            if bias:
                nn.init.zeros_(self.skip_layer[0].bias)
        else:
            self.skip_layer = None


        for i in range(num_convs):
            ### Normalizing layer
            if normalization == 'batch':
                self.norms.append(nn.BatchNorm2d(output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.BatchNorm2d(output_dim))
            elif normalization == 'instance':
                self.norms.append(nn.InstansNorm2d(output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.InstansNorm2d(output_dim))
            elif normalization == 'group':
                self.norms.append(nn.GroupNorm(32, output_dim))
                if self.skip_layer is not None:
                    self.skip_layer.append(nn.GroupNorm(32, output_dim))
            elif normalization == 'spectral':
                self.norms.append(None)
                self.layers[i] = nn.utils.spectral_norm(self.layers[i])
                if self.skip_layer is not None:
                    self.skip_layer[0] = nn.utils.spectral_norm(self.skip_layer[0])
            elif normalization == None:
                self.norms.append(None)
            else:
                raise Exception('normalization={} is not implemented.'.format(normalization))

            ### Activation layer
            if activation == 'relu':
                self.acts.append(nn.ReLU(True))
            elif activation == 'lrelu':
                self.acts.append(nn.LeakyReLU(0.01, True))
            elif activation == 'prelu':
                self.acts.append(nn.PReLU(init=0.01))
            elif activation == 'tanh':
                self.acts.append(nn.Tanh())
                self.acts.append(nn.Sigmoid())
            elif activation == None:
                self.acts.append(None)
            else:
                raise Exception('activation={} is not implemented.'.format(activation))

            ### Initialize weights
            if activation == 'relu':
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity='relu')
            elif activation == 'lrelu' or activation == 'prelu':
                nn.init.kaiming_normal_(self.layers[i].weight, a=0.01, nonlinearity='leaky_relu')
            elif activation == 'tanh':
                nn.init.xavier_normal_(self.layers[i].weight, gain=5/3)
            elif activation == 'sigmoid':
                nn.init.xavier_normal_(self.layers[i].weight, gain=1)
            elif activation == None:
                nn.init.xaview_normal_(self.layers[i].weight, gain=1)
            else:
                raise Exception('activation={} is not implemented.'.format(activation))

            if bias:
                nn.init.zeros_(self.layers[i].bias)    

        self.layers = nn.ModuleList(self.layers)
        self.norms = nn.ModuleList(self.norms)
        self.acts = nn.ModuleList(self.acts)
        if self.skip_layer is not None:
            self.skip_layer = nn.Sequential(*self.skip_layer)

        self.cutoff = (math.floor(kernel_size/2) - padding) * num_convs

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        for i in range(self.num_convs):
            x = self.layers[i](x)

            if self.norms[i] is not None:
                x = self.norms[i](x)

            if i == self.num_convs - 1:
                if self.cutoff == 0:
                    x = x + residual
                else:
                    x = x + residual[:, :, self.cutoff:-self.cutoff, self.cutoff:-self.cutoff]
            
            if self.acts[i] is not None:
                x = self.acts[i](x)
        return x