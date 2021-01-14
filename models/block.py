from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import block as B
from . import spectral_norm as SN
import torch.nn.utils.spectral_norm as spectral_norm

####################
# Basic blocks
####################
import os

class AtMp(nn.Module):
  
    def __init__(self, n_features, reduction=4):
        super(AtMp, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction       (default = 16)')

        self.map1 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(n_features)

        self.map2 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(n_features)

        self.map21 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn21 = nn.BatchNorm2d(n_features)

        self.nonlin2 = nn.ReLU(inplace=True)


        self.map3 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(n_features)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.map4 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn4 = nn.BatchNorm2d(n_features)
        self.nonlin4 = nn.ReLU(inplace=True)

        self.atmap1 = nn.Conv2d(n_features, int(n_features/reduction), kernel_size=1, stride=1, padding=0,bias=False)
        self.bnat1 = nn.BatchNorm2d(int(n_features/reduction))
        self.nonlinat1 = nn.ReLU(inplace=True)
        self.atmap2 = nn.Conv2d(int(n_features/reduction), 1, kernel_size=1, stride=1, padding=0,bias=False)

        self.atmap3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0,bias=False)
       
        self.nonliner = nn.Sigmoid()
    
    def forward(self, x, z):    
        bs = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        a = x
        y = x
        x = self.map1(x)
        x = self.bn1(x)
        
        y = self.map2(y)
        y = self.bn2(y)
      
        x = x * y 
       
        z = self.map21(z)
        z = self.bn21(z)

        x = x + z
        x = self.nonlin2(x)

        y = x
        x = self.map3(x)
        x = self.bn3(x)
        x = self.nonlin3(x)
 
        y = self.map4(y)
        y = self.bn4(y)
        y = self.nonlin4(y)

        x = x * y

        x = self.atmap1(x)
        x = self.bnat1(x)
        x = self.nonlinat1(x)
        x = self.atmap2(x)
        
        y = x
        x = self.atmap3(x)
        x = y + x

        x = self.nonliner(x)
        x = x.view(bs,1,h,w).repeat(1,dim,1,1)
        x = x * a
        return x

class AtUni(nn.Module):
  
    def __init__(self, n_features, reduction=4):
        super(AtUni, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction       (default = 16)')

        self.map1 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(n_features)
        self.map2 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(n_features)
        self.map21 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn21 = nn.BatchNorm2d(n_features)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.map3 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(n_features)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.map4 = nn.Conv2d(n_features, n_features, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn4 = nn.BatchNorm2d(n_features)
        self.nonlin4 = nn.ReLU(inplace=True)

        self.atmap1 = nn.Conv2d(n_features, int(n_features/reduction), kernel_size=1, stride=1, padding=0,bias=False)
        self.bnat1 = nn.BatchNorm2d(int(n_features/reduction))
        self.nonlinat1 = nn.ReLU(inplace=True)
        self.atmap2 = nn.Conv2d(int(n_features/reduction), 1, kernel_size=1, stride=1, padding=0,bias=False)

        self.atmap3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0,bias=False)
       
        self.nonliner = nn.Sigmoid()
    
    def forward(self, x):    
        bs = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        a = x
        y = x
        x = self.map1(x)
        x = self.bn1(x)
        
        y = self.map2(y)
        y = self.bn2(y)
      
        x = x * y 
       
        z = self.map21(x)
        z = self.bn21(x)

        x = x + z
        x = self.nonlin2(x)

        y = x
        x = self.map3(x)
        x = self.bn3(x)
        x = self.nonlin3(x)
 
        y = self.map4(y)
        y = self.bn4(y)
        y = self.nonlin4(y)

        x = x * y

        x = self.atmap1(x)
        x = self.bnat1(x)
        x = self.nonlinat1(x)
        x = self.atmap2(x)
        
        y = x
        x = self.atmap3(x)
        x = y + x

        x = self.nonliner(x)
        x = x.view(bs,1,h,w).repeat(1,dim,1,1)
        x = x * a
        return x


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc, num_group=1):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=True)
    elif norm_type == 'layer_norm':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm_type == 'group_norm':
        layer = nn.GroupNorm(num_group,nc, affine=True)
    elif norm_type == 'local_response_norm':
        layer = nn.LocalResponseNorm(num_group)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        return x, self.sub 

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type=None, mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = spectral_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups))
    # c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
    #         dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc, groups) if norm_type else None
        return sequential(n, a, p, c)


def linear_block(in_nc, out_nc, bias=True, norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
   
    c = nn.Linear(in_nc, out_nc, bias=bias)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc, groups) if norm_type else None
        return sequential(n, a, c)

class bilinear_block(nn.Module):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''

    def __init__(self,in1_nc, in2_nc, out_nc, bias=True, norm_type=None, act_type='relu', mode='CNA'):
        super(bilinear_block, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
        
        self.c = nn.Bilinear(in1_nc, in2_nc, out_nc, bias=bias)
        self.a = act(act_type) if act_type else None
        self.norm_type = norm_type
        self.act_type = act_type
        self.mode = mode
        self.in1_nc = in1_nc
        self.in2_nc = in2_nc
        self.out_nc = out_nc

    def forward(self, x):    
                
        if 'CNA' in self.mode:
            n = norm(self.norm_type, self.out_nc) if self.norm_type else None
            # return sequential(c, n, a)
            return self.c(n(self.a(x)))

        elif mode == 'NAC':
            if self.norm_type is None and self.act_type is not None:
                a = act(self.act_type, inplace=False)
                # Important!
                # input----ReLU(inplace)----Conv--+----output
                #        |________________________|
                # inplace ReLU will modify the input, therefore wrong output
            n = norm(self.norm_type, self.in_nc, 1) if self.norm_type else None            
            return n(self.a(self.c(x)))

####################
# Useful blocks
####################

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)       
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)
