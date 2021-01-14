import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from .gaborlayer import GaborConv2d
from . import block as B
from model import common
import torch.nn.functional as F
def make_model(args, parent=False):
    return MODEL(args)


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.tanh = nn.Tanh()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        
        return input * self.tanh(self.scale.to(input.device))


## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size,C,H,W = x.shape

        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)        
        f_div_C = F.softmax(f, dim=-1)        
        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)





## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8,sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat,inter_channels=inter_feat, sub_sample=sub_sample,bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ## divide feature map into 4 part
        batch_size,C,H,W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]


        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return  nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1, dilation=2):
        super(RB, self).__init__()
        modules_body = []
        self.gamma1 = 1.0
        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )
        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y

## Local Residual Group (LRG)
class LRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(LRG, self).__init__()
        ##
        self.rb= nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, \
                                       bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in range(n_resblocks)])
        self.conv_last1 = (conv(n_feat, n_feat, kernel_size))
        self.conv_last2 = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm2d(n_feat)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()        

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)

    def forward(self, x):
        residual = x      
        for i,l in enumerate(self.rb):
            x = l(x) + self.tanh(self.gamma)*residual        
        x = self.conv_last2(x)
        x = x + residual
        return x
        
class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        norm_type=None
        out_nc=3
        gc=32
        in_nc=3
        act_type='leakyrelu'
        mode='CNA'
        upsample_mode='upconv'
        n_resgroups = 20
        nb = 23
        nf = 64
        kernel_size = 3
        layers = [2, 2, 2, 2]
        reduction = 16
        args.n_colors = 3
        args.res_scale = 1
        args.rgb_range = 255
        self.args = args
        upscale = args.scale[0]
        scale = args.scale[0]
        nf = 64
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        n_upscale = int(math.log(upscale, 2))
        n_resblocks = 23
        n_feats = 64
        self.weights_dense = []
        for _ in range(n_resgroups):
            self.weights_dense.append(Scale(1)) 

        self.weights_frechet = []
        for _ in range(6):
            self.weights_frechet.append(Scale(1)) 


        self.fusion1a = Scale(1)
        self.fusion1b = Scale(1)
        self.fusion1c = Scale(1)
        self.fusion2a = Scale(1)
        self.fusion2b = Scale(1)
        self.fusion2c = Scale(1)
        self.fusion2d = Scale(1)
        self.fusion3a = Scale(1)
        self.fusion3b = Scale(1)
        self.fusion3c = Scale(1)
        self.fusion3d = Scale(1)
        self.fusion4a = Scale(1)
        self.fusion4b = Scale(1)
        self.fusion4c = Scale(1)
        self.fusion4d = Scale(1)
        self.fusion5a = Scale(1)
        self.fusion5b = Scale(1)
        self.fusion5c = Scale(1)
        self.fusion5d = Scale(1)
        self.fusion6a = Scale(1)
        self.fusion6b = Scale(1)
        self.fusion6c = Scale(1)
        self.fusion6d = Scale(1)
        self.fusion7a = Scale(1)
        self.fusion7b = Scale(1)
        self.fusion7c = Scale(1)
        self.fusion7d = Scale(1)

        self.sum1 = Scale(1)
        self.sum2 = Scale(1)
        self.sum3 = Scale(1)
        self.sum4 = Scale(1)
        self.internal_sum_fusion1_edge = Scale(1)
        self.internal_sum_fusion1_fea = Scale(1)
        self.internal_sum_fusion1_text = Scale(1)
        self.internal_sum_fusion2_edge = Scale(1)
        self.internal_sum_fusion2_fea = Scale(1)
        self.internal_sum_fusion2_text = Scale(1)
        self.internal_sum_fusion3_edge = Scale(1)
        self.internal_sum_fusion3_fea = Scale(1)
        self.internal_sum_fusion3_text = Scale(1)        
        act = nn.ReLU(inplace=True)

        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]        
        
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        self.HR_conv0_new = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        self.HR_conv1_new = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, self.HR_conv0_new)
        self.b_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        self.texture = B.conv_block(n_feats, nf, kernel_size=3, norm_type=norm_type, act_type=None)

        self.b_concat_1 = B.conv_block(3*nf, nf, kernel_size=1, norm_type=norm_type, act_type = None)
        
        conv=common.default_conv

        self.b_block_1 = nn.ModuleList([LRG(conv, 3*nf, 1, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])
        

        self.b_concat_2 = B.conv_block(4*nf, nf, kernel_size=1, norm_type=norm_type, act_type = None)
        
        self.b_block_2 = nn.ModuleList([LRG(conv, 4*nf, 1, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])

        self.b_concat_3 = B.conv_block(4*nf, nf, kernel_size=1, norm_type=norm_type, act_type = None)
        
        self.b_block_3 = nn.ModuleList([LRG(conv, 4*nf, 1, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])

        self.b_concat_4 = B.conv_block(4*nf, nf, kernel_size=1, norm_type=norm_type, act_type = None)
        
        self.b_block_4 = nn.ModuleList([LRG(conv, 4*nf, 1, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])

        self.b_LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        
        b_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=None)

        self.f_concat = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=None)

        self.f_block = B.RRDB(nf*2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA')

        self.f_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.f_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        modules_psr = [
            conv(args.n_colors, args.n_colors, kernel_size)]

        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_head_fea = [conv(nf//2, n_feats, kernel_size)]
        modules_head_text = [conv(nf//2, n_feats, kernel_size)]
        
        kernel_size = 3

        modules_tail_cat = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        modules_tail_text = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        modules_tail_fea = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.tail_res_LRG = nn.ModuleList([LRG(conv, n_feats, 3, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])

        modules_tail_res = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]      
        modules_tail_res_no_fusion = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        modules_tail_top = [
            conv(3, args.n_colors, 1)]
        down_conv = common.down_conv
        modules_down_sample = [
            common.Downsampler(down_conv, scale, args.n_colors, act=False)]
        modules_down_sample2 = [
            down_conv(args.n_colors, n_feats, kernel_size, padding=1, bias=True),
            down_conv(n_feats, args.n_colors, kernel_size, padding=1, bias=True)]      
        self.tail_psr_no_fusion = nn.Sequential(*modules_tail_res_no_fusion)
        self.tail_psr = nn.Sequential(*modules_psr)
        self.tail_cat2 = nn.Sequential(*modules_tail_cat)
        self.tail_text2 = nn.Sequential(*modules_tail_text)
        self.tail_fea2 = nn.Sequential(*modules_tail_fea)
        self.tail_res = nn.Sequential(*modules_tail_res)

        self.tail_text = nn.ModuleList([LRG(conv, n_feats, 3, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(3)])

        self.tail_cat = nn.ModuleList([LRG(conv, n_feats, 3, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(3)])

        self.tail_fea = nn.ModuleList([LRG(conv, n_feats, 3, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=5) for _ in range(3)])

        self.down_sample = nn.Sequential(*modules_down_sample)

        self.down_sample2 = nn.Sequential(*modules_down_sample2)
        self.tail_top = nn.ModuleList([LRG(conv, 3, 3, reduction, \
                                              act=act,res_scale=args.res_scale, n_resblocks=5) for _ in range(5)])

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_text = nn.Parameter(torch.zeros(1))
        self.gamma_grad = nn.Parameter(torch.zeros(1))
        self.n_resgroups = n_resgroups
        
        self.RG = nn.ModuleList([LRG(conv, n_feats, kernel_size, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        
        self.RG_text = nn.ModuleList([LRG(conv, n_feats, kernel_size, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(5)])

        self.RG_fea = nn.ModuleList([LRG(conv, n_feats, kernel_size, reduction, \
                                              act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in range(5)])

        self.conv_last = conv(n_feats, n_feats, kernel_size)

        self.non_local1 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats//8, reduction=8,sub_sample=False, bn_layer=False)
        self.non_local2 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats//8, reduction=8,sub_sample=False, bn_layer=False)
        self.non_local3 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats//8, reduction=8,sub_sample=False, bn_layer=False)
        

        self.head_san1 = nn.Sequential(*modules_head)
        self.head_san2 = nn.Sequential(*modules_head_text)
        self.head_san3 = nn.Sequential(*modules_head_fea)

        self.gabor_layer1 = GaborConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.gabor_layer2 = GaborConv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        
        self.top1 = Scale(1)
        self.top2 = Scale(1)
        self.top3 = Scale(1)
        self.top4 = Scale(1)
        self.top5 = Scale(1)

        self.top6 = Scale(1)
        self.top7 = Scale(1)

        self.AtMp1 = B.AtUni(3, reduction=1)
        self.AtMp2 = B.AtUni(3, reduction=1)
        self.AtMp3 = B.AtUni(3, reduction=1)
        self.AtMp4 = B.AtUni(3, reduction=1)

        
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)

    def make_layer_attn(self, block, planes, blocks, stride=1, down_size=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


    def forward(self, x):            
        total_loss = 0    
        x = (x - self.rgb_mean.cuda()*255)/127.5
        x_0 = x
        x_texture = self.gabor_layer1(x)
        x_grad = self.gabor_layer2(x)
        edge_dis_loss =0 
        edge_gen_loss = 0
        x = self.head_san1(x)
        x_texture = self.head_san2(x_texture)
        x_grad = self.head_san3(x_grad)        
        xx = self.non_local1(x)        
        x_texture = self.non_local2(x_texture)
        x_grad = self.non_local3(x_grad)
        x_ori_text = x_texture
        x_ori = xx
        x_ori_grad = x_grad

        for i,l in enumerate(self.RG):
            w = self.weights_dense[i]
            xx = l(xx) + w(x_ori)        
            if i==5:
                x_fea1 = xx
                l_text = self.RG_text[1]
                x_texture = l_text(x_texture) + x_ori_text
                x_text1 = x_texture
                l_fea = self.RG_fea[1]
                x_grad = l_fea(x_grad) + x_ori_grad
                x_b_fea1 = x_grad
                x_cat_1 = torch.cat([self.fusion1a(x_b_fea1), self.fusion1b(x_fea1), self.fusion1c(x_text1)], dim=1)        
                for j,bb in enumerate(self.b_block_1):
                    x_cat_1 = bb(x_cat_1)
                x_cat_1 = self.b_concat_1(x_cat_1) 

            if i==10:                
                x_fea2 = xx + self.internal_sum_fusion1_fea(x_cat_1)
                l_text = self.RG_text[2]
                x_texture = l_text(x_texture) + x_ori_text + self.internal_sum_fusion1_fea(x_cat_1)
                x_text2 = x_texture
                l_fea = self.RG_fea[2]
                x_grad = l_fea(x_grad) + x_ori_grad + self.internal_sum_fusion1_edge(x_cat_1)
                x_b_fea2 = x_grad
                
                x_cat_2 = torch.cat([self.fusion2a(x_b_fea2),self.fusion2b(x_cat_1), self.fusion2c(x_fea2), self.fusion2d(x_text2)], dim=1)        
                for j,bb in enumerate(self.b_block_2):
                    x_cat_2 = bb(x_cat_2)
                x_cat_2 = self.b_concat_2(x_cat_2)
            if i==15:
                x_fea3 = xx+ self.internal_sum_fusion2_fea(x_cat_2)
                l_text = self.RG_text[3]
                x_texture = l_text(x_texture) +x_ori_text+ self.internal_sum_fusion2_fea(x_cat_2)
                x_text3 = x_texture
                l_fea = self.RG_fea[3]
                x_grad = l_fea(x_grad) + x_ori_grad + self.internal_sum_fusion2_edge(x_cat_2)
                x_b_fea3 = x_grad
                x_cat_3 = torch.cat([self.fusion3a(x_b_fea3),self.fusion3b(x_cat_2), self.fusion3c(x_fea3), self.fusion3d(x_text3)], dim=1)        
                for j,bb in enumerate(self.b_block_3):
                    x_cat_3 = bb(x_cat_3)
                x_cat_3 = self.b_concat_3(x_cat_3) 
            if i==19:
                x_fea4 = xx+ self.internal_sum_fusion3_fea(x_cat_3)
                l_text = self.RG_text[4]
                x_texture = l_text(x_texture) +x_ori_text + self.internal_sum_fusion3_fea(x_cat_3)
                x_text4 = x_texture
                l_fea = self.RG_fea[4]
                x_grad = l_fea(x_grad) + x_ori_grad + self.internal_sum_fusion3_edge(x_cat_3)
                x_b_fea4 = x_grad
                x_cat_4 = torch.cat([self.fusion4a(x_b_fea4),self.fusion4b(x_cat_3), self.fusion4c(x_fea4), self.fusion4d(x_text4)], dim=1)
                for j,bb in enumerate(self.b_block_4):
                    x_cat_4 = bb(x_cat_4)
                x_cat_4 = self.b_concat_4(x_cat_4) 
           
        res = self.non_local1(x_cat_4)
        res_x_text4 = self.non_local2(x_text4)
        res_x_b_fea4 = self.non_local3(x_b_fea4)       

        res = self.sum1(res)+ self.top5(x) + self.sum3(res_x_text4) + self.sum4(res_x_b_fea4)
        for j,bb in enumerate(self.tail_res_LRG):
            res = bb(res)
        x_out = self.tail_res(res)
        for j,bb in enumerate(self.tail_text):
            res_x_text4 = bb(res_x_text4)
        x_out_res_x_text4 = self.tail_text2(res_x_text4)
        for j,bb in enumerate(self.tail_cat):
            x_cat_4 = bb(x_cat_4)
        x_out_x_cat_4 = self.tail_cat2(x_cat_4)
        for j,bb in enumerate(self.tail_fea):
            res_x_b_fea4 = bb(res_x_b_fea4)
        x_b_fea_top = self.tail_fea2(res_x_b_fea4)

        x_out = self.top4(self.AtMp4(x_out)) + self.top1(self.AtMp1(x_out_res_x_text4)) + self.top2(self.AtMp2(x_out_x_cat_4)) + self.top3(self.AtMp3(x_b_fea_top))
        for j,bb in enumerate(self.tail_top):
            x_out = bb(x_out)

        x_out = x_out*127.5 + self.rgb_mean.cuda()*255

        return x_out, total_loss