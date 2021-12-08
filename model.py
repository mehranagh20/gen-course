import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetowrk, AdaptiveInstanceNorm
from collections import defaultdict
import numpy as np
import itertools


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class ModelBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.adaIN = AdaptiveInstanceNorm(H.width, H.latent_dim)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x, w):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base // self.mixin)
        x = self.adaIN(x, w)
        x = self.resnet(x)
        return x


class Model(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.dci_db = None
        self.H = H
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim, n_mlp=H.n_mpl)
        resos = set()
        architecture = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.architecture)
        for idx, (res, mixin) in enumerate(blocks):
            architecture.append(ModelBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.architecture = nn.ModuleList(architecture)
        first_res = self.resolutions[0]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(H.width, H.image_channels)
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))

    def forward(self, latent_code):
        w = self.mapping_network(latent_code)[0]
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)
        for idx, block in enumerate(self.architecture):
            x = block(x, w)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x
