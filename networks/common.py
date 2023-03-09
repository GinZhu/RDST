import torch
from torch import nn
import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias
    )


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class DenseLayer(nn.Module):
    def __init__(self, conv, in_channels, growth_rate, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True), dense_scale=1.):
        super(DenseLayer, self).__init__()

        m = [conv(in_channels, growth_rate, kernel_size, bias)]

        if bn:
            m.append(nn.BatchNorm2d(growth_rate))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.dense_scale = dense_scale

    def forward(self, x):
        dense = self.body(x).mul(self.dense_scale)
        dense = torch.cat((x, dense), 1)

        return dense


class DenseBlock(nn.Module):
    def __init__(self, conv, in_channels, growth_rate, kernel_size, bias=True, bn=False, act=nn.ReLU(True),
                 dense_scale=1., n_dense_layers=8):
        super(DenseBlock, self).__init__()

        m = []
        for i in range(int(n_dense_layers)):
            m.append(DenseLayer(conv, in_channels, growth_rate, kernel_size, bias, bn, act, dense_scale))
            in_channels += growth_rate

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualDenseBlock(DenseBlock):
    def __init__(self, conv, in_channels, growth_rate, kernel_size, bias=True, bn=False, act=nn.ReLU(True),
                 dense_scale=1., n_dense_layers=8, res_scale=1.0):
        super(ResidualDenseBlock, self).__init__(
            conv, in_channels, growth_rate, kernel_size, bias, bn, act, dense_scale, n_dense_layers)

        n_feats = in_channels + growth_rate * n_dense_layers
        self.bottle_neck = conv(n_feats, in_channels, 1)

        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.bottle_neck(res).mul(self.res_scale)
        res += x
        return res


class ResidualRDB(nn.Module):
    def __init__(self, conv, in_channels, growth_rate, kernel_size, bias=True, bn=False,
                 act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 dense_scale=1., n_dense_layers=4, rdb_res_scale=0.2,
                 rrdb_res_scale=0.2, n_rdb=3):
        super(ResidualRDB, self).__init__()
        self.res_scale = rrdb_res_scale

        m = []
        for i in range(int(n_rdb)):
            m.append(ResidualDenseBlock(
                conv, in_channels, growth_rate, kernel_size, bias, bn, act, dense_scale, n_dense_layers, rdb_res_scale
            ))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class UpSampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=None, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    m.append(act)

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                m.append(act)
        else:
            raise NotImplementedError('SR scale {} is not valid.'.format(scale))

        super(UpSampler, self).__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, mean=(0.,), std=(1.0,), mode='sub'):
        if len(mean) != len(std):
            raise ValueError('Size of means and stds should be the same')
        nc = len(mean)

        super(MeanShift, self).__init__(nc, nc, kernel_size=1)

        std = torch.Tensor(std)
        if mode == 'sub':
            self.weight.data = torch.eye(nc).view(nc, nc, 1, 1) / std.view(nc, 1, 1, 1)
            self.bias.data = -1 * torch.Tensor(mean) / std
        elif mode == 'add':
            self.weight.data = torch.eye(nc).view(nc, nc, 1, 1) * std.view(nc, 1, 1, 1)
            self.bias.data = 1 * torch.Tensor(mean)
        for p in self.parameters():
            p.requires_grad = False


class WeightsInitializer(object):

    def __init__(self, act='relu', leaky_relu_slope=0.01):
        assert act in ['relu', 'leaky_relu', 'tanh'], 'Activation {} not support.'.format(act)
        self.act = act

        self.leaky_relu_slope = leaky_relu_slope

    def __call__(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if self.act == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=self.act)
                elif self.act == 'leaky_relu':
                    nn.init.kaiming_normal_(m.weight, a=self.leaky_relu_slope,
                                            mode='fan_in', nonlinearity=self.act)
                elif self.act == 'tanh':
                    nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                # nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
