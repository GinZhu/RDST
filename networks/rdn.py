from torch import nn
from networks.common import default_conv, MeanShift, UpSampler
from networks.common import ResidualDenseBlock
import torch

"""
[RDN]
rdn_growth_rate = 32
rdn_n_dense_layers = 6
rdn_n_blocks = 20
rdn_dense_scale = 1.0
rdn_bn = None
rdn_n_feats = 64
rdn_local_res_scale = 1.0
rdn_global_res_scale = 1.0
"""


class RDN(nn.Module):
    def __init__(self, paras, mean=None, std=None, feature_maps_only=False):
        super(RDN, self).__init__()

        self.sr_scale = int(paras.sr_scale)
        self.input_channel = paras.input_channel
        self.growth_rate = paras.rdn_growth_rate
        self.n_dense_layers = paras.rdn_n_dense_layers
        self.n_dense_blocks = paras.rdn_n_blocks
        self.dense_scale = paras.rdn_dense_scale
        self.local_res_scale = paras.rdn_local_res_scale
        self.global_res_scale = paras.rdn_global_res_scale
        self.n_feats = paras.rdn_n_feats
        # ## bn
        self.bn = paras.rdn_bn

        self.feature_maps_only = feature_maps_only

        # ## mean shift layers
        if not self.feature_maps_only:
            if mean is None:
                mean = [0. for _ in range(self.input_channel)]
            if std is None:
                std = [1. for _ in range(self.input_channel)]
            if len(mean) != len(std) or len(mean) != self.input_channel:
                raise ValueError(
                    'Dimension of mean {} / std {} should fit input channels {}'.format(
                        len(mean), len(std), self.input_channel
                    )
                )
            self.mean = mean
            self.std = std
            self.add_mean = MeanShift(mean, std, 'add')
            self.sub_mean = MeanShift(mean, std, 'sub')

        # ## act
        self.act = paras.act
        if self.act == 'relu':
            act = nn.ReLU(True)
        elif self.act == 'leaky_relu':
            slope = paras.leaky_relu_slope
            act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif self.act == 'prelu':
            act = nn.PReLU()
        else:
            raise ValueError('Invalid activation {}, should be one of [relu, leaky_relu, prelu]'.format(self.act))

        kernel_size = 3

        # input layer
        self.head = nn.Sequential(default_conv(
            self.input_channel, self.n_feats, kernel_size
        ))

        self.F0 = default_conv(self.n_feats, self.n_feats, kernel_size)

        # dense blocks
        self.body = nn.ModuleList()
        for i in range(self.n_dense_blocks):
            self.body.append(ResidualDenseBlock(
                default_conv, self.n_feats, self.growth_rate, kernel_size, act=act, bn=self.bn,
                dense_scale=self.dense_scale, n_dense_layers=self.n_dense_layers,
                res_scale=self.local_res_scale
            ))

        # bottleneck layer
        bottleneck_in_channels = self.n_feats * self.n_dense_blocks
        self.bottleneck = nn.Sequential(
            default_conv(bottleneck_in_channels, self.n_feats, 1),
            default_conv(self.n_feats, self.n_feats, kernel_size)
        )

        # upsample layers + reconstruciton layer
        if self.sr_scale > 1:
            m_tail = [UpSampler(default_conv, self.sr_scale, self.n_feats, act=None, bn=self.bn)]
        else:
            m_tail = []
        m_tail.append(default_conv(self.n_feats, self.input_channel, kernel_size))

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if not self.feature_maps_only:
            x = self.sub_mean(x)
        fn1 = self.head(x)
        x = self.F0(fn1)

        # RDB body
        feature_maps = []
        for block in self.body:
            x = block(x)
            feature_maps.append(x)

        feature_maps = torch.cat(feature_maps, 1)

        x = self.bottleneck(feature_maps).mul(self.global_res_scale)

        x += fn1

        if self.feature_maps_only:
            return x

        x = self.tail(x)
        x = self.add_mean(x)

        return x
