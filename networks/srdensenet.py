from torch import nn
from networks.common import default_conv, MeanShift, ResBlock, UpSampler
from networks.common import DenseBlock
import torch


class SRDenseNet(nn.Module):
    def __init__(self, paras, mean=None, std=None, feature_maps_only=False):
        super(SRDenseNet, self).__init__()

        self.sr_scale = int(paras.sr_scale)
        self.input_channel = paras.input_channel
        self.growth_rate = paras.srdensenet_growth_rate
        self.n_dense_layers = paras.srdensenet_n_dense_layers
        self.n_dense_blocks = paras.srdensenet_n_dense_blocks
        self.type = paras.srdensenet_type
        self.dense_scale = paras.srdensenet_dense_scale
        self.n_feats = paras.srdensenet_n_feats
        self.bn = paras.srdensenet_bn
        # ## act
        self.act = paras.srdensenet_act

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
            self.input_channel, self.growth_rate, kernel_size
        ))

        # dense blocks
        self.body = nn.ModuleList()
        n_out_feats_of_blocks = []
        in_channels = self.growth_rate
        for i in range(self.n_dense_blocks):
            self.body.append(DenseBlock(
                default_conv, in_channels, self.growth_rate, kernel_size, act=act, bn=self.bn,
                dense_scale=self.dense_scale, n_dense_layers=self.n_dense_layers
            ))
            in_channels += self.growth_rate * self.n_dense_layers
            n_out_feats_of_blocks.append(in_channels)

        # bottleneck layer
        assert self.type in ['h', 'hl', 'all'], 'Invalid SRDenseNet type: {}, one of [h, hl, all]'.format(self.type)
        if self.type == 'h':
            bottleneck_in_channels = n_out_feats_of_blocks[-1]
        elif self.type == 'hl':
            bottleneck_in_channels = self.growth_rate + n_out_feats_of_blocks[-1]
        elif self.type == 'all':
            bottleneck_in_channels = self.growth_rate + sum(n_out_feats_of_blocks)
        self.bottleneck = default_conv(bottleneck_in_channels, self.n_feats, 1)

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
        x = self.head(x)

        feature_maps = []
        if self.type in ['hl', 'all']:
            feature_maps.append(x)
        for block in self.body:
            x = block(x)
            if self.type == 'all':
                feature_maps.append(x)
        if self.type in ['h', 'hl']:
            feature_maps.append(x)

        feature_maps = torch.cat(feature_maps, 1)

        x = self.bottleneck(feature_maps)

        if self.feature_maps_only:
            return x

        x = self.tail(x)
        x = self.add_mean(x)

        return x
