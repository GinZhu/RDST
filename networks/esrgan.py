from torch import nn
from networks.common import default_conv, MeanShift, UpSampler
from networks.common import ResidualRDB
import torch


class ESRGAN(nn.Module):
    def __init__(self, paras, mean=None, std=None, feature_maps_only=False):
        super(ESRGAN, self).__init__()

        self.sr_scale = int(paras.sr_scale)
        self.input_channel = paras.input_channel
        self.growth_rate = paras.esrgan_growth_rate
        self.n_dense_layers = paras.esrgan_n_dense_layers
        self.n_rdb = paras.esrgan_n_rdb
        self.n_blocks = paras.esrgan_n_blocks
        self.dense_layer_scale = paras.esrgan_dense_scale
        self.rdb_res_scale = paras.esrgan_rdb_res_scale
        self.rrdb_res_scale = paras.esrgan_rrdb_res_scale
        self.global_res_scale = paras.esrgan_global_res_scale
        self.n_feats = paras.esrgan_n_feats
        # ## bn
        self.bn = paras.esrgan_bn
        # act
        self.act = paras.esrgan_act

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
        if self.act == 'relu':
            act = nn.ReLU(True)
        elif self.act == 'leaky_relu':
            slope = paras.esrgan_leaky_relu_slope
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

        # dense blocks
        m = [ResidualRDB(
            default_conv, self.n_feats, self.growth_rate, kernel_size, bn=self.bn,
            act=act, dense_scale=self.dense_layer_scale, n_dense_layers=self.n_dense_layers,
            rdb_res_scale=self.rdb_res_scale, rrdb_res_scale=self.rrdb_res_scale, n_rdb=self.n_rdb
        ) for _ in range(self.n_blocks)]
        m.append(default_conv(self.n_feats, self.n_feats, kernel_size))

        self.body = nn.Sequential(*m)

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

        res = self.body(x).mul(self.global_res_scale)

        res += x

        if self.feature_maps_only:
            return res

        x = self.tail(res)
        x = self.add_mean(x)

        return x
