from torch import nn
from networks.common import default_conv, MeanShift, ResBlock, UpSampler


class MDSR(nn.Module):
    def __init__(self, paras, mean=None, std=None, feature_maps_only=False):
        super(MDSR, self).__init__()

        self.sr_scales = [2, 3, 4]
        self.input_channel = paras.input_channel
        self.n_feats = paras.mdsr_n_feats
        self.res_scale = paras.mdsr_res_scale
        self.n_resblocks = paras.mdsr_n_resblocks
        # ## bn
        self.bn = paras.mdsr_bn
        # ## act
        self.act = paras.mdsr_act

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
        else:
            raise ValueError('activation should be either relu or leaky_relu')

        kernel_size = 3

        # ## define input layers
        m_input = [default_conv(
            self.input_channel, self.n_feats, kernel_size
        )]
        self.input_layer = nn.Sequential(*m_input)

        # ## define head layers for each sr_scale
        self.head_2 = nn.Sequential(default_conv(
            self.input_channel, self.n_feats, kernel_size
        ))
        self.head_3 = nn.Sequential(default_conv(
            self.input_channel, self.n_feats, kernel_size
        ))
        self.head_4 = nn.Sequential(default_conv(
            self.input_channel, self.n_feats, kernel_size
        ))

        # ## define Res Blocks module
        m_body = [
            ResBlock(
                default_conv, self.n_feats, kernel_size, act=act, res_scale=self.res_scale, bn=self.bn
            ) for _ in range(self.n_resblocks)]
        m_body.append(default_conv(self.n_feats, self.n_feats, kernel_size))
        self.body = nn.Sequential(*m_body)

        self.tail_2 = nn.Sequential(
            UpSampler(default_conv, 2, self.n_feats, act=None),
            default_conv(self.n_feats, self.input_channel, kernel_size)
        )
        self.tail_3 = nn.Sequential(
            UpSampler(default_conv, 3, self.n_feats, act=None),
            default_conv(self.n_feats, self.input_channel, kernel_size)
        )
        self.tail_4 = nn.Sequential(
            UpSampler(default_conv, 4, self.n_feats, act=None),
            default_conv(self.n_feats, self.input_channel, kernel_size)
        )

    def forward(self, x, sr_scale):
        if not self.feature_maps_only:
            x = self.sub_mean(x)

        if sr_scale == 2.0:
            x = self.head_2(x)
        elif sr_scale == 3.0:
            x = self.head_3(x)
        elif sr_scale == 4.0:
            x = self.head_4(x)
        else:
            raise ValueError('Invalid sr_scale {}, should be one of [2.0, 3.0, 4.0]'.format(sr_scale))

        res_global = self.body(x)
        res_global += x

        if self.feature_maps_only:
            return res_global

        if sr_scale == 2.0:
            x = self.tail_2(res_global)
        elif sr_scale == 3.0:
            x = self.tail_3(res_global)
        elif sr_scale == 4.0:
            x = self.tail_4(res_global)
        else:
            raise ValueError('Invalid sr_scale {}, should be one of [2.0, 3.0, 4.0]'.format(sr_scale))

        x = self.add_mean(x)

        return x


