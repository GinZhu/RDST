from torch import nn
from networks.common import default_conv, MeanShift, ResBlock, UpSampler


class SRResNet(nn.Module):
    def __init__(self, paras, mean=None, std=None, feature_maps_only=False):
        super(SRResNet, self).__init__()

        self.sr_scale = int(paras.sr_scale)
        self.input_channel = paras.input_channel
        self.n_feats = paras.srresnet_n_feats
        self.res_scale = paras.srresnet_res_scale
        self.n_resblocks = paras.srresnet_n_resblocks
        # ## bn
        self.bn = paras.srresnet_bn
        # ## act
        self.act = paras.srresnet_act

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

        # ## define input layer
        m_head = [default_conv(
            self.input_channel, self.n_feats, kernel_size
        )]

        # define Res Blocks module
        m_body = [
            ResBlock(
                default_conv, self.n_feats, kernel_size, act=act, res_scale=self.res_scale, bn=self.bn
            ) for _ in range(self.n_resblocks)]
        m_body.append(default_conv(self.n_feats, self.n_feats, kernel_size))

        # define tail module
        if self.sr_scale > 1:
            m_tail = [UpSampler(default_conv, self.sr_scale, self.n_feats, act=None, bn=self.bn)]
        else:
            m_tail = []
        m_tail.append(default_conv(self.n_feats, self.input_channel, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if not self.feature_maps_only:
            x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.feature_maps_only:
            return res

        x = self.tail(res)
        x = self.add_mean(x)

        return x


