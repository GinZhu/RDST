from torch import nn
from networks.common import default_conv, MeanShift, ResBlock, UpSampler
from networks.meta_upscale import MetaUpSampler


class EDSR(nn.Module):
    def __init__(self, paras, mean=None, std=None):
        super(EDSR, self).__init__()

        self.sr_scale = int(paras.sr_scale)
        self.input_channel = paras.input_channel
        self.n_feats = paras.edsr_n_feats
        self.res_scale = paras.edsr_res_scale
        self.n_resblocks = paras.edsr_n_resblocks
        # ## bn
        self.bn = paras.edsr_bn
        # ## act
        self.act = paras.edsr_act

        # ## mean shift layers
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

        # upscale
        self.scale_free = paras.scale_free
        if self.scale_free:
            self.tail = MetaUpSampler(self.n_feats, self.input_channel, 3)
        else:
            # define tail module
            if self.sr_scale > 1:
                m_tail = [UpSampler(default_conv, self.sr_scale, self.n_feats, act=None)]
            else:
                m_tail = []
            m_tail.append(default_conv(self.n_feats, self.input_channel, kernel_size))
            self.tail = nn.Sequential(*m_tail)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x, sr_scale=None):
        x = self.sub_mean(x)
        x = self.head(x)

        res_global = self.body(x)
        res_global += x

        # up-sample
        if self.scale_free:
            x = self.tail(res_global, sr_scale)
        else:
            x = self.tail(res_global)

        x = self.add_mean(x)

        return x


