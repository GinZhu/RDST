from networks.common import default_conv, MeanShift, UpSampler
import torch
import torch.nn as nn


def rcan_make_model(paras, mean=None, std=None):
    n_colors = paras.input_channel
    sr_scale = int(paras.sr_scale)
    model = RCAN(default_conv, n_colors, 10, 20, 64, 16, sr_scale, 1., mean, std)
    return model

def RCAN_make_model(n_colors, sr_scale, mean=None, std=None):
    model = RCAN(default_conv, n_colors, 10, 20, 64, 16, sr_scale, 1., mean, std)
    return model


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)

class Ada_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, category=2):
        super(Ada_conv, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, out_channels, 1,
            padding=0, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.category = category
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        # c = list(np.arange(0,1,1/self.category))
        # c += 1
        m_batchsize, C, height, width = x.size()
        mask = self.sigmoid(self.conv0(x.permute(0, 1, 3, 2).contiguous().view(m_batchsize, C, height, width)))
        # mask = self.sigmoid(self.conv0(x))
        mask = torch.where(mask < 0.5, torch.full_like(mask, 1), torch.full_like(mask, 0))
        # pdb.set_trace()
        out = self.conv1(x) * mask + self.conv2(x) * (1 - mask)
        return out


class ResAda_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, category=2):
        super(ResAda_conv, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, 1, 1,
            padding=0, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.category = category
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        # c = list(np.arange(0,1,1/self.category))
        # c += 1
        m_batchsize, C, height, width = x.size()
        mask = self.sigmoid(self.conv0(x))
        mask = torch.where(mask < 0.5, torch.full_like(mask, 1), torch.full_like(mask, 0))
        # pdb.set_trace()
        # mask = mask[mask<0.5].view(m_batchsize,C,height,width)
        out = self.conv1(x) * mask + self.conv2(x) * (1 - mask)
        return out + x


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            # modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            modules_body.append(Ada_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        # modules_body.append(Ada_conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, conv, n_colors, n_resgroups, n_resblocks, n_feats, reduction, scale, res_scale,
                 mean=None, std=None):
        super(RCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # ## mean shift layers
        if mean is None:
            mean = [0. for _ in range(n_colors)]
        if std is None:
            std = [1. for _ in range(n_colors)]
        if len(mean) != len(std) or len(mean) != n_colors:
            raise ValueError(
                'Dimension of mean {} / std {} should fit input channels {}'.format(
                    len(mean), len(std), n_colors
                )
            )
        self.mean = mean
        self.std = std
        self.add_mean = MeanShift(mean, std, 'add')
        self.sub_mean = MeanShift(mean, std, 'sub')

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            UpSampler(conv, scale, n_feats, act=None),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
