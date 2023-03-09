import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from networks.common import MeanShift, default_conv, UpSampler


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNetSR(nn.Module):

    def __init__(self, n_colors, sr_scale, n_feats, n_blocks, res_scale, mean=None, std=None):
        super(ConvNetSR, self).__init__()

        self.input_channel = n_colors
        self.res_scale = res_scale

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

        m_head = [default_conv(self.input_channel, n_feats, kernel_size=3)]

        m_blocks = [Block(n_feats) for _ in range(n_blocks)]

        m_tail = [UpSampler(default_conv, sr_scale, n_feats, act=None),
            default_conv(n_feats, n_colors, 3)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_blocks)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        fn = x
        x = self.body(x)

        x += fn * self.res_scale

        x = self.tail(x)
        return x


def ConvNetSR_model_large(paras, mean=None, std=None):
    n_colors = paras.input_channel
    sr_scale = int(paras.sr_scale)
    model = ConvNetSR(n_colors, sr_scale, 192, 32, 1., mean, std)
    return model


def ConvNetSR_model_lite(paras, mean=None, std=None):
    n_colors = paras.input_channel
    sr_scale = int(paras.sr_scale)
    model = ConvNetSR(n_colors, sr_scale, 64, 16, 1., mean, std)
    return model


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


