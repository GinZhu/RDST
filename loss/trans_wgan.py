import torch.nn as nn
import torch
from networks.swin_transformer_sr import BasicLayer, PatchEmbed, PatchUnEmbed
from networks.common import BasicBlock
import torch.nn.functional as F


def make_STD(paras):
    gan_type = paras.gan_type
    in_channels = paras.input_channel

    # swin-transformer related parameters
    basic_dim = paras.stgan_dim
    input_resolution = paras.stgan_input_resolution
    num_heads = paras.stgan_num_heads
    depth = paras.stgan_depth
    window_size = paras.stgan_window_size
    downsample = paras.stgan_downsample

    d_act = paras.d_act

    D = STDiscriminator(
        gan_type=gan_type,
        in_channels=in_channels,
        basic_dim=basic_dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        depth=depth,
        window_size=window_size,
        downsample=downsample,
        d_act=d_act
    )
    return D


class RSTB4GAN(nn.Module):
    """
    SwinTransformer-based basic block for GANs.

    Including:
        1. Swin layer * r
        2. if downsample,
            PatchMerging
            Conv2D with stride=2
    Input: N x C x H x W
    output: N x C*2 x H/2 x W/2 if downsampling
            N x C x H x W if no down-sampling

    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',
                 conv_bn=False, conv_act=None):
        super(RSTB4GAN, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # ## body blocks
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=None,
                                         use_checkpoint=use_checkpoint)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # ## downsample if possible
        self.downsample = downsample
        if downsample in ['conv']:
            self.downsample_layer = BasicBlock(
                dim, dim * 2, 3, stride=2, bn=conv_bn, act=conv_act)
        elif downsample in ['patchmerging']:
            self.downsample_layer = PatchMerging(dim=dim)
            self.patch_unembed_after_downsample = PatchUnEmbed(
                embed_dim=2*dim, norm_layer=None
            )
        else:
            self.downsample_layer = nn.Identity()

        # embed: N x C x H x W -> N x P x C(D)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        # unembed: N x P x C(D) -> N x C x H x W
        self.patch_unembed = PatchUnEmbed(
            embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        # x: N x C x H x W
        x = self.patch_embed(x)
        x = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

        if self.downsample in ['conv']:
            x = self.patch_unembed(x, x_size)
            return self.downsample_layer(x)
        elif self.downsample in ['patchmerging']:
            x = self.downsample_layer(x, x_size)
            d_x_size = (x_size[0] // 2, x_size[1] // 2)
            x = self.patch_unembed_after_downsample(x, d_x_size)
            return x
        else:
            return self.patch_unembed(x, x_size)


class STDiscriminator(nn.Module):
    """
    This is more like SwinIR:
        Conv head
        Using conv2d+stride for down-sampling
    """
    def __init__(self, gan_type, in_channels, basic_dim, input_resolution,
                 num_heads, depth, window_size, downsample, d_act='leaky_relu'):
        super(STDiscriminator, self).__init__()

        # ## parameters
        self.gan_type = gan_type
        in_channels = in_channels

        # swin-transformer related parameters
        self.basic_dim = basic_dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.depth = depth
        self.window_size = window_size
        self.downsample = downsample

        if 'GP' in self.gan_type:
            bn = False
        else:
            bn = True

        if d_act == 'relu':
            act = nn.ReLU(inplace=False)
        elif d_act == 'leaky_relu':
            act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        # ## head
        self.head = BasicBlock(in_channels, self.basic_dim, 3, bn=bn, act=act)

        # ## body
        dim = self.basic_dim
        input_size = self.input_resolution
        self.blocks = nn.ModuleList()
        for h, d, w in zip(self.num_heads, self.depth, self.window_size):
            self.blocks.append(
                RSTB4GAN(
                    dim=dim, input_resolution=input_size,
                    depth=d, num_heads=h, window_size=w,
                    downsample=self.downsample
                )
            )
            dim *= 2
            input_size = (input_size[0] // 2, input_size[1] //2)

        # ## tail
        m_classifier = [
            nn.Linear(dim * input_size[0] * input_size[1], 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        x, ori_size = self.pad(x)
        x = self.head(x)

        x_size = self.input_resolution
        for b in self.blocks:
            x = b(x, x_size)
            x_size = (x_size[0] // 2, x_size[1] // 2)

        output = self.classifier(x.contiguous().view(x.size(0), -1))

        return output

    def pad(self, x):
        h, w = x.shape[-2:]
        if h == self.input_resolution[0] and w == self.input_resolution[1]:
            pass
        else:
            ph = self.input_resolution[0] - h
            pw = self.input_resolution[1] - w
            x = F.pad(x, [0, pw, 0, ph], 'replicate')
        return x, [h, w]

    def ipad(self, x, ori_size):
        ori_h, ori_w = ori_size
        return x[:, :, :ori_h, :ori_w]


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C -> B, H/2 * W/2, 2*C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"
    #
    # def flops(self):
    #     H, W = self.input_resolution
    #     flops = H * W * self.dim
    #     flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
    #     return flops


