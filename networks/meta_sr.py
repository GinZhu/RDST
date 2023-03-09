from networks.common import MeanShift, ResBlock
from networks.edsr import EDSR
from networks.srresnet import SRResNet
from networks.srdensenet import SRDenseNet
from networks.rdn import RDN
from networks.esrgan import ESRGAN
from networks.mdsr import MDSR
from networks.meta_upscale import MetaUpSampler
import math
import torch.nn as nn
import torch


class MetaSR(nn.Module):

    def __init__(self, paras, mean=None, std=None):
        super(MetaSR, self).__init__()

        self.all_sr_scales = paras.all_sr_scales
        self.n_colors = paras.input_channel

        self.feature_extractor_mode = paras.feature_generator
        if self.feature_extractor_mode == 'EDSR':
            self.feature_extractor = EDSR(paras, feature_maps_only=True)
        elif self.feature_extractor_mode == 'SRResNet':
            self.feature_extractor = SRResNet(paras, feature_maps_only=True)
        elif self.feature_extractor_mode == 'SRDenseNet':
            self.feature_extractor = SRDenseNet(paras, feature_maps_only=True)
        elif self.feature_extractor_mode == 'RDN':
            self.feature_extractor = RDN(paras, feature_maps_only=True)
        elif self.feature_extractor_mode == 'ESRGAN':
            self.feature_extractor = ESRGAN(paras, feature_maps_only=True)
        elif self.feature_extractor_mode == 'Meta_MDSR':
            self.feature_extractor = MDSR(paras, feature_maps_only=True)
        # todo: add more potential generators here
        else:
            raise ValueError('LR feature maps extractor should be one of [EDSR, SRResNet, UNet')
        self.n_feats = self.feature_extractor.n_feats

        # ## mean shift layers
        # ##
        if mean is None:
            mean = [0. for _ in range(self.n_colors)]
        if std is None:
            std = [1. for _ in range(self.n_colors)]
        if len(mean) != len(std) or len(mean) != self.n_colors:
            raise ValueError(
                'Dimension of mean {} / std {} should fit input channels {}'.format(
                    len(mean), len(std), self.n_colors
                )
            )
        self.mean = mean
        self.std = std
        self.add_mean = MeanShift(mean, std, 'add')
        self.sub_mean = MeanShift(mean, std, 'sub')

        # ##
        self.meta_upsampler = MetaUpSampler(
            self.n_feats, self.n_colors, paras.meta_sr_kernel_size
        )

        # ## transfer learning of feature extractor
        if paras.pre_trained_f:
            # TL
            ptm = torch.load(paras.pre_trained_f)
            self.feature_extractor.load_state_dict(ptm, strict=False)
            # Freeze layers or not
            self.feature_extractor.requires_grad_(paras.train_meta_feature_extractor)

    def forward(self, x, sr_scale):
        x = self.sub_mean(x)

        if self.feature_extractor_mode == 'Meta_MDSR':
            lr_features = self.feature_extractor(x, math.ceil(sr_scale))
        else:
            lr_features = self.feature_extractor(x)

        out = self.meta_upsampler(lr_features, sr_scale)

        out = self.add_mean(out)

        return out

