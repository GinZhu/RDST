from networks.common import MeanShift

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from os.path import exists

"""
VGG loss by the pre-trained model in pytorch
"""


class VGG(nn.Module):
    """
    conv_index: '22' or '54'

    rgb_range: for now useless. if want to use this should modify the code
    Behaviours:
        1. loss_names
        2. return [loss, loss_names, loss_item]
    """
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()

        vgg_path = 'loss/vgg19.pt'
        if exists(vgg_path):
            vgg_features = models.vgg19(pretrained=False)
            vgg_features.load_state_dict(torch.load(vgg_path))
        else:
            vgg_features = models.vgg19(pretrained=True)
        vgg_features = vgg_features.features

        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(vgg_mean, vgg_std, 'sub')
        self.vgg.requires_grad = False

        self.loss_names = ['VGG{}'.format(conv_index)]

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        if sr.shape != hr.shape:
            raise ValueError(
                'SR shape {} should be the same as HR shape {}'.format(
                    sr.shape, hr.shape
                ))
        if sr.size(1) != 3 and sr.size(1) != 1:
            loss = 0.
            for c in range(sr.size(1)):
                c_sr = torch.stack([sr[:, c, :, :]] * 3, dim=1)
                c_hr = torch.stack([hr[:, c, :, :]] * 3, dim=1)
                vgg_sr = _forward(c_sr)
                with torch.no_grad():
                    vgg_hr = _forward(c_hr)
                loss += F.mse_loss(vgg_sr, vgg_hr)
            return loss, {self.loss_names[0]: loss.item()}

        elif sr.size(1) == 1:
            sr = torch.cat([sr]*3, dim=1)
            hr = torch.cat([hr]*3, dim=1)

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr)

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss, {self.loss_names[0]: loss.item()}

