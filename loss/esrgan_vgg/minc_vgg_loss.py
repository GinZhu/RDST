from loss.esrgan_vgg.minc_vgg19_nets import VGG19

import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import exists

"""
VGG loss by the pre-trained model in pytorch
"""


class MincVGG(nn.Module):
    """
    conv_index: '22' or '54'

    rgb_range: for now useless. if want to use this should modify the code
    Behaviours:
        1. loss_names
        2. return [loss, loss_names, loss_item]
    """
    def __init__(self, mode='Minc_VGG22', pre_activation=True, model_path='loss/minc_vgg19.pt'):
        super(MincVGG, self).__init__()

        if not exists(model_path):
            raise ValueError('Pre_trained model must be valid.')

        self.vgg = VGG19(mode=mode, pre_activation=pre_activation)
        self.vgg.load_state_dict(torch.load(model_path), strict=False)

        self.loss_names = [mode]

    def forward(self, sr, hr):
        def _forward(x):
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

